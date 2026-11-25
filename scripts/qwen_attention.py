"""Attention from each answer to the video, for calls of a finished Qwen3-VL vLLM run.

Teacher-forced: one forward pass over (exact frozen prompt + the clip vLLM saw + the model's
saved response), no generation. For every answer value in the response JSON it records, per text
layer and averaged over heads, the share of attention its tokens put on video tokens, on the
prompt before the video (system prompt and chat header, where attention sinks sit), on the rest of
the prompt (timestamps, questions), and on earlier response tokens. Output: one JSON line per
field. Calls already in OUT are skipped, so a stopped run resumes by rerunning the same command.
Calls run longest clip first, so an out-of-memory failure shows up at the start.

    python scripts/qwen_attention.py RUN_DIR OUT.jsonl --model Qwen/Qwen3-VL-32B-Instruct \
        [--call-id ID ...] [--limit N] [--min-seconds S] [--max-seconds S] [--shard I/N] \
        [--mlp-chunk TOKENS] [--max-gpu-mem GIB]

For prompts near 230k tokens (clips of 11-17 min) add --mlp-chunk 32768 --max-gpu-mem 22: the
text MLP then runs over the sequence in slices (position-wise, so the result is the same up to
bf16 kernel rounding) and weights are spread evenly over the GPUs, leaving room for activations.

Run on the machine that holds the run directory and the videos (paths are absolute in the
snapshot). Needs torch, transformers with Qwen3-VL, and a GPU; the model must be the one that
produced the run, otherwise the response being scored is not its own.

Checks to read in the output before trusting it: ``prompt_tokens`` should match
``recorded_input_tokens`` (vLLM's count) up to a few tokens; a large gap means the processor
built a different video input (resolution, frame count or timestamps) than the server did.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

CHUNK = 16  # answer tokens scored at once; scores are heads x CHUNK x seq_len floats


def answer_spans(text: str, field_ids: list[str]) -> dict[str, tuple[int, int]]:
    """Character span of each field's JSON value in the response (fields that are absent skip)."""
    dec = json.JSONDecoder()
    spans = {}
    for f in field_ids:
        key = text.find(json.dumps(f))
        if key < 0:
            continue
        start = text.index(":", key + len(json.dumps(f))) + 1
        while text[start].isspace():
            start += 1
        try:
            _, end = dec.raw_decode(text, start)
        except json.JSONDecodeError:
            continue
        spans[f] = (start, end)
    return spans


def query_positions(offsets, spans, n_prompt):
    """Response token positions used as attention queries, per field.

    Returns two dicts of absolute positions (prompt length added):
    * ``value``: tokens whose character offsets overlap the field's JSON value (the old measure).
    * ``pred``: the positions that *predict* those tokens, i.e. each value position minus one.
      In a causal LM the row at position p produces token p+1, so the choice of the value's first
      token (``true``, ``[``, the first digit) is made at the token before it, typically ``":``.
    """
    value = {
        f: [n_prompt + i for i, (a, b) in enumerate(offsets) if a < e and b > s]
        for f, (s, e) in spans.items()
    }
    pred = {f: [p - 1 for p in ps] for f, ps in value.items()}
    return value, pred


def clip_seconds(r) -> float:
    return sum(v.end_s - v.start_s for v in r.videos)


def load_calls(
    run: Path,
    call_ids: list[str] | None,
    limit: int | None,
    max_seconds: float | None = None,
    min_seconds: float | None = None,
    shard: tuple[int, int] = (0, 1),
    done: frozenset[str] = frozenset(),
) -> list[tuple]:
    from video_report.providers.base import ModelRequest
    from video_report.run_store import call_key

    reqs = [
        ModelRequest.from_json(json.loads(x))
        for x in (run / "snapshot/requests.jsonl").read_text().splitlines()
    ]
    out = []
    for i, r in sorted(enumerate(reqs), key=lambda x: -clip_seconds(x[1])):
        seconds = clip_seconds(r)
        if (
            (call_ids and r.call_id not in call_ids)
            or (max_seconds and seconds > max_seconds)
            or (min_seconds and seconds <= min_seconds)
            or i % shard[1] != shard[0]
            or r.call_id in done
        ):
            continue
        state = json.loads((run / "calls" / f"{call_key(r.call_id)}.json").read_text())
        if state["status"] != "succeeded":
            continue
        n = state["final_attempt"]
        resp = json.loads(
            (run / f"responses/{call_key(r.call_id)}/attempt-{n:04d}.json").read_text()
        )
        out.append((r, resp["response"]["text"], (resp["response"].get("usage") or {})))
        if limit and len(out) >= limit:
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("run", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--model", required=True)
    ap.add_argument("--call-id", action="append")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--max-seconds", type=float, help="skip calls whose clips are longer")
    ap.add_argument("--min-seconds", type=float, help="skip calls whose clips are not longer")
    ap.add_argument("--mlp-chunk", type=int, default=0, help="run text MLPs in slices (0 = off)")
    ap.add_argument("--max-gpu-mem", type=int, help="GiB of weights per GPU (balances layers)")
    ap.add_argument("--shard", default="0/1", help="I/N: take every N-th call, offset I")
    # Defaults mirror the k32 vLLM server (~/serve_tp4.sh): keep every frame of the pre-cut
    # 2 fps clip, never resize it. Check prompt_tokens vs recorded_input_tokens.
    ap.add_argument(
        "--processor-kwargs",
        default='{"do_sample_frames": false, "cap_pixels_per_frame": false, '
        '"size": {"longest_edge": 600000000, "shortest_edge": 4096}}',
    )
    args = ap.parse_args()

    import torch
    from transformers import AttentionInterface, AutoModelForImageTextToText, AutoProcessor
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    from transformers.video_utils import load_video

    from video_report.config import VLLMProviderConfig
    from video_report.providers.media import LocalMedia
    from video_report.providers.vllm import choose_height

    cfg = json.loads((args.run / "snapshot/config.json").read_text())
    opts = VLLMProviderConfig(**cfg["provider"].get("vllm", {}))
    media = LocalMedia(opts.max_height, opts.media_timeout_s)

    probe = {"on": False}  # set per call: query positions, key segment matrix, sink

    def attention(module, query, key, value, attention_mask, **kw):
        out = sdpa_attention_forward(module, query, key, value, attention_mask, **kw)
        layer = text_layers.get(id(module))
        if probe["on"] and layer is not None:
            hkv, seq, d = key.shape[1:]
            q = query[0].reshape(hkv, -1, seq, d)  # (kv heads, group, seq, d); HF repeat_kv order
            k = key[0]
            scale = kw.get("scaling") or d**-0.5
            seg = probe["seg"].to(k.device)
            res = []
            for c in probe["pos"].split(CHUNK):
                c = c.to(k.device)
                s = torch.einsum("hgcd,hld->hgcl", q[:, :, c].float(), k.float()) * scale
                s.masked_fill_(torch.arange(seq, device=k.device) > c[:, None], float("-inf"))
                res.append((s.softmax(-1) @ seg).mean((0, 1)))  # (chunk, segments)
            probe["sink"][layer] = torch.cat(res).cpu()
        return out

    AttentionInterface.register("answer_probe", attention)
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="auto",
        max_memory=(
            {i: f"{args.max_gpu_mem}GiB" for i in range(torch.cuda.device_count())}
            if args.max_gpu_mem
            else None
        ),
        attn_implementation="answer_probe",
    ).eval()
    if offloaded := {d for d in model.hf_device_map.values() if d in ("cpu", "disk")}:
        sys.exit(f"model does not fit on the GPUs (layers on {offloaded}); raise --max-gpu-mem")
    if args.mlp_chunk:
        for n, m in model.named_modules():
            if n.endswith(".mlp") and "visual" not in n:
                m.forward = lambda x, f=m.forward, c=args.mlp_chunk: torch.cat(
                    [f(x[:, i : i + c]) for i in range(0, x.shape[1], c)], 1
                )
    text_layers = {
        id(m): m.layer_idx
        for n, m in model.named_modules()
        if n.endswith("self_attn") and "visual" not in n
    }
    video_id = model.config.video_token_id
    tok = processor.tokenizer
    pkw = json.loads(args.processor_kwargs)
    shard = tuple(map(int, args.shard.split("/")))
    done = frozenset(
        json.loads(x)["call_id"]
        for x in (args.out.read_text().splitlines() if args.out.exists() else [])
    )
    calls = load_calls(
        args.run, args.call_id, args.limit, args.max_seconds, args.min_seconds, shard, done
    )  # type: ignore[arg-type]
    print(f"{len(calls)} calls to do ({len(done)} already in {args.out})", file=sys.stderr)

    with open(args.out, "a") as fh:
        for req, text, usage in calls:
            t0 = time.time()
            fps = req.fps or opts.frame_fps
            height = choose_height(req, opts)
            # decode ourselves: every frame of the pre-cut clip, like vLLM's num_frames=-1
            videos = [
                load_video(str(media.clip(v, fps, height)), backend="pyav") for v in req.videos
            ]
            messages = [{"role": "system", "content": req.system}] if req.system else []
            content = [{"type": "video"} for _ in videos]
            messages.append(
                {"role": "user", "content": [*content, {"type": "text", "text": req.user}]}
            )
            prompt = processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            enc = processor(
                text=[prompt],
                videos=[f for f, _ in videos],
                video_metadata=[m for _, m in videos],
                return_tensors="pt",
                **pkw,
            )
            resp = tok(text, add_special_tokens=False, return_offsets_mapping=True)
            n_prompt = enc["input_ids"].shape[1]
            ids = torch.cat([enc["input_ids"], torch.tensor([resp["input_ids"]])], 1)
            enc["input_ids"] = ids
            enc["attention_mask"] = torch.ones_like(ids)
            for k in ("mm_token_type_ids", "token_type_ids"):  # response tokens are text (0)
                if k in enc:
                    pad = torch.zeros(1, ids.shape[1] - n_prompt, dtype=enc[k].dtype)
                    enc[k] = torch.cat([enc[k], pad], 1)

            spans = answer_spans(text, list(req.field_ids))
            field_pos, pred_pos = query_positions(resp["offset_mapping"], spans, n_prompt)
            pos = sorted({p for d in (field_pos, pred_pos) for ps in d.values() for p in ps})
            is_video = ids[0] == video_id
            in_prompt = torch.arange(ids.shape[1]) < n_prompt
            before = torch.arange(ids.shape[1]) < int(is_video.nonzero()[0])
            rest = in_prompt & ~is_video & ~before
            seg = torch.stack([is_video, before, rest, ~in_prompt], 1).float()
            probe.update(on=True, pos=torch.tensor(pos), seg=seg, sink={})
            with torch.no_grad():
                model(**enc.to(model.device), logits_to_keep=1)
            probe["on"] = False

            index = {p: i for i, p in enumerate(pos)}
            n = len(req.field_ids)
            for i, f in enumerate(req.field_ids):
                rows = [index[p] for p in field_pos.get(f, [])]
                if not rows:
                    print(f"{req.call_id}: no answer span for {f}", file=sys.stderr)
                    continue
                prows = [index[p] for p in pred_pos[f]]
                layers, players = (
                    [probe["sink"][layer][r].mean(0).tolist() for layer in sorted(probe["sink"])]
                    for r in (rows, prows)
                )
                fh.write(
                    json.dumps(
                        {
                            "run": args.run.name,
                            "call_id": req.call_id,
                            "report_id": req.report_id,
                            "field_id": f,
                            "index": i,
                            "call_len": n,
                            "from_end": n - 1 - i,
                            "answer_tokens": len(rows),
                            "height": height,
                            "prompt_tokens": n_prompt,
                            "video_tokens": int(is_video.sum()),
                            "recorded_input_tokens": usage.get("input_tokens"),
                            # per text layer: [to video, to prompt before the video,
                            # to rest of prompt, to earlier response]
                            "attention": layers,
                            # same, but at the positions that predict the value tokens (shifted
                            # left by one; includes the decision token before the value)
                            "attention_pred": players,
                        }
                    )
                    + "\n"
                )
            fh.flush()
            print(
                f"{req.call_id}: {len(spans)}/{n} fields, prompt {n_prompt} tokens "
                f"(vLLM {usage.get('input_tokens')}), {time.time() - t0:.0f} s",
                file=sys.stderr,
            )
    media.close()


if __name__ == "__main__":
    main()
