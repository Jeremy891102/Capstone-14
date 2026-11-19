"""Attention intervention: regenerate a finished Qwen3-VL vLLM run with video attention scaled up.

For each call of the run, prefill once with HF Transformers, then decode once per condition from
the same KV cache (cropped back to the prompt after each condition). In the chosen layers, for the
tokens being generated only, log(alpha) is added to the attention scores on the target keys before
the softmax. The first generated token comes from the prefill and is not modified. Output: one JSON
line per (call, condition) with the generated text and where the attention went.

    python scripts/qwen_intervene.py RUN_DIR OUT.jsonl --model Qwen/Qwen3-VL-32B-Instruct \
        --conditions base,a1.5,a2,a3,all2,text2 [--call-id ID ...] [--limit N] \
        [--min-seconds S] [--max-seconds S] [--shard I/N]

Conditions (late = last quarter of the text layers):
    base       alpha = 1, same code path as the others (the baseline to compare against)
    a<alpha>   late layers, video keys
    all<alpha> all layers, video keys
    text<alpha> late layers, prompt-text keys (control: perturb attention on something that is not
               the video)

Use `base`, not vLLM's saved answers, as the baseline: the manual decode attention differs from
the fused kernel by bf16 rounding, enough to change a few greedy answers.
Rerunning the same command resumes: finished (call, condition) pairs are skipped.
For clips of 11-17 min add --mlp-chunk 32768 --max-gpu-mem 22 (as in qwen_attention.py).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qwen_attention import load_calls  # noqa: E402

MAX_NEW = 4096
PKW = {  # as qwen_attention.py: keep every frame of the pre-cut 2 fps clip, never resize it
    "do_sample_frames": False,
    "cap_pixels_per_frame": False,
    "size": {"longest_edge": 600000000, "shortest_edge": 4096},
}
VIDEO, BEFORE, TEXT, ANSWERS = range(4)  # columns of the key-segment matrix


def parse_conditions(spec: str) -> list[dict]:
    out = []
    for name in spec.split(","):
        if name == "base":
            out.append(dict(name=name, alpha=1.0, layers="late", target=VIDEO))
            continue
        m = re.fullmatch(r"(a|all|text)(\d+(?:\.\d+)?)", name)
        if not m or float(m[2]) <= 0:
            raise SystemExit(f"bad condition {name!r}: use base, a<alpha>, all<alpha>, text<alpha>")
        kind = m[1]
        out.append(
            dict(
                name=name,
                alpha=float(m[2]),
                layers="all" if kind == "all" else "late",
                target=TEXT if kind == "text" else VIDEO,
            )
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("run", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--model", required=True)
    ap.add_argument("--conditions", default="base,a2")
    ap.add_argument("--call-id", action="append")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--min-seconds", type=float)
    ap.add_argument("--max-seconds", type=float)
    ap.add_argument("--shard", default="0/1", help="I/N: take every N-th call, offset I")
    ap.add_argument("--mlp-chunk", type=int, default=0, help="run text MLPs in slices (0 = off)")
    ap.add_argument("--max-gpu-mem", type=int, help="GiB of weights per GPU (balances layers)")
    args = ap.parse_args()
    conds = parse_conditions(args.conditions)

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

    st: dict = {"on": False, "logalpha": 0.0, "layers": set(), "target": VIDEO, "seg": None}
    seg_on: dict = {}  # device -> key-segment matrix on that device
    acc: dict = {}  # device -> summed attention share per segment (decode steps, late layers)
    n_acc: dict = {}  # device -> number of summed (step, layer) pairs

    def attention(module, query, key, value, attention_mask, **kw):
        layer = layer_of.get(id(module))
        if not st["on"] or query.shape[2] != 1 or layer not in st["layers"]:
            return sdpa_attention_forward(module, query, key, value, attention_mask, **kw)
        hkv, length, d = key.shape[1:]
        dev = key.device
        if dev not in seg_on:
            seg_on[dev] = st["seg"].to(dev)
            acc[dev] = torch.zeros(4, device=dev)
            n_acc[dev] = 0
        seg = seg_on[dev][:length]
        q = query[0, :, 0].reshape(hkv, -1, d)  # (kv heads, group, d): HF repeat_kv head order
        s = torch.matmul(q, key[0].transpose(-1, -2)).float() * (kw.get("scaling") or d**-0.5)
        s = s + st["logalpha"] * seg[:, st["target"]]
        p = s.softmax(-1)
        if layer in stat_layers:
            acc[dev] += (p @ seg).mean((0, 1))
            n_acc[dev] += 1
        o = torch.matmul(p.to(value.dtype), value[0])  # (kv heads, group, d)
        return o.reshape(1, 1, -1, d), None

    AttentionInterface.register("intervene", attention)
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
        attn_implementation="intervene",
    ).eval()
    if offloaded := {d for d in model.hf_device_map.values() if d in ("cpu", "disk")}:
        sys.exit(f"model does not fit on the GPUs (layers on {offloaded}); raise --max-gpu-mem")
    layer_of = {
        id(m): m.layer_idx
        for n, m in model.named_modules()
        if n.endswith("self_attn") and "visual" not in n
    }
    n_layers = len(layer_of)
    layer_sets = {"late": set(range(n_layers * 3 // 4, n_layers)), "all": set(range(n_layers))}
    stat_layers = layer_sets["late"]
    if args.mlp_chunk:
        for n, m in model.named_modules():
            if n.endswith(".mlp") and "visual" not in n:
                m.forward = lambda x, f=m.forward, c=args.mlp_chunk: torch.cat(
                    [f(x[:, i : i + c]) for i in range(0, x.shape[1], c)], 1
                )
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    video_id = model.config.video_token_id
    tok = processor.tokenizer

    def build(req):
        fps = req.fps or opts.frame_fps
        height = choose_height(req, opts)
        videos = [load_video(str(media.clip(v, fps, height)), backend="pyav") for v in req.videos]
        msgs = [{"role": "system", "content": req.system}] if req.system else []
        content = [{"type": "video"} for _ in videos] + [{"type": "text", "text": req.user}]
        msgs.append({"role": "user", "content": content})
        prompt = processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        return processor(
            text=[prompt],
            videos=[f for f, _ in videos],
            video_metadata=[m for _, m in videos],
            return_tensors="pt",
            **PKW,
        ).to(model.device)

    def set_segments(input_ids, n_prompt):
        n = n_prompt + MAX_NEW
        pos = torch.arange(n)
        vid = torch.zeros(n, dtype=torch.bool)
        vid[:n_prompt] = input_ids[0, :n_prompt].cpu() == video_id
        before = pos < int(vid.nonzero()[0])
        in_prompt = pos < n_prompt
        st["seg"] = torch.stack([vid, before, in_prompt & ~vid & ~before, ~in_prompt], 1).float()
        seg_on.clear()

    def decode(cache, logits, cond):
        st.update(
            on=True,
            logalpha=math.log(cond["alpha"]),
            layers=layer_sets[cond["layers"]],
            target=cond["target"],
        )
        for dev in acc:
            acc[dev].zero_()
            n_acc[dev] = 0
        toks = []
        with torch.no_grad():
            for _ in range(MAX_NEW):
                t = logits.argmax(-1)
                if int(t) in eos:
                    break
                toks.append(int(t))
                out = model(input_ids=t[:, None], past_key_values=cache, use_cache=True)
                logits = out.logits[:, -1]
        st["on"] = False
        if toks:
            cache.crop(-len(toks))  # transformers 5: negative = number of tokens to drop
        # mean over decode steps and the late layers; segments: video, before video, text, answers
        n = sum(n_acc.values())
        share = (sum(a.cpu() for a in acc.values()) / max(n, 1)).tolist()
        late = share[VIDEO] / (share[VIDEO] + share[TEXT]) if n else None
        return tok.decode(toks, skip_special_tokens=True), len(toks), share, late

    done = set()
    if args.out.exists():
        for line in args.out.read_text().splitlines():
            r = json.loads(line)
            if "condition" in r:
                done.add((r["call_id"], r["condition"]))
    shard = tuple(map(int, args.shard.split("/")))
    calls = load_calls(
        args.run, args.call_id, args.limit, args.max_seconds, args.min_seconds, shard
    )
    print(f"{len(calls)} calls, conditions {[c['name'] for c in conds]}", file=sys.stderr)

    with open(args.out, "a") as fh:

        def write(rec):
            fh.write(json.dumps(rec) + "\n")
            fh.flush()

        for req, _, usage in calls:
            todo = [c for c in conds if (req.call_id, c["name"]) not in done]
            if not todo:
                continue
            try:
                enc = build(req)
                n_prompt = enc["input_ids"].shape[1]
                set_segments(enc["input_ids"], n_prompt)
                torch.cuda.synchronize()
                t0 = time.time()
                with torch.no_grad():
                    o = model(**enc, use_cache=True, logits_to_keep=1)
                cache, logits = o.past_key_values, o.logits[:, -1]
                del o
                torch.cuda.synchronize()
                prefill_s = round(time.time() - t0, 1)
                for c in todo:
                    t1 = time.time()
                    text, n_out, share, late = decode(cache, logits, c)
                    torch.cuda.synchronize()
                    assert cache.get_seq_length() == n_prompt, "cache not restored"
                    write(
                        {
                            "run": args.run.name,
                            "call_id": req.call_id,
                            "report_id": req.report_id,
                            "field_ids": list(req.field_ids),
                            "condition": c["name"],
                            "alpha": c["alpha"],
                            "layers": c["layers"],
                            "target": ["video", "before", "text", "answers"][c["target"]],
                            "prompt_tokens": n_prompt,
                            "recorded_input_tokens": usage.get("input_tokens"),
                            "out_tokens": n_out,
                            "prefill_s": prefill_s,
                            "decode_s": round(time.time() - t1, 1),
                            # generated tokens' attention, late layers: video, before, text, answers
                            "share": [round(x, 4) for x in share],
                            "late_video_share": None if late is None else round(late, 4),
                            "text": text,
                        }
                    )
                    print(f"{req.call_id} {c['name']}: late video {late}", file=sys.stderr)
                del cache, logits
            except Exception as e:  # an out-of-memory call is a result; keep going
                write({"call_id": req.call_id, "error": f"{type(e).__name__}: {str(e)[:300]}"})
                print(f"{req.call_id}: {type(e).__name__}", file=sys.stderr)
            torch.cuda.empty_cache()
    media.close()


if __name__ == "__main__":
    main()
