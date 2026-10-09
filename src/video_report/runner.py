"""Execute a run's planned calls: concurrency, retries, checkpoints, resume.

This module is the ONLY retry/scheduling layer. Providers perform one attempt each (SDK retries
are disabled) and classify failures; the runner decides whether to try again.

Per-attempt protocol (each call is owned by one worker thread at a time):

1. append attempt ``{outcome: in_progress}`` and set call ``status: in_progress`` (atomic write)
2. call the provider
3a. success: write ``responses/<key>/attempt-N.json`` atomically, THEN mark the attempt
    ``success`` and the call ``succeeded``. The raw response is on disk before anything parses it.
3b. failure: record the classified error. Permanent -> call ``failed``. Retryable -> back off and
    retry until ``max_attempts`` for this session is used, then ``failed``.

Crash recovery (on resume): a call left ``in_progress`` either has its response file (crash
between 3a's two writes -> recovered as success, no new request) or not (attempt marked
``interrupted``: the remote call may have run and been billed; the call is re-queued). This
means a crash after remote success but before step 3a's first write causes a duplicate request
on resume: execution is at-least-once, not exactly-once.
"""

from __future__ import annotations

import os
import random
import socket
import threading
import time
from collections import Counter
from collections.abc import Callable
from concurrent.futures import FIRST_EXCEPTION, Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass, replace
from typing import Any

from video_report.config import ExecutionConfig
from video_report.io_utils import utc_now_iso
from video_report.providers.base import (
    ModelRequest,
    Provider,
    ProviderError,
    ProviderTimeoutError,
    redact,
)
from video_report.run_log import export_run_log
from video_report.run_store import RunDir

TERMINAL_OK = "succeeded"


@dataclass(frozen=True)
class ExecutionPolicy:
    concurrency: int
    timeout_s: float
    max_attempts: int
    backoff_initial_s: float
    backoff_max_s: float
    backoff_jitter: float

    @staticmethod
    def from_config(cfg: ExecutionConfig, **overrides: Any) -> ExecutionPolicy:
        base = ExecutionPolicy(**cfg.model_dump())
        clean = {k: v for k, v in overrides.items() if v is not None}
        if clean:
            # Re-validate overrides through the config model's constraints.
            ExecutionConfig.model_validate({**cfg.model_dump(), **clean})
            base = replace(base, **clean)
        return base


def backoff_delay(retry_number: int, policy: ExecutionPolicy, rng: random.Random) -> float:
    """Exponential backoff: initial * 2**(retry_number-1), capped, plus up to +jitter fraction."""
    base: float = min(policy.backoff_max_s, policy.backoff_initial_s * 2.0 ** (retry_number - 1))
    return base * (1.0 + policy.backoff_jitter * rng.random())


def drain(pool: ThreadPoolExecutor, *, cancel: bool = True) -> None:
    """Wait for all worker threads, even if more Ctrl-C/SIGTERM arrive meanwhile.

    The caller holds the run's writer lock; returning early would release it while workers
    are still writing checkpoints, letting a concurrent ``resume`` race with them.
    """
    while True:
        try:
            pool.shutdown(wait=True, cancel_futures=cancel)
            return
        except KeyboardInterrupt:
            continue


@dataclass
class SessionResult:
    state: str
    counts: dict[str, int]
    recovered: dict[str, int]
    executed_calls: int


class Runner:
    def __init__(
        self,
        run: RunDir,
        provider: Provider,
        policy: ExecutionPolicy,
        *,
        mode: str,
        retry_failed: bool = False,
        sleep: Callable[[float], None] | None = None,
        rng: random.Random | None = None,
        on_call_done: Callable[[dict[str, Any]], None] | None = None,
        session_info: dict[str, Any] | None = None,
    ) -> None:
        self.run = run
        self.provider = provider
        self.policy = policy
        self.mode = mode
        self.retry_failed = retry_failed
        self._stop = threading.Event()
        self._sleep = sleep or (lambda s: self._stop.wait(s) and None)
        self._rng = rng or random.Random()
        self._rng_lock = threading.Lock()
        self._on_call_done = on_call_done
        self._session_info = session_info or {}
        self._session_index = 0
        self._executed = 0  # calls that made at least one provider attempt this session
        self._executed_lock = threading.Lock()

    def request_stop(self) -> None:
        self._stop.set()

    # ----------------------------------------------------------------- session

    def execute(self) -> SessionResult:
        with self.run.writer_lock():
            meta = self.run.read_meta()
            self._session_index = len(meta["sessions"]) + 1
            session = {
                "index": self._session_index,
                "mode": self.mode,
                "started_at": utc_now_iso(),
                "pid": os.getpid(),
                "host": socket.gethostname(),
                "policy": asdict(self.policy),
                "retry_failed": self.retry_failed,
                **self._session_info,
            }
            meta["sessions"].append(session)
            meta["state"] = "running"
            self.run.write_meta(meta)

            recovered = self._recover_interrupted()
            requests = self.run.load_requests()
            runnable = {"pending"} | ({"failed"} if self.retry_failed else set())
            todo = [s["call_id"] for s in self.run.read_all_calls() if s["status"] in runnable]

            interrupted = False
            pool = ThreadPoolExecutor(max_workers=self.policy.concurrency)
            futures: list[Future[None]] = []
            try:
                futures = [pool.submit(self._execute_call, requests[c]) for c in todo]
                pending_set = set(futures)
                while pending_set:
                    done, pending_set = wait(pending_set, return_when=FIRST_EXCEPTION)
                    for fut in done:
                        fut.result()  # re-raises a worker's BaseException (simulated crash)
            except KeyboardInterrupt:
                interrupted = True
                self._stop.set()
                drain(pool)
            except BaseException:
                # Unexpected crash: stop workers, leave checkpoints as they are (resume repairs).
                self._stop.set()
                drain(pool)
                raise
            else:
                drain(pool, cancel=False)

            self.run.rebuild_predictions()
            states = [s["status"] for s in self.run.read_all_calls()]
            counts = dict(Counter(states))
            if all(s == TERMINAL_OK for s in states):
                final = "completed"
            elif (
                interrupted
                or self._stop.is_set()
                or any(s in ("pending", "in_progress") for s in states)
            ):
                final = "interrupted"
            else:
                final = "completed_with_failures"
            meta = self.run.read_meta()
            meta["sessions"][-1].update(
                {
                    "ended_at": utc_now_iso(),
                    "result_state": final,
                    "counts": counts,
                    "recovered": recovered,
                    "executed_calls": self._executed,
                }
            )
            meta["state"] = final
            meta["counts"] = counts
            self.run.write_meta(meta)
            export_run_log(self.run)
            return SessionResult(final, counts, recovered, self._executed)

    def _recover_interrupted(self) -> dict[str, int]:
        recovered = {"from_response_file": 0, "marked_interrupted": 0}
        for state in self.run.read_all_calls():
            if state["status"] != "in_progress":
                continue
            last = state["attempts"][-1]
            rel = self.run.response_relpath(state["call_id"], last["attempt"])
            if (self.run.path / rel).is_file():
                payload = self.run.read_response(rel)
                last.update(
                    outcome="success",
                    response_file=rel,
                    usage=payload["response"].get("usage"),
                    ended_at=payload.get("received_at"),
                    recovered=True,
                )
                state.update(status="succeeded", final_attempt=last["attempt"])
                recovered["from_response_file"] += 1
            else:
                last.update(
                    outcome="interrupted",
                    ended_at=None,
                    error={
                        "type": "Interrupted",
                        "kind": "interrupted",
                        "message": "process stopped while the request was in flight",
                        "retryable": True,
                        "outcome_unknown": True,
                    },
                    detected_at=utc_now_iso(),
                )
                state["status"] = "pending"
                recovered["marked_interrupted"] += 1
            self.run.write_call(state)
        return recovered

    # ----------------------------------------------------------------- one call

    def _next_delay(self, retry_number: int) -> float:
        with self._rng_lock:
            return backoff_delay(retry_number, self.policy, self._rng)

    def _execute_call(self, request: ModelRequest) -> None:
        run = self.run
        state = run.read_call(request.call_id)
        for i in range(1, self.policy.max_attempts + 1):
            if self._stop.is_set():
                return
            n = len(state["attempts"]) + 1
            if i == 1:
                with self._executed_lock:
                    self._executed += 1
            attempt: dict[str, Any] = {
                "attempt": n,
                "session": self._session_index,
                "outcome": "in_progress",
                "started_at": utc_now_iso(),
            }
            state["attempts"].append(attempt)
            state["status"] = "in_progress"
            run.write_call(state)

            t0 = time.monotonic()
            try:
                response = self.provider.generate(request, timeout_s=self.policy.timeout_s)
            except ProviderError as exc:
                attempt.update(
                    outcome=(
                        "timeout"
                        if isinstance(exc, ProviderTimeoutError)
                        else "retryable_error"
                        if exc.retryable
                        else "permanent_error"
                    ),
                    error=exc.to_json(),
                    ended_at=utc_now_iso(),
                    latency_s=round(time.monotonic() - t0, 4),
                    usage=None,
                )
                if not exc.retryable:
                    state.update(status="failed", failure="permanent_error")
                elif i == self.policy.max_attempts:
                    state.update(status="failed", failure="retries_exhausted")
                else:
                    state["status"] = "pending"
                    attempt["retry_delay_s"] = round(self._next_delay(i), 4)
                run.write_call(state)
                if state["status"] == "failed":
                    self._notify(state)
                    return
                self._sleep(attempt["retry_delay_s"])
                continue
            except Exception as exc:  # bug in an adapter: record, do not retry
                attempt.update(
                    outcome="permanent_error",
                    error={
                        "type": type(exc).__name__,
                        "kind": "unexpected_error",
                        "message": redact(str(exc)),
                        "retryable": False,
                        "outcome_unknown": True,
                    },
                    ended_at=utc_now_iso(),
                    latency_s=round(time.monotonic() - t0, 4),
                    usage=None,
                )
                state.update(status="failed", failure="unexpected_error")
                run.write_call(state)
                self._notify(state)
                return
            except BaseException:
                # Ctrl-C / crash inside a worker: stop the other workers *now*, before they
                # pick up queued calls, then let the exception reach the main thread.
                self._stop.set()
                raise

            received_at = utc_now_iso()
            rel = run.write_response(
                request.call_id,
                n,
                {
                    "call_id": request.call_id,
                    "attempt": n,
                    "received_at": received_at,
                    "provider": self.provider.name,
                    "model": request.model,
                    "response": response.to_json(),
                },
            )
            attempt.update(
                outcome="success",
                response_file=rel,
                usage=response.usage.to_json() if response.usage else None,
                ended_at=received_at,
                latency_s=round(time.monotonic() - t0, 4),
            )
            state.update(status="succeeded", final_attempt=n)
            state.pop("failure", None)
            run.write_call(state)
            self._notify(state)
            return

    def _notify(self, state: dict[str, Any]) -> None:
        if self._on_call_done is not None:
            self._on_call_done(state)
