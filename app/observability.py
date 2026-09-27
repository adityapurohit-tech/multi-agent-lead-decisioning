"""Structured tracing for every agent step.

Each node emits node_started / node_finished (or node_paused / node_failed) with timing,
a one-line summary and the state it produced. Events go to three sinks:
  * Redis list run:{id}:events (+ pub/sub)  -> replayable per run
  * outputs/traces.jsonl                    -> durable audit log
  * in-process EventBus                     -> live WebSocket push to the dashboard
"""
import functools
import json
import queue
import threading
import time

from langgraph.errors import GraphBubbleUp

from app.config import TRACE_LOG


class EventBus:
    def __init__(self):
        self._subs: list[queue.Queue] = []
        self._lock = threading.Lock()

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=1000)
        with self._lock:
            self._subs.append(q)
        return q

    def unsubscribe(self, q: queue.Queue):
        with self._lock:
            if q in self._subs:
                self._subs.remove(q)

    def publish(self, event: dict):
        with self._lock:
            subs = list(self._subs)
        for q in subs:
            try:
                q.put_nowait(event)
            except queue.Full:
                pass


class Tracer:
    def __init__(self, store, bus: EventBus):
        self.store, self.bus = store, bus
        self._file_lock = threading.Lock()

    def emit(self, run_id: str, type_: str, **fields):
        event = {"run_id": run_id, "type": type_, "ts": round(time.time(), 3), **fields}
        event = json.loads(json.dumps(event, default=str))   # make JSON-safe once
        self.store.append_event(run_id, event)
        self.bus.publish(event)
        with self._file_lock, TRACE_LOG.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event) + "\n")
        return event


def traced(tracer_getter, node_name: str):
    """Wrap a LangGraph node so every execution is timed and traced.

    Agents may return a private "_summary" key: a one-line human explanation of what they
    decided. It is stripped before the update reaches graph state.
    """
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(state):
            tracer = tracer_getter()
            run_id = state["run_id"]
            tracer.emit(run_id, "node_started", node=node_name)
            t0 = time.perf_counter()
            try:
                update = fn(state) or {}
            except GraphBubbleUp:            # interrupt() for human review -- not an error
                tracer.emit(run_id, "node_paused", node=node_name,
                            ms=round((time.perf_counter() - t0) * 1000, 2))
                raise
            except Exception as exc:
                tracer.emit(run_id, "node_failed", node=node_name, error=f"{type(exc).__name__}: {exc}")
                raise
            ms = round((time.perf_counter() - t0) * 1000, 2)
            summary = update.pop("_summary", "")
            tracer.store.record_node_latency(node_name, ms)
            tracer.emit(run_id, "node_finished", node=node_name, ms=ms, summary=summary,
                        update={k: v for k, v in update.items() if k != "trace"})
            update["trace"] = [{"node": node_name, "ms": ms, "summary": summary}]
            return update
        return wrapper
    return decorator
