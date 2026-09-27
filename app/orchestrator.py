"""Runs workflows: start, pause for human review, resume, and concurrent batch execution."""
import threading
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from langgraph.types import Command

from app.graph import build_graph, graph_structure
from app.services import services


def warm_up():
    """Load data, features, model and portfolio analytics once so the first demo run is fast."""
    from app import data
    from app.agents.optimizer import _variant_stats, portfolio
    from app.ml import model
    from app.ml.features import feature_table
    data._interactions_by_lead(), data._leads_indexed(), data._ltm_indexed()
    feature_table(), model.load(), portfolio(), _variant_stats()
    services.store, services.graph


class Orchestrator:
    def __init__(self):
        warm_up()
        self.graph = build_graph()
        self._pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="run")

    @staticmethod
    def _config(run_id: str) -> dict:
        return {"configurable": {"thread_id": run_id}, "recursion_limit": 40}

    # ---- lifecycle ---------------------------------------------------------
    def start(self, lead_id: str, background: bool = True, source: str = "ui") -> str:
        run_id = f"run_{uuid.uuid4().hex[:10]}"
        services.store.save_run(run_id, lead_id=lead_id, status="running", started_at=time.time(), source=source)
        services.tracer.emit(run_id, "run_started", lead_id=lead_id)
        payload = {"run_id": run_id, "lead_id": lead_id, "attempts": 0, "trace": []}
        if background:
            self._pool.submit(self._execute, run_id, payload)
        else:
            self._execute(run_id, payload)
        return run_id

    def resume(self, run_id: str, action: str, note: str = "", reviewer: str = "reviewer", background: bool = True):
        snap = self.graph.get_state(self._config(run_id))
        if not snap.next:
            raise ValueError(f"{run_id} is not awaiting review")
        services.store.save_run(run_id, status="running")
        services.tracer.emit(run_id, "review_submitted", action=action, note=note, reviewer=reviewer)
        cmd = Command(resume={"action": action, "note": note, "reviewer": reviewer})
        if background:
            self._pool.submit(self._execute, run_id, cmd)
        else:
            self._execute(run_id, cmd)

    def _execute(self, run_id: str, payload):
        cfg = self._config(run_id)
        try:
            for chunk in self.graph.stream(payload, cfg, stream_mode="updates"):
                if "__interrupt__" in chunk:
                    value = chunk["__interrupt__"][0].value
                    services.store.save_run(run_id, status="awaiting_review", review_request=value)
                    services.tracer.emit(run_id, "awaiting_review", request=value)
            snap = self.graph.get_state(cfg)
            if snap.next:          # paused at interrupt
                return
            state = snap.values
            outcome = state.get("outcome", {})
            run = services.store.get_run(run_id)
            duration = round(time.time() - run.get("started_at", time.time()), 3)
            services.store.save_run(run_id, status="completed", route=outcome.get("route"),
                                    outcome=outcome, finished_at=time.time(), duration_s=duration)
            services.tracer.emit(run_id, "run_finished", outcome=outcome, duration_s=duration,
                                 path=[t["node"] for t in state.get("trace", [])])
        except Exception as exc:
            services.store.save_run(run_id, status="failed", error=f"{type(exc).__name__}: {exc}")
            services.tracer.emit(run_id, "run_failed", error=f"{type(exc).__name__}: {exc}")

    # ---- inspection --------------------------------------------------------
    def state(self, run_id: str) -> dict:
        snap = self.graph.get_state(self._config(run_id))
        return {"values": snap.values, "next": list(snap.next)}

    def structure(self) -> dict:
        return graph_structure(self.graph)

    # ---- batch -------------------------------------------------------------
    def batch(self, lead_ids: list[str], workers: int = 8) -> dict:
        """Run many leads concurrently. Runs needing review stay paused in the review queue."""
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            run_ids = list(pool.map(lambda lid: self.start(lid, background=False, source="batch"), lead_ids))
        runs = [services.store.get_run(r) for r in run_ids]
        routes = Counter((r.get("route") or r.get("status")) for r in runs)
        return {"n": len(run_ids), "wall_time_s": round(time.time() - t0, 2),
                "routes": dict(routes), "run_ids": run_ids}


_orchestrator = None
_lock = threading.Lock()


def get_orchestrator() -> Orchestrator:
    global _orchestrator
    with _lock:
        if _orchestrator is None:
            _orchestrator = Orchestrator()
        return _orchestrator
