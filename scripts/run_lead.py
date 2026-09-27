"""Run one or more leads through the graph from the terminal and print the agent trace.

    python scripts/run_lead.py L0000003
    python scripts/run_lead.py L0000173 --review approve
    python scripts/run_lead.py --mermaid
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
logging.disable(logging.WARNING)

from app.orchestrator import get_orchestrator  # noqa: E402
from app.services import services  # noqa: E402


def show(run_id):
    for e in services.store.get_events(run_id):
        if e["type"] == "node_finished":
            print(f"  {e['node']:<19}{e['ms']:>8.2f} ms  {e['summary']}")
        elif e["type"] in ("node_paused", "awaiting_review"):
            if e["type"] == "awaiting_review":
                print(f"  {'** PAUSED':<19}{'':>11}  {e['request']['reason']}")
        elif e["type"] == "review_submitted":
            print(f"  {'** REVIEW':<19}{'':>11}  {e['action']}")
        elif e["type"] == "run_finished":
            o = e["outcome"]
            print(f"  => {o['status'].upper()}: {o['route']}" + (f" via {o.get('channel')}" if o.get("channel") else "")
                  + (f" ({o['reason']})" if o.get("reason") else "") + f"  [{e['duration_s'] * 1000:.0f} ms]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("leads", nargs="*", default=["L0000003"])
    ap.add_argument("--review", choices=["approve", "nurture", "reject"], help="auto-answer human review")
    ap.add_argument("--mermaid", action="store_true", help="print the compiled graph as Mermaid")
    args = ap.parse_args()
    o = get_orchestrator()
    if args.mermaid:
        print(o.structure()["mermaid"])
        return
    print(f"Redis: {services.store.backend} | Graph: {services.graph.backend}\n")
    for lead in args.leads:
        run_id = o.start(lead.upper(), background=False, source="cli")
        print(f"{lead}  ({run_id})")
        if services.store.get_run(run_id)["status"] == "awaiting_review" and args.review:
            o.resume(run_id, args.review, "cli", background=False)
        show(run_id)
        print()


if __name__ == "__main__":
    main()
