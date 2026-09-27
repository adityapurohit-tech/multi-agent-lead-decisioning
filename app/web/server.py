"""Flask REST API + WebSocket live trace stream.

    python -m app.web.server         ->  http://localhost:5000
"""
import json
import logging
import queue
import random

from flask import Flask, abort, jsonify, request, send_from_directory
from flask_sock import Sock

from app import data
from app.ml.model import model_card
from app.orchestrator import get_orchestrator
from app.scenarios import FREQUENCY_CAP_TIP, SCENARIOS
from app.services import services

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("server")

app = Flask(__name__, static_folder="static", static_url_path="/static")
sock = Sock(app)


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/health")
def health():
    card = model_card()
    return jsonify({
        "status": "ok",
        "redis": {"backend": services.store.backend, "live": services.store.live,
                  "keys": services.store.r.dbsize()},
        "neo4j": {"backend": services.graph.backend, "live": services.graph.live, **services.graph.stats()},
        "model": {"name": card.get("model"), "roc_auc": card.get("test_metrics", {}).get("roc_auc"),
                  "trained_at": card.get("trained_at")},
    })


@app.get("/api/scenarios")
def scenarios():
    return jsonify({"scenarios": SCENARIOS, "tip": FREQUENCY_CAP_TIP})


@app.get("/api/leads")
def leads():
    return jsonify(data.search_leads(request.args.get("q", ""), int(request.args.get("limit", 20))))


@app.get("/api/leads/<lead_id>")
def lead_detail(lead_id):
    lead = data.get_lead(lead_id)
    if not lead:
        abort(404)
    return jsonify({
        "lead_id": lead_id,
        "short_term_memory": services.store.get_stm(lead_id),
        "frequency_cap": services.store.last_contact(lead_id),
        "decision_history": services.graph.lead_history(lead_id),
    })


@app.get("/api/leads/<lead_id>/graph")
def lead_graph(lead_id):
    return jsonify(services.graph.lead_subgraph(lead_id))


@app.post("/api/leads/<lead_id>/clear-cap")
def clear_cap(lead_id):
    services.store.clear_contact(lead_id)
    return jsonify({"ok": True})


@app.post("/api/runs")
def start_run():
    lead_id = (request.get_json(silent=True) or {}).get("lead_id", "").strip().upper()
    if not data.get_lead(lead_id):
        return jsonify({"error": f"unknown lead {lead_id}"}), 404
    return jsonify({"run_id": get_orchestrator().start(lead_id)}), 202


@app.get("/api/runs")
def list_runs():
    return jsonify(services.store.list_runs(int(request.args.get("limit", 25))))


@app.get("/api/runs/<run_id>")
def get_run(run_id):
    meta = services.store.get_run(run_id)
    if not meta:
        abort(404)
    return jsonify({"run": meta, "events": services.store.get_events(run_id),
                    "state": get_orchestrator().state(run_id)})


@app.post("/api/runs/<run_id>/review")
def review(run_id):
    body = request.get_json(silent=True) or {}
    action = body.get("action")
    if action not in ("approve", "nurture", "reject"):
        return jsonify({"error": "action must be approve | nurture | reject"}), 400
    try:
        get_orchestrator().resume(run_id, action, body.get("note", ""), body.get("reviewer", "dashboard"))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 409
    return jsonify({"ok": True}), 202


@app.get("/api/review-queue")
def review_queue():
    runs = [r for r in services.store.list_runs(200) if r.get("status") == "awaiting_review"]
    return jsonify(runs)


@app.post("/api/batch")
def batch():
    n = max(1, min(int((request.get_json(silent=True) or {}).get("n", 50)), 500))
    ids = random.sample(data.table("leads")["lead_id"].tolist(), n)
    return jsonify(get_orchestrator().batch(ids))


@app.get("/api/graph")
def graph():
    return jsonify(get_orchestrator().structure())


@app.get("/api/model")
def model():
    return jsonify(model_card())


@app.get("/api/metrics")
def metrics():
    return jsonify({**services.store.metrics(), "graph": services.graph.stats()})


@app.post("/api/reset")
def reset():
    services.store.reset()
    services.graph.clear_decisions()
    return jsonify({"ok": True})


@app.get("/api/campaigns")
def campaigns():
    from app.agents.optimizer import portfolio
    pf = portfolio()
    rows = [{"campaign_id": cid, **{k: r[k] for k in ("health", "roas", "cpl_usd", "ctr", "trend", "health_reason")}}
            for cid, r in pf["campaigns"].items()]
    return jsonify({"campaigns": rows, "cluster_sizes": pf["cluster_sizes"], "centroids": pf["centroids"]})


@sock.route("/ws")
def ws(socket):
    """Push every trace event to the dashboard as it happens."""
    q = services.bus.subscribe()
    try:
        while True:
            try:
                event = q.get(timeout=20)
                socket.send(json.dumps(event))
            except queue.Empty:
                socket.send(json.dumps({"type": "ping"}))
    except Exception:
        pass
    finally:
        services.bus.unsubscribe(q)


def main():
    import os
    get_orchestrator()      # warms up data, model and state backends before the first request
    port = int(os.getenv("PORT", 5000))
    log.info("Dashboard: http://localhost:%s", port)
    app.run(host="0.0.0.0", port=port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
