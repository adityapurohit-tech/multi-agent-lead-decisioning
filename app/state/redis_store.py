"""Redis-backed fast, ephemeral state shared by all agents.

Key layout
    stm:{lead_id}            hash   short-term memory (last route, last outreach...)   TTL 24h
    contact:{lead_id}        string frequency-cap marker                                TTL 72h
    cache:{key}              string cached campaign analyses                            TTL 15m
    run:{run_id}             hash   run metadata (status, lead, route, timings)
    run:{run_id}:events      list   ordered trace events for the run
    runs                     zset   run ids by start time
    metrics:nodes            hash   per-agent call count and cumulative latency
    metrics:routes           hash   decision counters
    channel agent-events     pub/sub fan-out of trace events

If no Redis server is reachable the same code runs against fakeredis, an in-process
implementation of the Redis protocol, so behaviour is identical in a laptop demo.
"""
import json
import logging
import time

import redis

from app.config import CAMPAIGN_CACHE_TTL, FREQUENCY_CAP_HOURS, REDIS_URL, STM_TTL_SECONDS

log = logging.getLogger(__name__)


class RedisStore:
    def __init__(self, url: str = REDIS_URL):
        try:
            client = redis.Redis.from_url(url, decode_responses=True, socket_connect_timeout=1, socket_timeout=2,
                                           protocol=2)   # RESP2: works with every Redis server version
            client.ping()
            self.backend, self.live = f"redis @ {url}", True
        except Exception:
            import fakeredis
            client = fakeredis.FakeRedis(decode_responses=True)
            self.backend, self.live = "fakeredis (in-process fallback)", False
            log.warning("Redis not reachable at %s -- using in-process fakeredis", url)
        self.r = client

    # ---- short-term memory -------------------------------------------------
    def get_stm(self, lead_id: str) -> dict:
        return {k: json.loads(v) for k, v in self.r.hgetall(f"stm:{lead_id}").items()}

    def update_stm(self, lead_id: str, **fields):
        key = f"stm:{lead_id}"
        self.r.hset(key, mapping={k: json.dumps(v, default=str) for k, v in fields.items()})
        self.r.expire(key, STM_TTL_SECONDS)

    # ---- frequency cap -----------------------------------------------------
    def last_contact(self, lead_id: str) -> dict | None:
        raw = self.r.get(f"contact:{lead_id}")
        if not raw:
            return None
        info = json.loads(raw)
        info["ttl_hours"] = round(self.r.ttl(f"contact:{lead_id}") / 3600, 1)
        return info

    def record_contact(self, lead_id: str, channel: str, run_id: str):
        payload = json.dumps({"channel": channel, "run_id": run_id, "at": time.time()})
        self.r.set(f"contact:{lead_id}", payload, ex=FREQUENCY_CAP_HOURS * 3600)

    def clear_contact(self, lead_id: str):
        self.r.delete(f"contact:{lead_id}")

    # ---- cache -------------------------------------------------------------
    def cache_get(self, key: str):
        raw = self.r.get(f"cache:{key}")
        return json.loads(raw) if raw else None

    def cache_set(self, key: str, value, ttl: int = CAMPAIGN_CACHE_TTL):
        self.r.set(f"cache:{key}", json.dumps(value, default=str), ex=ttl)

    # ---- runs & trace events -------------------------------------------------
    def save_run(self, run_id: str, **fields):
        key = f"run:{run_id}"
        if not self.r.exists(key):
            self.r.zadd("runs", {run_id: time.time()})
        self.r.hset(key, mapping={k: json.dumps(v, default=str) for k, v in fields.items()})

    def get_run(self, run_id: str) -> dict:
        return {k: json.loads(v) for k, v in self.r.hgetall(f"run:{run_id}").items()}

    def list_runs(self, limit: int = 20) -> list[dict]:
        ids = self.r.zrevrange("runs", 0, limit - 1)
        return [{"run_id": i, **self.get_run(i)} for i in ids]

    def append_event(self, run_id: str, event: dict):
        raw = json.dumps(event, default=str)
        self.r.rpush(f"run:{run_id}:events", raw)
        self.r.publish("agent-events", raw)

    def get_events(self, run_id: str) -> list[dict]:
        return [json.loads(e) for e in self.r.lrange(f"run:{run_id}:events", 0, -1)]

    # ---- metrics -----------------------------------------------------------
    def record_node_latency(self, node: str, ms: float):
        self.r.hincrby("metrics:nodes", f"{node}:count", 1)
        self.r.hincrbyfloat("metrics:nodes", f"{node}:ms", ms)

    def incr(self, name: str, field: str):
        self.r.hincrby(f"metrics:{name}", field, 1)

    def metrics(self) -> dict:
        nodes: dict[str, dict] = {}
        for k, v in self.r.hgetall("metrics:nodes").items():
            node, kind = k.rsplit(":", 1)
            nodes.setdefault(node, {})[kind] = float(v)
        latency = {n: {"calls": int(d.get("count", 0)), "avg_ms": round(d.get("ms", 0) / max(d.get("count", 1), 1), 2)}
                   for n, d in nodes.items()}
        routes = {k: int(v) for k, v in self.r.hgetall("metrics:routes").items()}
        return {"node_latency": latency, "routes": routes}

    def reset(self):
        """Clear demo state (contacts, memory, runs, metrics); keep nothing between demos."""
        for pattern in ("contact:*", "stm:*", "run:*", "cache:*"):
            for key in self.r.scan_iter(pattern):
                self.r.delete(key)
        self.r.delete("runs", "metrics:nodes", "metrics:routes")
