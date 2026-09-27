"""Neo4j-backed relationship state: leads, campaigns, segments and every agent decision.

Graph model
    (:Lead)-[:FROM_CAMPAIGN]->(:Campaign)
    (:Lead)-[:IN_INDUSTRY]->(:Industry)     (:Lead)-[:HAS_PERSONA]->(:Persona)
    (:Lead)-[:CONVERTED {value, type}]->(:Campaign)
    (:Lead)-[:HAS_DECISION]->(:Decision {run_id, route, p_convert, channel, at})
    (:Decision)-[:FOR_CAMPAIGN]->(:Campaign)

Questions the agents ask of it are relationship questions -- "how did leads like this one
convert, and through which channel?" -- which is why this lives in a graph, not in Redis.

When Neo4j is unreachable, InMemoryGraph answers the same questions from a NetworkX graph.
"""
import logging
import threading
import time
from collections import Counter

import networkx as nx

from app import data
from app.config import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER

log = logging.getLogger(__name__)


def _lead_rows():
    leads = data.table("leads")
    return leads[["lead_id", "campaign_id", "industry", "persona", "region", "preferred_channel"]].to_dict("records")


def _conversion_rows():
    return data.table("conversions")[["lead_id", "campaign_id", "conversion_value_usd", "conversion_type"]].to_dict("records")


NEIGHBOURHOOD_CYPHER = """MATCH (l:Lead {id: $id})
OPTIONAL MATCH (l)-[:FROM_CAMPAIGN]->(c:Campaign)
OPTIONAL MATCH (l)-[:IN_INDUSTRY]->(i:Industry)
OPTIONAL MATCH (l)-[:HAS_PERSONA]->(p:Persona)
OPTIONAL MATCH (l)-[:HAS_DECISION]->(d:Decision)
OPTIONAL MATCH (d)-[:FOR_CAMPAIGN]->(dc:Campaign)
RETURN l, c, i, p, d, dc"""

PEERS_CYPHER = """MATCH (l:Lead {id: $id})-[:IN_INDUSTRY]->(i)<-[:IN_INDUSTRY]-(peer:Lead)
      -[:HAS_PERSONA]->(p)<-[:HAS_PERSONA]-(l)
MATCH (peer)-[:CONVERTED]->(pc:Campaign)
RETURN peer.id AS peer, pc.id AS campaign LIMIT 4"""


def _subgraph(lead_id, campaign, industry, persona, decisions, peers) -> dict:
    """Shape a lead's neighbourhood as nodes + relationships for the dashboard."""
    nodes = {lead_id: {"id": lead_id, "label": "Lead", "name": lead_id}}
    edges = []

    def add(node_id, label, name, src, rel):
        nodes.setdefault(node_id, {"id": node_id, "label": label, "name": name})
        edges.append({"source": src, "target": node_id, "type": rel})

    if campaign:
        add(f"c:{campaign}", "Campaign", campaign, lead_id, "FROM_CAMPAIGN")
    if industry:
        add(f"i:{industry}", "Industry", industry, lead_id, "IN_INDUSTRY")
    if persona:
        add(f"p:{persona}", "Persona", persona, lead_id, "HAS_PERSONA")
    for d in sorted(decisions, key=lambda d: -(d.get("at") or 0))[:4]:
        add(f"d:{d['run_id']}", "Decision", d.get("route", "?"), lead_id, "HAS_DECISION")
        if d.get("campaign"):
            add(f"c:{d['campaign']}", "Campaign", d["campaign"], f"d:{d['run_id']}", "FOR_CAMPAIGN")
    for peer, camp in peers:
        add(peer, "Lead", peer, f"i:{industry}", "IN_INDUSTRY")
        edges.append({"source": peer, "target": f"p:{persona}", "type": "HAS_PERSONA"})
        add(f"c:{camp}", "Campaign", camp, peer, "CONVERTED")
    return {"nodes": list(nodes.values()), "edges": edges,
            "cypher": NEIGHBOURHOOD_CYPHER + "\n\n// converted peers\n" + PEERS_CYPHER}


class InMemoryGraph:
    backend, live = "networkx (in-memory fallback)", False

    def __init__(self):
        self.g = nx.MultiDiGraph()
        self._lock = threading.Lock()
        self._peers: dict[tuple, list[str]] = {}
        for row in _lead_rows():
            lid = row["lead_id"]
            self.g.add_node(lid, label="Lead", **row)
            self.g.add_edge(lid, row["campaign_id"], key="FROM_CAMPAIGN")
            self.g.add_edge(lid, f"industry:{row['industry']}", key="IN_INDUSTRY")
            self.g.add_edge(lid, f"persona:{row['persona']}", key="HAS_PERSONA")
            self._peers.setdefault((row["industry"], row["persona"]), []).append(lid)
        self.converted: dict[str, list[dict]] = {}
        for c in _conversion_rows():
            self.g.add_edge(c["lead_id"], c["campaign_id"], key="CONVERTED",
                            value=c["conversion_value_usd"], type=c["conversion_type"])
            self.converted.setdefault(c["lead_id"], []).append(c)

    def peer_insights(self, lead: dict) -> dict:
        peers = [p for p in self._peers.get((lead["industry"], lead["persona"]), []) if p != lead["lead_id"]]
        conv = [p for p in peers if p in self.converted]
        channels = Counter(self.g.nodes[p]["preferred_channel"] for p in conv)
        camp_conv = sum(1 for _, _, k in self.g.in_edges(lead["campaign_id"], keys=True) if k == "CONVERTED")
        return {
            "peer_group": f"{lead['industry']} / {lead['persona']}",
            "peers": len(peers),
            "peer_conversion_rate": round(len(conv) / max(len(peers), 1), 3),
            "top_converting_channels": [c for c, _ in channels.most_common(3)],
            "campaign_conversions": camp_conv,
        }

    def record_decision(self, lead_id: str, decision: dict):
        node = f"decision:{decision['run_id']}"
        with self._lock:
            self.g.add_node(node, label="Decision", at=time.time(), **decision)
            self.g.add_edge(lead_id, node, key="HAS_DECISION")
            if decision.get("campaign_id"):
                self.g.add_edge(node, decision["campaign_id"], key="FOR_CAMPAIGN")

    def lead_history(self, lead_id: str, limit: int = 10) -> list[dict]:
        with self._lock:
            if lead_id not in self.g:
                return []
            decs = [dict(self.g.nodes[v]) for _, v, k in self.g.out_edges(lead_id, keys=True) if k == "HAS_DECISION"]
        decs = [{k: v for k, v in d.items() if k != "label"} for d in decs]
        return sorted(decs, key=lambda d: -d["at"])[:limit]

    def clear_decisions(self):
        with self._lock:
            self.g.remove_nodes_from([n for n, d in self.g.nodes(data=True) if d.get("label") == "Decision"])

    def lead_subgraph(self, lead_id: str) -> dict:
        with self._lock:
            if lead_id not in self.g:
                return {"nodes": [], "edges": [], "cypher": NEIGHBOURHOOD_CYPHER}
            attrs = self.g.nodes[lead_id]
            decisions = []
            for _, v, k in self.g.out_edges(lead_id, keys=True):
                if k == "HAS_DECISION":
                    d = dict(self.g.nodes[v])
                    d["campaign"] = d.get("campaign_id")
                    decisions.append(d)
        peers = [p for p in self._peers.get((attrs["industry"], attrs["persona"]), [])
                 if p != lead_id and p in self.converted][:4]
        return _subgraph(lead_id, attrs["campaign_id"], attrs["industry"], attrs["persona"], decisions,
                         [(p, self.converted[p][0]["campaign_id"]) for p in peers])

    def stats(self) -> dict:
        labels = Counter(d.get("label", "Other") for _, d in self.g.nodes(data=True))
        return {"nodes": self.g.number_of_nodes(), "relationships": self.g.number_of_edges(),
                "decisions": labels.get("Decision", 0)}


class Neo4jGraph:
    live = True

    def __init__(self, driver):
        self.driver = driver
        self.backend = f"neo4j @ {NEO4J_URI}"
        self._ensure_seeded()

    def _run(self, query: str, **params):
        with self.driver.session() as s:
            return [r.data() for r in s.run(query, **params)]

    def _ensure_seeded(self):
        self._run("CREATE CONSTRAINT lead_id IF NOT EXISTS FOR (l:Lead) REQUIRE l.id IS UNIQUE")
        self._run("CREATE CONSTRAINT campaign_id IF NOT EXISTS FOR (c:Campaign) REQUIRE c.id IS UNIQUE")
        if self._run("MATCH (l:Lead) RETURN count(l) AS n")[0]["n"] > 0:
            return
        log.info("Seeding Neo4j graph from seeds/ ...")
        rows = _lead_rows()
        for i in range(0, len(rows), 1000):
            self._run("""
                UNWIND $rows AS r
                MERGE (l:Lead {id: r.lead_id})
                  SET l.region = r.region, l.preferred_channel = r.preferred_channel,
                      l.industry = r.industry, l.persona = r.persona
                MERGE (c:Campaign {id: r.campaign_id})
                MERGE (i:Industry {name: r.industry})
                MERGE (p:Persona {name: r.persona})
                MERGE (l)-[:FROM_CAMPAIGN]->(c)
                MERGE (l)-[:IN_INDUSTRY]->(i)
                MERGE (l)-[:HAS_PERSONA]->(p)""", rows=rows[i:i + 1000])
        self._run("""
            UNWIND $rows AS r
            MATCH (l:Lead {id: r.lead_id})
            MERGE (c:Campaign {id: r.campaign_id})
            MERGE (l)-[:CONVERTED {value: r.conversion_value_usd, type: r.conversion_type}]->(c)""",
                  rows=_conversion_rows())

    def peer_insights(self, lead: dict) -> dict:
        rec = self._run("""
            MATCH (l:Lead {id: $id})-[:IN_INDUSTRY]->(:Industry)<-[:IN_INDUSTRY]-(p:Lead)-[:HAS_PERSONA]->(:Persona)<-[:HAS_PERSONA]-(l)
            WHERE p <> l
            WITH p, EXISTS { (p)-[:CONVERTED]->() } AS converted
            RETURN count(p) AS peers,
                   sum(CASE WHEN converted THEN 1 ELSE 0 END) AS conv,
                   collect(CASE WHEN converted THEN p.preferred_channel END) AS channels""", id=lead["lead_id"])[0]
        camp = self._run("MATCH (:Lead)-[:CONVERTED]->(c:Campaign {id: $cid}) RETURN count(*) AS n",
                         cid=lead["campaign_id"])[0]["n"]
        return {
            "peer_group": f"{lead['industry']} / {lead['persona']}",
            "peers": rec["peers"],
            "peer_conversion_rate": round(rec["conv"] / max(rec["peers"], 1), 3),
            "top_converting_channels": [c for c, _ in Counter(rec["channels"]).most_common(3)],
            "campaign_conversions": camp,
        }

    def record_decision(self, lead_id: str, decision: dict):
        props = {k: v for k, v in decision.items() if isinstance(v, (str, int, float, bool)) or v is None}
        self._run("""
            MATCH (l:Lead {id: $lead_id})
            CREATE (d:Decision) SET d = $props, d.at = timestamp() / 1000.0
            MERGE (l)-[:HAS_DECISION]->(d)
            WITH d
            OPTIONAL MATCH (c:Campaign {id: $props.campaign_id})
            FOREACH (_ IN CASE WHEN c IS NULL THEN [] ELSE [1] END | MERGE (d)-[:FOR_CAMPAIGN]->(c))""",
                  lead_id=lead_id, props=props)

    def lead_history(self, lead_id: str, limit: int = 10) -> list[dict]:
        rows = self._run("""
            MATCH (:Lead {id: $id})-[:HAS_DECISION]->(d:Decision)
            RETURN d ORDER BY d.at DESC LIMIT $limit""", id=lead_id, limit=limit)
        return [r["d"] for r in rows]

    def clear_decisions(self):
        self._run("MATCH (d:Decision) DETACH DELETE d")

    def lead_subgraph(self, lead_id: str) -> dict:
        rows = self._run(NEIGHBOURHOOD_CYPHER, id=lead_id)
        if not rows:
            return {"nodes": [], "edges": [], "cypher": NEIGHBOURHOOD_CYPHER}
        r0 = rows[0]
        decisions = [{**r["d"], "campaign": (r["dc"] or {}).get("id")} for r in rows if r["d"]]
        peers = [(r["peer"], r["campaign"]) for r in self._run(PEERS_CYPHER, id=lead_id)]
        return _subgraph(lead_id, (r0["c"] or {}).get("id"), (r0["i"] or {}).get("name"),
                         (r0["p"] or {}).get("name"), decisions, peers)

    def stats(self) -> dict:
        r = self._run("""
            CALL () { MATCH (n) RETURN count(n) AS nodes }
            CALL () { MATCH ()-[r]->() RETURN count(r) AS rels }
            CALL () { MATCH (d:Decision) RETURN count(d) AS decisions }
            RETURN nodes, rels, decisions""")[0]
        return {"nodes": r["nodes"], "relationships": r["rels"], "decisions": r["decisions"]}


def connect():
    """Connect to Neo4j, waiting up to NEO4J_WAIT_SECONDS for it to come up (it boots slowly)."""
    import os
    from neo4j import GraphDatabase
    deadline = time.time() + float(os.getenv("NEO4J_WAIT_SECONDS", 0))
    while True:
        try:
            driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD), connection_timeout=2,
                                          notifications_min_severity="OFF")
            driver.verify_connectivity()
            return Neo4jGraph(driver)
        except Exception as exc:
            if time.time() < deadline:
                time.sleep(2)
                continue
            log.warning("Neo4j not reachable at %s (%s) -- using in-memory graph", NEO4J_URI, type(exc).__name__)
            return InMemoryGraph()
