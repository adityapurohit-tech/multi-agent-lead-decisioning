"""Central configuration. Every value can be overridden with an environment variable."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEEDS_DIR = ROOT / "seeds"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"
MODELS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

# State backends. If these are unreachable the app falls back to in-process equivalents.
REDIS_URL = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "neo4jpassword")

# Decision policy
HOT_THRESHOLD = float(os.getenv("HOT_THRESHOLD", 0.60))    # p(convert) >= this -> sales handoff
COLD_THRESHOLD = float(os.getenv("COLD_THRESHOLD", 0.25))  # p(convert) <  this -> nurture
# Predictions whose probability falls inside this band around the HOT boundary are
# escalated to a human instead of being auto-decided.
UNCERTAINTY_BAND = float(os.getenv("UNCERTAINTY_BAND", 0.04))
MAX_REVISIONS = int(os.getenv("MAX_REVISIONS", 3))          # generator -> critic loop limit
FREQUENCY_CAP_HOURS = int(os.getenv("FREQUENCY_CAP_HOURS", 72))
STM_TTL_SECONDS = int(os.getenv("STM_TTL_SECONDS", 24 * 3600))
CAMPAIGN_CACHE_TTL = int(os.getenv("CAMPAIGN_CACHE_TTL", 900))

# The seed data window ends on this date; recency features are measured from it.
REFERENCE_DATE = "2025-08-29"

MODEL_PATH = MODELS_DIR / "lead_scoring.joblib"
MODEL_CARD_PATH = MODELS_DIR / "model_card.json"
DECISIONS_LOG = OUTPUTS_DIR / "decisions.jsonl"
TRACE_LOG = OUTPUTS_DIR / "traces.jsonl"
