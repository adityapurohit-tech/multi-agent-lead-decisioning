import pandas as pd

files = ["ab_variants", "agent_actions", "campaign_daily"]
data = {f: pd.read_csv(f"seeds/{f}.csv") for f in files}

for k, v in data.items():
    print(f"\n=== {k.upper()} ===")
    print(v.head())
