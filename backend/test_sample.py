"""First test: send the 50 saved sample rows to the running API and compare with expected_output.csv.
Usage:  python test_sample.py [path_to_IDS_Project] [api_url]"""
import sys, requests, numpy as np, pandas as pd
base = sys.argv[1] if len(sys.argv) > 1 else "IDS_Project"
api = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8000"

raw = pd.read_csv(f"{base}/evaluation/sample_input.csv")
exp = pd.read_csv(f"{base}/evaluation/expected_output.csv")
raw = raw.replace([np.inf, -np.inf], np.nan).astype(object).where(raw.notna(), None)

flows = [{"features": {k: (None if v is None or (isinstance(v, float) and np.isnan(v)) else float(v))
                       for k, v in row.items()}} for row in raw.to_dict("records")]
res = requests.post(f"{api}/predict/batch?store=false", json={"flows": flows}, timeout=120)
res.raise_for_status()
got = pd.DataFrame(res.json()["results"])

print("RF match: ", (got["rf_pred"].values == exp["rf_pred"].values).all())
print("XGB match:", (got["xgb_pred"].values == exp["xgb_pred"].values).all())
print("AE match: ", np.allclose(got["recon_error"], exp["recon_error"], atol=1e-4))
print(got["final_verdict"].value_counts())
