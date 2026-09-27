"""
EMR Anomaly Detection — Live Thesis Demo
Suvechchha Subedi · Purbanchal University · MIT Thesis Defense

Runs the actual middleware pipeline (JWT auth -> RBAC -> feature build ->
ML inference) against the trained models, with a browser dashboard.

Run:    python demo_app.py          (then open http://127.0.0.1:8000)
Needs:  pip install fastapi uvicorn scikit-learn xgboost pandas numpy pyjwt
Files:  expects ./models/*.pkl, ./data/emr_synthetic_dataset.csv,
        ./data/model_results.json, ./data/api_benchmark.json  (bundled)
"""
import json, os, time, random
import numpy as np
import pandas as pd
import pickle
import jwt as pyjwt
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse

BASE = os.path.dirname(os.path.abspath(__file__))
SECRET = os.getenv("JWT_SECRET", "local-development-secret")
ALGO = "HS256"
ROLES = {"doctor": ["write_encounter"], "nurse": ["write_vitals"], "clerk": []}
app = FastAPI(title="EMR Anomaly Detection — Thesis Demo")
# ---------------- load artifacts ----------------
models = {}
for nm in ["iso", "ocsvm", "rf", "xgb", "imputer", "scaler"]:
    with open(f"{BASE}/models/model_{nm}.pkl", "rb") as f:
        models[nm] = pickle.load(f)
FEATS = json.load(open(f"{BASE}/models/feature_columns.json"))
DF = pd.read_csv(f"{BASE}/data/emr_synthetic_dataset.csv")
RESULTS = json.load(open(f"{BASE}/data/model_results.json"))
BENCH = json.load(open(f"{BASE}/data/api_benchmark.json"))
TEST = DF.sample(frac=0.2, random_state=42)  # demo pool

MODEL_LABEL = {"iso": "Isolation Forest", "ocsvm": "One-Class SVM",
               "rf": "Random Forest", "xgb": "XGBoost"}

def make_token(role, tamper=False):
    t = pyjwt.encode({"sub": f"user-{role}", "role": role, "exp": time.time() + 3600},
                     SECRET, algorithm=ALGO)
    return t[:-4] + "XXXX" if tamper else t

def build_features(rec: dict) -> pd.DataFrame:
    row = {c: 0.0 for c in FEATS}
    row.update({
        "age": rec["age"], "weight_kg": rec["weight_kg"],
        "heart_rate": rec["heart_rate"], "resp_rate": rec["resp_rate"],
        "sbp": rec["sbp"], "dbp": rec["dbp"], "temp_c": rec["temp_c"],
        "spo2": rec["spo2"], "dose_mg": rec["dose_mg"],
        "consult_min": rec.get("consult_min", 10),
        "dose_per_kg": rec["dose_mg"] / max(rec["weight_kg"], 0.1),
        "age_weight_ratio": rec["age"] / max(rec["weight_kg"], 0.1),
        "shock_index": rec["heart_rate"] / max(rec["sbp"], 1),
        "pulse_pressure": rec["sbp"] - rec["dbp"],
        "sex_enc": 1 if rec.get("sex", "F") == "M" else 0})
    dxcol = f"dx_{rec['diagnosis']}"
    if dxcol in row:
        row[dxcol] = 1.0
    X = pd.DataFrame([row])[FEATS].astype(float)
    X[:] = models["imputer"].transform(X)
    X[:] = models["scaler"].transform(X)
    return X

def score_all(X):
    out = {}
    for key in ["iso", "ocsvm", "rf", "xgb"]:
        m = models[key]
        t0 = time.perf_counter()
        if key in ("iso", "ocsvm"):
            raw = float(-m.decision_function(X)[0])
            flagged = bool(m.predict(X)[0] == -1)
            score = raw
        else:
            score = float(m.predict_proba(X)[0, 1])
            flagged = score >= 0.5
        out[key] = {"model": MODEL_LABEL[key], "score": round(score, 4),
                    "flagged": flagged,
                    "infer_ms": round((time.perf_counter() - t0) * 1000, 2)}
    return out

app = FastAPI(title="EMR Anomaly Detection Demo")

@app.get("/", response_class=HTMLResponse)
def dashboard():
    return open(f"{BASE}/dashboard.html", encoding="utf-8").read()

@app.get("/api/token")
def token(role: str = "doctor", tamper: int = 0):
    return {"token": make_token(role, bool(tamper)), "role": role}

@app.get("/api/next_record")
def next_record(kind: str = "random"):
    pool = TEST
    if kind == "anomaly":
        pool = TEST[TEST.is_anomaly == 1]
    elif kind == "normal":
        pool = TEST[TEST.is_anomaly == 0]
    r = pool.sample(1).iloc[0].to_dict()
    for k, v in r.items():
        if isinstance(v, float) and np.isnan(v):
            r[k] = 0.0
        if isinstance(v, (np.integer,)):
            r[k] = int(v)
        if isinstance(v, (np.floating,)):
            r[k] = float(v)
    return r

@app.post("/api/encounter")
def encounter(payload: dict, authorization: str = Header(None)):
    """The real middleware path: auth -> RBAC -> features -> inference."""
    t_start = time.perf_counter()
    steps = []
    # 1. JWT
    t0 = time.perf_counter()
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Missing bearer token")
    try:
        claims = pyjwt.decode(authorization[7:], SECRET, algorithms=[ALGO])
    except pyjwt.PyJWTError:
        raise HTTPException(401, "Invalid or tampered token signature")
    steps.append({"step": "JWT signature verified (stateless)",
                  "ms": round((time.perf_counter() - t0) * 1000, 2)})
    # 2. RBAC
    t0 = time.perf_counter()
    role = claims.get("role", "?")
    if "write_encounter" not in ROLES.get(role, []):
        raise HTTPException(403, f"Role '{role}' lacks write_encounter permission")
    steps.append({"step": f"RBAC: role '{role}' authorized",
                  "ms": round((time.perf_counter() - t0) * 1000, 2)})
    # 3. features
    t0 = time.perf_counter()
    rec = payload["record"]
    X = build_features(rec)
    steps.append({"step": "25-dim feature vector built (impute + scale)",
                  "ms": round((time.perf_counter() - t0) * 1000, 2)})
    # 4. inference
    scores = score_all(X)
    total_ms = round((time.perf_counter() - t_start) * 1000, 2)
    xgb = scores["xgb"]
    derived = {"dose_per_kg": round(rec["dose_mg"] / max(rec["weight_kg"], 0.1), 3),
               "shock_index": round(rec["heart_rate"] / max(rec["sbp"], 1), 3)}
    return {"decision": "FLAGGED — routed to review queue" if xgb["flagged"]
            else "ACCEPTED — written to EMR",
            "flagged": xgb["flagged"], "steps": steps, "scores": scores,
            "derived": derived, "total_ms": total_ms,
            "truth": {"is_anomaly": int(rec.get("is_anomaly", -1)),
                      "anomaly_type": rec.get("anomaly_type", "unknown")}}

@app.get("/api/summary")
def summary():
    m = RESULTS["metrics"]
    lat = BENCH["latency"]
    return {
        "metrics": {k: {x: round(m[k][x], 3) for x in
                        ["precision", "recall", "f1", "auc_roc"]}
                    for k in MODEL_LABEL.values()},
        "cm": RESULTS["cm"],
        "per_type": m["per_type_recall"],
        "latency": {k: {"mean": round(v["mean"], 1), "p95": round(v["p95"], 1)}
                    for k, v in lat.items()},
        "security": BENCH["security"],
    }

if __name__ == "__main__":
    import uvicorn
    print("\n  EMR Anomaly Detection Demo  ->  http://127.0.0.1:8000\n")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
