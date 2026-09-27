# EMR Anomaly Detection — Live Thesis Demo

Interactive demo of the thesis *"AI-Driven Electronic Medical Record (EMR) System
for Real-Time Anomaly Detection"* (Suvechchha Subedi, Purbanchal University).

It runs the **actual middleware pipeline** from the thesis — stateless JWT
authentication → RBAC → 25-feature construction → inference with the four
trained models — and shows everything live in a browser dashboard. No internet
connection needed once installed (all assets are local).

## Setup (once)

```
pip install fastapi uvicorn scikit-learn xgboost pandas numpy pyjwt
```

## Run

```
python demo_app.py
```

Then open **http://127.0.0.1:8000** in any browser.

## What to show at the defense

**Tab 1 — Live Feed** (how data is fed and tested)
- Click **Feed anomaly** / **Feed normal**, or **Auto-stream** to process a
  record every ~1.3 s.
- Middle panel shows each middleware step with its measured latency, the
  end-to-end time, and the accept/flag decision.
- Right panel accumulates a **live confusion matrix per model** (TP/FP/FN,
  precision, recall, F1) against ground truth — this is "accuracy checked in
  the backend", happening in front of the committee.
- Security demo: set role to **clerk** (→ HTTP 403) or tick **Tamper token**
  (→ HTTP 401). The request is rejected before any ML runs.

**Tab 2 — Why XGBoost?** (algorithm choice)
- Bar comparisons of precision/recall/F1/AUC and end-to-end latency for all
  four models, plus the per-error-class table showing the Isolation Forest's
  5.6% vs XGBoost's 78.9% on age–dose mismatches.

**Tab 3 — Custom Record** (the killer demo)
- Preset: a 6-year-old, 20 kg, prescribed 2,000 mg ceftriaxone — a full adult
  dose. Submit → **flagged, score ≈ 1.00**.
- Change age to 30 (weight ~62) → same dose is **accepted, score ≈ 0.01**.
- Point out: every field passes a static range check; only the *combination*
  is wrong. This is exactly what rule-based validation cannot do.

## Files

| Path | Purpose |
|------|---------|
| `demo_app.py` | FastAPI backend (the real middleware path) |
| `dashboard.html` | Offline dashboard (vanilla JS, no CDN) |
| `models/*.pkl` | The four trained models + imputer/scaler from the thesis |
| `data/emr_synthetic_dataset.csv` | 30,000-record synthetic dataset |
| `data/model_results.json` | Test-set metrics, per-class recall, CV results |
| `data/api_benchmark.json` | Latency benchmark + security test results |

Note: latencies shown live depend on the machine; benchmark figures quoted in
the thesis were measured on the 2-core reference node.
