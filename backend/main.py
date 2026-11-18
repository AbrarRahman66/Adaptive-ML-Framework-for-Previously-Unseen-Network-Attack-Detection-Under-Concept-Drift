"""
IDS backend: loads the saved models ONCE at startup, predicts, and stores results in Rayan's
MariaDB database `nids` (tables: datasets, experiments, flows, predictions, alerts, drift_events).

Run:   python -m uvicorn main:app --reload
Env:   MODEL_DIR     path to the IDS_Project folder          (default: ./IDS_Project)
       DATABASE_URL  mysql+pymysql://root:@127.0.0.1:3306/nids   (XAMPP default: user root, empty password)
       AUTO_MIGRATE  1 (default) = add the extra nullable columns from nids_migration.sql on startup

Preprocessing (must match training):
  RF + XGBoost : clean -> imputer -> predict            (NOT scaled)
  Autoencoder  : clean -> imputer -> scaler -> reconstruction error
"""
import os, json, math, logging
from contextlib import asynccontextmanager

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import tensorflow as tf
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, text

log = logging.getLogger("ids")
MODEL_DIR = os.getenv("MODEL_DIR", "IDS_Project")
DATABASE_URL = os.getenv("DATABASE_URL")
AUTO_MIGRATE = os.getenv("AUTO_MIGRATE", "1") == "1"
MAX_MISSING_FRACTION = 0.5
EXPERIMENT_NAME = "Hybrid RF+XGB+AE"          # value written to experiments.model
LABEL_ATTACK, LABEL_BENIGN, LABEL_UNSEEN = "attack", "benign", "unseen_anomaly"

S = {}                                         # everything loaded at startup


# ------------------------------------------------------------------ helpers
def severity(conf: float) -> str:
    """Same rule as Rayan's existing alerts: >=0.9 high, 0.7-0.89 medium, else low."""
    return "high" if conf >= 0.9 else "medium" if conf >= 0.7 else "low"


def jsonable(v):
    """NaN / inf are not valid JSON (the DB column has CHECK json_valid)."""
    if v is None:
        return None
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return None if (math.isnan(v) or math.isinf(v)) else v


def to_model_names(d: dict) -> dict:
    """Accept both naming styles: 'Flow Bytess' (model) and 'Flow_Bytess' (database JSON)."""
    out = {}
    for k, v in d.items():
        if k in S["feature_set"]:
            out[k] = v
        elif k in S["alias"]:
            out[S["alias"][k]] = v
    return out


def to_db_names(d: dict) -> dict:
    """Database convention: underscores instead of spaces."""
    return {k.replace(" ", "_"): jsonable(v) for k, v in d.items()}


def db_required():
    if S["engine"] is None:
        raise HTTPException(503, "Database not configured (set DATABASE_URL).")


# ------------------------------------------------------------------ startup
def ensure_experiment(engine) -> int:
    params = {"model_version": S["version"], "threshold_name": S["threshold_name"],
              "threshold": S["threshold"], "n_features": len(S["features"]),
              "trained_on": S["meta"].get("trained_on"), "components": ["RandomForest", "XGBoost", "Autoencoder"]}
    with engine.begin() as c:
        row = c.execute(text(
            "SELECT exp_id FROM experiments WHERE model=:m "
            "AND JSON_UNQUOTE(JSON_EXTRACT(params,'$.model_version'))=:v ORDER BY exp_id DESC LIMIT 1"),
            {"m": EXPERIMENT_NAME, "v": S["version"]}).first()
        if row:
            return int(row[0])
        r = c.execute(text("INSERT INTO experiments (model, params, held_out_attack) VALUES (:m, :p, 'none')"),
                      {"m": EXPERIMENT_NAME, "p": json.dumps(params)})
        return int(r.lastrowid)


@asynccontextmanager
async def lifespan(app: FastAPI):
    p = lambda *a: os.path.join(MODEL_DIR, *a)
    S["rf"] = joblib.load(p("models", "random_forest.joblib"))
    S["xgb"] = xgb.XGBClassifier(); S["xgb"].load_model(p("models", "xgboost.json"))
    S["ae"] = tf.keras.models.load_model(p("models", "autoencoder.keras"))
    S["imputer"] = joblib.load(p("preprocessing", "imputer.joblib"))
    S["scaler"] = joblib.load(p("preprocessing", "scaler.joblib"))
    S["features"] = json.load(open(p("configuration", "feature_names.json")))
    S["feature_set"] = set(S["features"])
    S["alias"] = {f.replace(" ", "_"): f for f in S["features"]}
    th = json.load(open(p("configuration", "thresholds.json")))
    S["thresholds"] = {k: v for k, v in th.items() if k.startswith("threshold_")}
    S["threshold_name"] = th["selected_threshold"]
    S["threshold"] = th[th["selected_threshold"]]
    S["meta"] = json.load(open(p("configuration", "model_metadata.json")))
    S["version"] = S["meta"]["model_version"]

    try:
        from river.drift import ADWIN
        S["adwin"] = ADWIN()
    except Exception as e:
        S["adwin"] = None
        log.warning("river not available, drift detection disabled: %s", e)
    S["n_seen"] = 0
    S["last_drift"] = None

    S["engine"], S["exp_id"] = None, None
    if DATABASE_URL:
        S["engine"] = create_engine(DATABASE_URL, pool_pre_ping=True)
        if AUTO_MIGRATE:
            sql = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "nids_migration.sql")).read()
            sql = "\n".join(l for l in sql.splitlines() if not l.strip().startswith("--"))
            with S["engine"].begin() as c:
                for stmt in [s for s in sql.split(";") if s.strip()]:
                    c.execute(text(stmt))
        S["exp_id"] = ensure_experiment(S["engine"])
    log.info("Models loaded (v%s). DB: %s, exp_id: %s", S["version"], bool(S["engine"]), S["exp_id"])
    yield


app = FastAPI(title="Adaptive NIDS API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class Flow(BaseModel):
    features: dict[str, float | None]


class Batch(BaseModel):
    flows: list[Flow]


# ------------------------------------------------------------------ models
def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.reindex(columns=S["features"])
    df = df.apply(pd.to_numeric, errors="coerce")
    return df.replace([np.inf, -np.inf], np.nan)


def verdict(rf_pred: int, xgb_pred: int, anomaly: bool) -> str:
    if rf_pred == 1 or xgb_pred == 1:
        return "known_attack"
    if anomaly:
        return "possible_unseen_attack"
    return "normal"


def run_models(rows: list[dict]) -> list[dict]:
    df = clean(pd.DataFrame([to_model_names(r) for r in rows]))
    Xi = pd.DataFrame(S["imputer"].transform(df), columns=S["features"])
    Xs = S["scaler"].transform(Xi)
    rf_pred = S["rf"].predict(Xi);   rf_prob = S["rf"].predict_proba(Xi)[:, 1]
    xg_pred = S["xgb"].predict(Xi);  xg_prob = S["xgb"].predict_proba(Xi)[:, 1]
    err = np.mean((Xs - S["ae"](Xs, training=False).numpy()) ** 2, axis=1)

    thr, out = S["threshold"], []
    for i in range(len(Xi)):
        anomaly = bool(err[i] > thr)
        v = verdict(int(rf_pred[i]), int(xg_pred[i]), anomaly)
        avg_att = float((rf_prob[i] + xg_prob[i]) / 2)
        if v == "known_attack":
            label, conf = LABEL_ATTACK, avg_att
        elif v == "possible_unseen_attack":
            label, conf = LABEL_UNSEEN, float(err[i] / (err[i] + thr))   # 0.5 at the threshold, ->1 as error grows
        else:
            label, conf = LABEL_BENIGN, 1.0 - avg_att
        out.append({
            "rf_pred": int(rf_pred[i]), "rf_prob_attack": float(rf_prob[i]),
            "xgb_pred": int(xg_pred[i]), "xgb_prob_attack": float(xg_prob[i]),
            "recon_error": float(err[i]), "anomaly_flag": anomaly,
            "threshold_name": S["threshold_name"], "threshold": thr,
            "final_verdict": v, "predicted_label": label, "confidence": round(conf, 4),
            "model_version": S["version"],
        })
    return out


# ------------------------------------------------------------------ database writes
def write_prediction(c, flow_id: int, r: dict) -> dict:
    """Insert one row into predictions (+ alerts for attacks / unseen anomalies). Returns the new ids."""
    res = c.execute(text(
        "INSERT INTO predictions (exp_id, flow_id, predicted_label, confidence, rf_pred, rf_prob_attack, "
        "xgb_pred, xgb_prob_attack, recon_error, anomaly_flag, final_verdict) VALUES "
        "(:e, :f, :l, :c, :rp, :rpp, :xp, :xpp, :re, :a, :v)"),
        {"e": S["exp_id"], "f": flow_id, "l": r["predicted_label"], "c": r["confidence"],
         "rp": r["rf_pred"], "rpp": r["rf_prob_attack"], "xp": r["xgb_pred"], "xpp": r["xgb_prob_attack"],
         "re": r["recon_error"], "a": int(r["anomaly_flag"]), "v": r["final_verdict"]})
    ids = {"flow_id": flow_id, "pred_id": int(res.lastrowid), "alert_id": None}
    if r["final_verdict"] != "normal":
        a = c.execute(text("INSERT INTO alerts (pred_id, severity, status) VALUES (:p, :s, 'new')"),
                      {"p": ids["pred_id"], "s": severity(r["confidence"])})
        ids["alert_id"] = int(a.lastrowid)
    return ids


def update_drift(err: float):
    """Feed reconstruction error to ADWIN; record an event in drift_events when drift is detected."""
    S["n_seen"] += 1
    det = S.get("adwin")
    if det is None:
        return
    det.update(err)
    if det.drift_detected:
        S["last_drift"] = {"n_samples_seen": S["n_seen"], "score": err}
        if S["engine"] is not None:
            try:
                with S["engine"].begin() as c:
                    c.execute(text("INSERT INTO drift_events (exp_id, detected_at, detector, score) "
                                   "VALUES (:e, NOW(), 'ADWIN', :s)"), {"e": S["exp_id"], "s": err})
            except Exception as e:
                log.error("drift write failed: %s", e)


def process(flows: list[Flow], store: bool) -> list[dict]:
    for f in flows:
        known = len(to_model_names(f.features))
        if known < len(S["features"]) * (1 - MAX_MISSING_FRACTION):
            raise HTTPException(422, f"Too few recognised features ({known}/{len(S['features'])}). "
                                     "Use the names from GET /features (spaces or underscores both work).")
    results = run_models([f.features for f in flows])
    for f, r in zip(flows, results):
        r["stored"] = False
        if store and S["engine"] is not None:
            try:
                with S["engine"].begin() as c:
                    fid = c.execute(text(
                        "INSERT INTO flows (dataset_id, ts, features, label, is_attack, split) "
                        "VALUES (NULL, NOW(), :f, NULL, NULL, NULL)"),
                        {"f": json.dumps(to_db_names(f.features))}).lastrowid
                    r.update(write_prediction(c, int(fid), r))
                r["stored"] = True
            except Exception as e:
                log.error("DB write failed: %s", e)
        update_drift(r["recon_error"])
    return results


# ------------------------------------------------------------------ endpoints
@app.get("/health")
def health():
    return {"status": "ok", "model_version": S.get("version"), "n_features": len(S.get("features", [])),
            "threshold": S.get("threshold_name"), "database": S.get("engine") is not None,
            "exp_id": S.get("exp_id"), "drift_detector": S.get("adwin") is not None}


@app.get("/features")
def features():
    return {"feature_names": S["features"], "thresholds": S["thresholds"], "selected": S["threshold_name"],
            "note": "Names with spaces or underscores are both accepted."}


@app.post("/predict")
def predict(flow: Flow, store: bool = True):
    """Score one NEW flow. With store=true it is saved to flows, predictions (and alerts)."""
    return process([flow], store)[0]


@app.post("/predict/batch")
def predict_batch(batch: Batch, store: bool = True):
    if not batch.flows:
        raise HTTPException(422, "No flows supplied.")
    return {"count": len(batch.flows), "results": process(batch.flows, store)}


@app.post("/predict/stored")
def predict_stored(limit: int = Query(200, ge=1, le=2000), split: str | None = Query("test")):
    """Score flows ALREADY in the database that this model has not scored yet.
    Defaults to split='test' because the model was trained on the 'train' flows (use split=all to include them)."""
    db_required()
    where, params = "", {"e": S["exp_id"], "l": limit}
    if split and split != "all":
        where, params["s"] = "AND f.split = :s", split
    with S["engine"].connect() as c:
        rows = c.execute(text(
            "SELECT f.flow_id, f.features, f.is_attack FROM flows f "
            "WHERE f.features IS NOT NULL AND NOT EXISTS "
            "(SELECT 1 FROM predictions p WHERE p.flow_id = f.flow_id AND p.exp_id = :e) "
            f"{where} ORDER BY f.flow_id LIMIT :l"), params).mappings().all()
    if not rows:
        return {"scored": 0, "message": "No unscored flows found for this filter."}
    feats = [json.loads(r["features"]) for r in rows]
    results = run_models(feats)
    counts, agree, labelled = {}, 0, 0
    with S["engine"].begin() as c:
        for row, r in zip(rows, results):
            write_prediction(c, int(row["flow_id"]), r)
            counts[r["final_verdict"]] = counts.get(r["final_verdict"], 0) + 1
            if row["is_attack"] is not None:
                labelled += 1
                agree += int((r["final_verdict"] != "normal") == bool(row["is_attack"]))
    for r in results:
        update_drift(r["recon_error"])
    return {"scored": len(results), "verdicts": counts,
            "agreement_with_ground_truth": round(agree / labelled, 4) if labelled else None}


@app.get("/predictions/recent")
def recent(limit: int = Query(50, ge=1, le=500), all_experiments: bool = False):
    db_required()
    cond = "" if all_experiments else "WHERE p.exp_id = :e"
    with S["engine"].connect() as c:
        rows = c.execute(text(
            "SELECT p.pred_id, p.exp_id, p.flow_id, p.predicted_label, p.confidence, p.final_verdict, "
            "p.rf_prob_attack, p.xgb_prob_attack, p.recon_error, p.anomaly_flag, p.created_at, "
            "a.alert_id, a.severity, a.status FROM predictions p "
            f"LEFT JOIN alerts a ON a.pred_id = p.pred_id {cond} ORDER BY p.pred_id DESC LIMIT :l"),
            {"e": S["exp_id"], "l": limit}).mappings().all()
    return [dict(r) for r in rows]


@app.get("/alerts/recent")
def alerts_recent(limit: int = Query(50, ge=1, le=500), status: str | None = None):
    db_required()
    cond = "WHERE a.status = :s" if status else ""
    with S["engine"].connect() as c:
        rows = c.execute(text(
            "SELECT a.alert_id, a.severity, a.status, p.pred_id, p.flow_id, p.predicted_label, "
            "p.confidence, p.final_verdict, p.recon_error FROM alerts a "
            f"JOIN predictions p ON p.pred_id = a.pred_id {cond} ORDER BY a.alert_id DESC LIMIT :l"),
            {"s": status, "l": limit}).mappings().all()
    return [dict(r) for r in rows]


@app.patch("/alerts/{alert_id}")
def alert_update(alert_id: int, status: str = Query(..., pattern="^(new|ack|closed)$")):
    db_required()
    with S["engine"].begin() as c:
        n = c.execute(text("UPDATE alerts SET status=:s WHERE alert_id=:a"), {"s": status, "a": alert_id}).rowcount
    if not n:
        raise HTTPException(404, "Alert not found (or status unchanged).")
    return {"alert_id": alert_id, "status": status}


@app.get("/drift/status")
def drift_status():
    out = {"detector": "ADWIN" if S.get("adwin") else None, "n_samples_seen": S.get("n_seen", 0),
           "last_drift": S.get("last_drift"), "model_version": S.get("version")}
    if S["engine"] is not None:
        with S["engine"].connect() as c:
            out["events_total"] = c.execute(text("SELECT count(*) FROM drift_events")).scalar()
    return out
