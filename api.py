"""
api.py — PhishGuard AI  |  FastAPI Backend
Serves the ML model as a REST API + static website files.

Run:  uvicorn api:app --host 0.0.0.0 --port 8000 --reload
Docs: http://localhost:8000/docs
"""

import os, json, pickle, warnings
from typing import Optional
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

warnings.filterwarnings("ignore")

from feature_engineering import extract_features, extra_security_checks, get_feature_dict, FEATURE_COLUMNS
from database import save_scan, get_history, get_stats, delete_all

# ── Load model ────────────────────────────────────────────────
def _load_model():
    for m_path, s_path in [
        ("phishing_model_rf.pkl", "scaler_rf.pkl"),
        ("phishing_model.pkl",    "scaler.pkl"),
    ]:
        if os.path.exists(m_path):
            try:
                return pickle.load(open(m_path, "rb")), pickle.load(open(s_path, "rb"))
            except Exception:
                continue
    raise RuntimeError("No model file found. Run model_evaluation.py first.")

model, scaler = _load_model()

# ── FastAPI app ───────────────────────────────────────────────
app = FastAPI(
    title="PhishGuard AI API",
    description="Real-time phishing website detection powered by Random Forest ML",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request / Response models ─────────────────────────────────
class ScanRequest(BaseModel):
    url: str

class ScanResponse(BaseModel):
    url:           str
    verdict:       str          # "Phishing" | "Suspicious" | "Legitimate"
    risk_score:    float        # 0.0 – 1.0
    ml_probability: float
    rule_boost:    float
    features:      dict
    safe:          bool

# ── Routes ───────────────────────────────────────────────────

@app.get("/")
def root():
    """Serve the frontend website."""
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return JSONResponse({"status": "PhishGuard AI API running", "docs": "/docs"})


@app.post("/scan", response_model=ScanResponse)
def scan_url(req: ScanRequest):
    """
    Analyse a URL and return phishing verdict + risk score.
    """
    url = req.url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="URL cannot be empty")
    if not (url.startswith("http://") or url.startswith("https://")):
        url = "http://" + url      # auto-prefix for convenience

    try:
        raw     = extract_features(url)
        df_f    = pd.DataFrame([raw], columns=FEATURE_COLUMNS)
        scaled  = scaler.transform(df_f)
        ml_prob = float(model.predict_proba(scaled)[0][1])
        extra   = float(extra_security_checks(url))
        final   = min(ml_prob + extra, 1.0)

        if final >= 0.70:
            verdict = "Phishing"
        elif final <= 0.30:
            verdict = "Legitimate"
        else:
            verdict = "Suspicious"

        features = get_feature_dict(url)
        save_scan(url, verdict, final)

        return ScanResponse(
            url=url,
            verdict=verdict,
            risk_score=round(final, 4),
            ml_probability=round(ml_prob, 4),
            rule_boost=round(extra, 4),
            features=features,
            safe=(verdict == "Legitimate"),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history")
def history(limit: int = 100):
    """Return recent scan history."""
    return get_history(limit)


@app.get("/stats")
def stats():
    """Return aggregate statistics."""
    return get_stats()


@app.delete("/history")
def clear_history():
    """Delete all scan history."""
    delete_all()
    return {"status": "cleared"}


@app.get("/health")
def health():
    return {"status": "ok", "model": type(model).__name__}


# ── Serve static files if present ────────────────────────────
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
