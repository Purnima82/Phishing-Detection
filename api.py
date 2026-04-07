"""
api.py — PhishGuard AI | FastAPI Backend
Run:  uvicorn api:app --host 0.0.0.0 --port 8000
"""

import os, pickle, warnings
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

warnings.filterwarnings("ignore")

from feature_engineering import (
    extract_features, extra_security_checks,
    get_feature_dict, FEATURE_COLUMNS
)
from database import save_scan, get_history, get_stats, delete_all

# ── Always resolve paths relative to THIS file ───────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def _path(filename):
    return os.path.join(BASE_DIR, filename)

# ── Load model ────────────────────────────────────────────────
def _load_model():
    for m, s in [
        ("phishing_model_rf.pkl", "scaler_rf.pkl"),
        ("phishing_model.pkl",    "scaler.pkl"),
    ]:
        mp, sp = _path(m), _path(s)
        if os.path.exists(mp) and os.path.exists(sp):
            try:
                return pickle.load(open(mp, "rb")), pickle.load(open(sp, "rb"))
            except Exception:
                continue
    raise RuntimeError(
        "No model file found. Either commit phishing_model_rf.pkl "
        "to your repo or ensure build.sh runs model_evaluation.py first."
    )

model, scaler = _load_model()

# ── App ───────────────────────────────────────────────────────
app = FastAPI(
    title="PhishGuard AI",
    description="Phishing website detection — Random Forest ML",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Models ────────────────────────────────────────────────────
class ScanRequest(BaseModel):
    url: str

class ScanResponse(BaseModel):
    url:            str
    verdict:        str
    risk_score:     float
    ml_probability: float
    rule_boost:     float
    features:       dict
    safe:           bool

# ── Routes ───────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def root():
    html = _path("index.html")
    if os.path.exists(html):
        return FileResponse(html, media_type="text/html")
    return JSONResponse({"status": "PhishGuard AI running", "docs": "/docs"})


@app.post("/scan", response_model=ScanResponse)
def scan_url(req: ScanRequest):
    url = req.url.strip()
    if not url:
        raise HTTPException(400, "URL cannot be empty")
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    try:
        raw    = extract_features(url)
        df_f   = pd.DataFrame([raw], columns=FEATURE_COLUMNS)
        scaled = scaler.transform(df_f)
        ml_p   = float(model.predict_proba(scaled)[0][1])
        extra  = float(extra_security_checks(url))
        final  = min(ml_p + extra, 1.0)

        if final >= 0.70:   verdict = "Phishing"
        elif final <= 0.30: verdict = "Legitimate"
        else:               verdict = "Suspicious"

        save_scan(url, verdict, final)

        return ScanResponse(
            url=url, verdict=verdict,
            risk_score=round(final, 4),
            ml_probability=round(ml_p, 4),
            rule_boost=round(extra, 4),
            features=get_feature_dict(url),
            safe=(verdict == "Legitimate"),
        )
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/history")
def history(limit: int = 100):
    return get_history(limit)


@app.get("/stats")
def stats():
    return get_stats()


@app.delete("/history")
def clear_history():
    delete_all()
    return {"status": "cleared"}


@app.get("/health")
def health():
    return {"status": "ok", "model": type(model).__name__}


# ── Mount static/ folder if it exists ────────────────────────
static_dir = _path("static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
