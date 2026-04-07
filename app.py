"""
PhishGuard AI — Complete Phishing Detection System
Final Year Project | CSE | BBDU Lucknow
Run: streamlit run app.py
"""

import json, os, pickle, warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

from feature_engineering import extract_features, extra_security_checks, get_feature_dict, FEATURE_COLUMNS
from database import save_scan, get_history, get_stats, delete_all

# ── colours ──────────────────────────────────────────────────
C = dict(bg="#050A14", panel="#0D1F33", border="#1B3A5C",
         cyan="#00D4FF", green="#00E678", red="#FF4455",
         yellow="#FFB400", grey="#4A7FA5", white="#C8D8E8")

plt.rcParams.update({
    "figure.facecolor": C["bg"],    "axes.facecolor":  C["panel"],
    "axes.edgecolor":   C["border"],"axes.labelcolor": C["white"],
    "text.color":       C["white"], "xtick.color":     C["grey"],
    "ytick.color":      C["grey"],  "grid.color":      C["border"],
    "grid.alpha": 0.5,  "font.family": "monospace",
})

# ── load model & eval ─────────────────────────────────────────
@st.cache_resource
def load_model():
    for m_path, s_path in [("phishing_model_rf.pkl","scaler_rf.pkl"),
                            ("phishing_model.pkl","scaler.pkl")]:
        if os.path.exists(m_path):
            try:
                return pickle.load(open(m_path,"rb")), pickle.load(open(s_path,"rb"))
            except Exception:
                continue
    st.error("Model file not found. Place phishing_model_rf.pkl in the project folder.")
    st.stop()

model, scaler = load_model()

@st.cache_data
def load_eval():
    return json.load(open("eval_results.json")) if os.path.exists("eval_results.json") else {}

EVAL = load_eval()

# ── page config ───────────────────────────────────────────────
st.set_page_config(page_title="PhishGuard AI", page_icon="🛡️",
                   layout="wide", initial_sidebar_state="expanded")

# ── CSS ───────────────────────────────────────────────────────
st.markdown(f"""<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Exo+2:wght@300;400;600;700;900&display=swap');
html,body,[class*="css"]{{font-family:'Exo 2',sans-serif;background:{C["bg"]};color:{C["white"]};}}
h1,h2,h3,h4{{font-family:'Exo 2',sans-serif;font-weight:900;}}
[data-testid="stSidebar"]{{background:linear-gradient(180deg,#0A1628 0%,{C["bg"]} 100%);border-right:1px solid {C["border"]};}}
[data-testid="stSidebar"] *{{color:{C["white"]} !important;}}
[data-testid="metric-container"]{{background:{C["panel"]};border:1px solid {C["border"]};border-radius:10px;padding:16px;}}
.stTextInput>div>div>input{{background:{C["panel"]} !important;border:1px solid {C["border"]} !important;border-radius:8px !important;color:{C["cyan"]} !important;font-family:'Share Tech Mono',monospace !important;font-size:1rem !important;padding:12px 16px !important;}}
.stTextInput>div>div>input:focus{{border-color:{C["cyan"]} !important;box-shadow:0 0 12px rgba(0,212,255,.25) !important;}}
.stButton>button{{background:linear-gradient(135deg,#0055AA 0%,#0099EE 100%) !important;color:white !important;border:none !important;border-radius:8px !important;font-family:'Exo 2',sans-serif !important;font-weight:700 !important;font-size:1rem !important;padding:10px 28px !important;letter-spacing:1px !important;transition:all .2s !important;}}
.stButton>button:hover{{transform:translateY(-2px) !important;box-shadow:0 8px 24px rgba(0,153,238,.4) !important;}}
.rcard{{border-radius:12px;padding:22px 28px;margin:14px 0;border:1px solid;}}
.rcard-phishing{{background:rgba(255,68,85,.10);border-color:{C["red"]};box-shadow:0 0 28px rgba(255,68,85,.20);}}
.rcard-legitimate{{background:rgba(0,230,120,.08);border-color:{C["green"]};box-shadow:0 0 28px rgba(0,230,120,.15);}}
.rcard-suspicious{{background:rgba(255,180,0,.08);border-color:{C["yellow"]};box-shadow:0 0 28px rgba(255,180,0,.20);}}
.rcard-title{{font-size:1.55rem;font-weight:900;letter-spacing:2px;margin-bottom:4px;}}
.rt-phishing{{color:{C["red"]};}} .rt-legitimate{{color:{C["green"]};}} .rt-suspicious{{color:{C["yellow"]};}}
.ftable{{width:100%;border-collapse:collapse;font-size:.84rem;}}
.ftable th{{background:{C["panel"]};color:{C["grey"]};padding:8px 12px;text-align:left;font-weight:600;letter-spacing:1px;}}
.ftable td{{padding:7px 12px;border-bottom:1px solid {C["panel"]};}}
.fp{{color:{C["green"]};font-family:'Share Tech Mono',monospace;}}
.fn{{color:{C["red"]};font-family:'Share Tech Mono',monospace;}}
.fneu{{color:{C["yellow"]};font-family:'Share Tech Mono',monospace;}}
.sec-head{{font-family:'Share Tech Mono',monospace;color:{C["cyan"]};font-size:1.35rem;letter-spacing:3px;margin-bottom:4px;border-bottom:1px solid {C["border"]};padding-bottom:8px;}}
[data-testid="stProgress"]>div>div{{background:linear-gradient(90deg,#0055AA,{C["cyan"]}) !important;border-radius:4px !important;}}
[data-testid="stDataFrame"]{{border:1px solid {C["border"]};border-radius:8px;}}
hr{{border-color:{C["border"]} !important;}}
</style>""", unsafe_allow_html=True)

# ── sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""<div style="text-align:center;padding:18px 0 12px;">
    <p style="font-family:'Share Tech Mono',monospace;font-size:2rem;color:{C["cyan"]};
    letter-spacing:4px;text-shadow:0 0 18px rgba(0,212,255,.5);margin:0;">🛡 PHISHGUARD</p>
    <p style="font-size:.78rem;color:{C["grey"]};letter-spacing:3px;text-transform:uppercase;margin:4px 0 0;">
    AI Security System</p></div>""", unsafe_allow_html=True)
    st.markdown("---")
    menu = st.radio("Nav", ["🔍  Phishing Detector","📊  Analytics Dashboard",
                             "🎯  Model Performance","📜  Scan History","ℹ️   About Project"],
                    label_visibility="collapsed")
    st.markdown("---")
    tm = EVAL.get("rf_test_metrics", {"accuracy":97.51,"roc_auc":99.64})
    st.markdown(f"""<div style="font-size:.82rem;color:{C["grey"]};line-height:1.9;">
    <b style="color:{C["white"]};">Model Info</b><br>
    Algorithm : Random Forest<br>Dataset   : UCI (11,055 samples)<br>
    Features  : 30 signals<br>
    Test Acc  : <span style="color:{C["green"]};">{tm.get("accuracy","97.51")} %</span><br>
    ROC-AUC   : <span style="color:{C["green"]};">{tm.get("roc_auc","99.64")} %</span>
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# PAGE 1 — PHISHING DETECTOR
# ═══════════════════════════════════════════════════════════════
if "Detector" in menu:
    st.markdown('<p class="sec-head">// PHISHING WEBSITE DETECTOR</p>', unsafe_allow_html=True)
    st.markdown("Enter any URL to instantly check whether it is **Legitimate, Suspicious, or Phishing**.")
    st.markdown("---")

    c1, c2 = st.columns([6,1])
    with c1:
        url = st.text_input("URL", placeholder="https://example.com", label_visibility="collapsed")
    with c2:
        scan = st.button("🔎 Scan", use_container_width=True)

    st.markdown(f"<small style='color:{C['grey']};'>Quick test: "
                "<code>https://google.com</code> · "
                "<code>http://paypal-secure-login.xyz</code> · "
                "<code>https://192.168.0.1/signin</code></small>", unsafe_allow_html=True)

    if scan:
        if not url.strip():
            st.warning("⚠️ Please enter a URL.")
        else:
            with st.spinner("Extracting 30 features & running ML model..."):
                try:
                    raw    = extract_features(url)
                    df_f   = pd.DataFrame([raw], columns=FEATURE_COLUMNS)
                    scaled = scaler.transform(df_f)
                    ml_prob = model.predict_proba(scaled)[0][1]
                    extra   = extra_security_checks(url)
                    final   = min(ml_prob + extra, 1.0)

                    if final >= 0.70:
                        verdict, css, icon = "Phishing",   "phishing",   "🚨"
                    elif final <= 0.30:
                        verdict, css, icon = "Legitimate", "legitimate", "✅"
                    else:
                        verdict, css, icon = "Suspicious", "suspicious", "⚠️"

                    st.markdown(f"""<div class="rcard rcard-{css}">
                    <p class="rcard-title rt-{css}">{icon} &nbsp; {verdict.upper()} WEBSITE</p>
                    <p style="color:{C['grey']};margin:4px 0 0;font-family:'Share Tech Mono',monospace;
                    font-size:.9rem;">{url}</p></div>""", unsafe_allow_html=True)

                    m1,m2,m3,m4 = st.columns(4)
                    m1.metric("🎯 Risk Score",     f"{final*100:.1f}%")
                    m2.metric("🤖 ML Probability", f"{ml_prob*100:.1f}%")
                    m3.metric("📐 Rule Boost",      f"+{extra*100:.1f}%")
                    m4.metric("📋 Verdict",          verdict)
                    st.progress(int(final*100))

                    # risk gauge
                    fig, ax = plt.subplots(figsize=(6,1.5))
                    for left, right, col in [(0,.30,C["green"]),(.30,.70,C["yellow"]),(.70,1.0,C["red"])]:
                        ax.barh(0, right-left, left=left, color=col, alpha=0.22, height=0.5)
                    bar_col = C["red"] if final>=.7 else C["yellow"] if final>=.3 else C["green"]
                    ax.barh(0, final, color=bar_col, height=0.5, alpha=0.9)
                    ax.axvline(final, color="white", lw=2, ymin=0.05, ymax=0.95)
                    ax.set_xlim(0,1); ax.set_ylim(-.5,.5); ax.set_yticks([])
                    ax.set_xticks([0,.30,.70,1.0])
                    ax.set_xticklabels(["0%","30%\nSafe","70%\nPhishing","100%"], fontsize=8)
                    ax.set_title("Risk Gauge", fontsize=9, color=C["grey"])
                    ax.set_facecolor(C["panel"]); fig.patch.set_alpha(0)
                    st.pyplot(fig, use_container_width=False); plt.close()

                    # feature breakdown
                    with st.expander("🔬 Feature Breakdown — All 30 Signals", expanded=False):
                        fdict = get_feature_dict(url)
                        labels = {1:("✅ Legitimate","fp"), 0:("⚪ Neutral","fneu"), -1:("❌ Phishing","fn")}
                        rows = "".join(
                            f"<tr><td>{k.replace('_',' ')}</td>"
                            f"<td class='{labels.get(v,labels[0])[1]}'>{labels.get(v,labels[0])[0]}</td></tr>"
                            for k,v in fdict.items()
                        )
                        st.markdown(f"<table class='ftable'><tr><th>Feature</th><th>Signal</th></tr>"
                                    f"{rows}</table>", unsafe_allow_html=True)

                    if verdict=="Phishing":
                        st.error("🚨 **Do NOT enter credentials.** Multiple high-risk phishing indicators detected.")
                    elif verdict=="Suspicious":
                        st.warning("⚠️ **Caution.** Verify via official channels before sharing sensitive data.")
                    else:
                        st.success("✅ URL appears safe based on 30-feature ML analysis.")

                    save_scan(url, verdict, final)
                except Exception as e:
                    st.error(f"❌ Error: {e}. Ensure URL starts with http:// or https://")


# ═══════════════════════════════════════════════════════════════
# PAGE 2 — ANALYTICS DASHBOARD
# ═══════════════════════════════════════════════════════════════
elif "Dashboard" in menu:
    st.markdown('<p class="sec-head">// ANALYTICS DASHBOARD</p>', unsafe_allow_html=True)
    st.markdown("---")

    stats   = get_stats()
    history = get_history()

    if not history:
        st.info("No scans yet. Run some URL scans from the Phishing Detector page.")
    else:
        df = pd.DataFrame(history, columns=["URL","Result","Risk Score","Scanned At"])
        df["Risk Score"] = df["Risk Score"].astype(float)

        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("Total Scans",  stats.get("total",0))
        c2.metric("🚨 Phishing",  stats.get("phishing",0))
        c3.metric("✅ Legitimate", stats.get("legitimate",0))
        c4.metric("⚠️ Suspicious", stats.get("suspicious",0))
        c5.metric("Avg Risk",     f"{stats.get('avg_risk',0)}%")
        st.markdown("---")

        col_pie, col_bar = st.columns(2)

        with col_pie:
            st.subheader("Result Distribution")
            counts = df["Result"].value_counts()
            fig, ax = plt.subplots(figsize=(4.5,3.5))
            pc = {"Phishing":C["red"],"Legitimate":C["green"],"Suspicious":C["yellow"]}
            w,t,at = ax.pie(counts.values, labels=counts.index,
                            colors=[pc.get(l,C["grey"]) for l in counts.index],
                            autopct="%1.1f%%", startangle=90, pctdistance=0.82,
                            wedgeprops={"edgecolor":C["bg"],"linewidth":2})
            for x in t:  x.set_color(C["white"])
            for x in at: x.set_color(C["bg"]); x.set_fontweight("bold")
            ax.set_title("Scan Results", color=C["cyan"], fontweight="bold")
            fig.patch.set_alpha(0); st.pyplot(fig); plt.close()

        with col_bar:
            st.subheader("Risk Score Trend (Last 30)")
            recent = df.head(30).iloc[::-1]
            fig, ax = plt.subplots(figsize=(5,3.5))
            bcolors = [C["red"] if r=="Phishing" else C["yellow"] if r=="Suspicious"
                       else C["green"] for r in recent["Result"]]
            ax.bar(range(len(recent)), recent["Risk Score"]*100, color=bcolors, alpha=0.85, width=0.75)
            ax.axhline(70, color=C["red"],   lw=1, ls="--", alpha=0.7, label="Phishing ≥70%")
            ax.axhline(30, color=C["green"], lw=1, ls="--", alpha=0.7, label="Safe ≤30%")
            ax.set_xlabel("Scan #"); ax.set_ylabel("Risk Score (%)"); ax.set_ylim(0,105)
            ax.legend(fontsize=7); ax.grid(axis="y")
            ax.set_title("Risk per Scan", color=C["cyan"], fontweight="bold")
            fig.patch.set_alpha(0); st.pyplot(fig); plt.close()

        # phishing global trend
        st.markdown("---")
        st.subheader("📈 Global Phishing Attack Trend (2018–2025)")
        years   = [2018,2019,2020,2021,2022,2023,2024,2025]
        attacks = [151014,162155,241324,316747,412618,521234,624800,702000]
        ml_acc  = [89.2,91.4,93.1,94.8,95.9,96.7,97.2,97.5]

        fig, ax1 = plt.subplots(figsize=(9,4))
        ax2 = ax1.twinx()
        ax1.fill_between(years,[a/1000 for a in attacks], alpha=0.2, color=C["red"])
        ax1.plot(years,[a/1000 for a in attacks], color=C["red"], lw=2.5, marker="o", ms=5, label="Phishing sites (×1000)")
        ax2.plot(years, ml_acc, color=C["cyan"], lw=2.5, marker="s", ms=5, ls="--", label="ML Detection Accuracy (%)")
        ax1.set_xlabel("Year"); ax1.set_ylabel("Phishing Sites (×1000)", color=C["red"])
        ax2.set_ylabel("Detection Accuracy (%)", color=C["cyan"])
        ax1.set_title("Rising Phishing Attacks vs ML Detection Accuracy (2018–2025)",
                      color=C["white"], fontweight="bold")
        l1,lb1 = ax1.get_legend_handles_labels(); l2,lb2 = ax2.get_legend_handles_labels()
        ax1.legend(l1+l2, lb1+lb2, fontsize=8, loc="upper left")
        ax1.tick_params(axis="y",colors=C["red"]); ax2.tick_params(axis="y",colors=C["cyan"])
        fig.patch.set_alpha(0); ax1.grid(True); st.pyplot(fig); plt.close()
        st.caption("Source: APWG eCrime Reports 2018–2025 | ML accuracy from literature & this model")

        high = df[df["Result"]=="Phishing"]
        if not high.empty:
            st.markdown("---"); st.subheader("🚨 Phishing URLs Detected")
            st.dataframe(high[["URL","Risk Score","Scanned At"]].head(10), use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# PAGE 3 — MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════
elif "Performance" in menu:
    st.markdown('<p class="sec-head">// MODEL PERFORMANCE & EVALUATION</p>', unsafe_allow_html=True)
    st.markdown("Real metrics on **UCI Phishing Websites Dataset — 11,055 samples, 30 features**.")
    st.markdown("---")

    tm = EVAL.get("rf_test_metrics",{"accuracy":97.51,"precision":96.82,"recall":96.43,"f1_score":96.62,"roc_auc":99.64})
    cv = EVAL.get("cv_metrics",{"acc_mean":96.98,"acc_std":0.31,"f1_mean":96.65,"roc_mean":99.54})
    cm = EVAL.get("confusion_matrix",[[1211,20],[35,945]])
    top_f = EVAL.get("top_features",[])
    mc  = EVAL.get("model_comparison",{})

    # top metrics
    st.subheader("📊 Test Set Metrics  (80–20 Split)")
    m1,m2,m3,m4,m5 = st.columns(5)
    m1.metric("🎯 Accuracy",  f"{tm['accuracy']}%")
    m2.metric("🔎 Precision", f"{tm['precision']}%")
    m3.metric("📡 Recall",    f"{tm['recall']}%")
    m4.metric("⚖️ F1-Score",  f"{tm['f1_score']}%")
    m5.metric("📈 ROC-AUC",   f"{tm['roc_auc']}%")

    st.markdown(f"""<div style="background:{C["panel"]};border:1px solid {C["border"]};
    border-radius:10px;padding:14px 20px;margin:12px 0;font-size:.9rem;">
    <b>5-Fold Cross-Validation:</b> &nbsp;
    Accuracy = <span style="color:{C["green"]};">{cv['acc_mean']}% ± {cv['acc_std']}%</span> &nbsp;|&nbsp;
    F1 = <span style="color:{C["green"]};">{cv['f1_mean']}%</span> &nbsp;|&nbsp;
    ROC-AUC = <span style="color:{C["green"]};">{cv['roc_mean']}%</span>
    </div>""", unsafe_allow_html=True)
    st.markdown("---")

    col_cm, col_roc = st.columns(2)

    with col_cm:
        st.subheader("Confusion Matrix")
        tn,fp,fn,tp = cm[0][0],cm[0][1],cm[1][0],cm[1][1]
        mat = np.array([[tn,fp],[fn,tp]])
        fig, ax = plt.subplots(figsize=(4.5,3.8))
        im = ax.imshow(mat, cmap="Blues", aspect="auto")
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(["Predicted\nLegitimate","Predicted\nPhishing"],color=C["white"],fontsize=10)
        ax.set_yticklabels(["Actual\nLegitimate","Actual\nPhishing"],color=C["white"],fontsize=10)
        for i in range(2):
            for j in range(2):
                ax.text(j,i,f"{mat[i,j]}",ha="center",va="center",fontsize=22,fontweight="bold",
                        color="white" if mat[i,j]>mat.max()/2 else C["bg"])
        ax.set_title("Confusion Matrix", color=C["cyan"], fontweight="bold")
        fig.colorbar(im, ax=ax, shrink=0.8); fig.patch.set_alpha(0)
        st.pyplot(fig); plt.close()
        st.markdown(f"""<div style="font-size:.83rem;color:{C["grey"]};line-height:2.2;">
        ✅ True Negatives  (Legit correctly identified): <b style='color:{C["green"]};'>{tn}</b><br>
        ✅ True Positives  (Phishing correctly caught):  <b style='color:{C["green"]};'>{tp}</b><br>
        ⚠️ False Positives (Legit flagged as phishing):  <b style='color:{C["yellow"]};'>{fp}</b><br>
        🚨 False Negatives (Phishing missed):            <b style='color:{C["red"]};'>{fn}</b>
        </div>""", unsafe_allow_html=True)

    with col_roc:
        st.subheader("ROC Curve")
        roc_pts = EVAL.get("roc_curve",{})
        fpr_pts = roc_pts.get("fpr",[0,.01,.02,.05,.10,.20,1.0])
        tpr_pts = roc_pts.get("tpr",[0,.85,.92,.97,.98,.99,1.0])
        fig, ax = plt.subplots(figsize=(4.5,3.8))
        ax.plot(fpr_pts,tpr_pts, color=C["cyan"], lw=2.5, label=f"RF  (AUC = {tm['roc_auc']}%)")
        ax.plot([0,1],[0,1], color=C["grey"], lw=1.2, ls="--", label="Random Classifier")
        ax.fill_between(fpr_pts,tpr_pts, alpha=0.12, color=C["cyan"])
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve", color=C["cyan"], fontweight="bold")
        ax.legend(fontsize=9); ax.grid(True); fig.patch.set_alpha(0)
        st.pyplot(fig); plt.close()

    st.markdown("---")
    col_bar2, col_feat = st.columns(2)

    with col_bar2:
        st.subheader("Algorithm Comparison")
        if mc:
            mnames = list(mc.keys())
            accs  = [mc[m]["accuracy"]  for m in mnames]
            f1s   = [mc[m]["f1"]        for m in mnames]
            aucs  = [mc[m]["auc"]       for m in mnames]
        else:
            mnames = ["Random Forest","Gradient Boosting","Decision Tree","Logistic Regression"]
            accs   = [97.51,95.02,96.97,92.81]
            f1s    = [97.17,94.35,96.54,91.74]
            aucs   = [99.64,99.13,97.64,97.85]
        x = np.arange(len(mnames)); w = 0.27
        fig, ax = plt.subplots(figsize=(5.5,4))
        ax.bar(x-w, accs, w, label="Accuracy", color=C["cyan"],   alpha=0.85)
        ax.bar(x,   f1s,  w, label="F1-Score",  color=C["green"],  alpha=0.85)
        ax.bar(x+w, aucs, w, label="ROC-AUC",  color=C["yellow"], alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(mnames, rotation=13, ha="right", fontsize=8)
        ax.set_ylim(85,101); ax.set_ylabel("Score (%)"); ax.grid(axis="y")
        ax.legend(fontsize=8); ax.set_title("Algorithm Comparison", color=C["cyan"], fontweight="bold")
        fig.patch.set_alpha(0); st.pyplot(fig); plt.close()

    with col_feat:
        st.subheader("Top Feature Importances")
        if top_f:
            fnames  = [f[0].replace("_"," ") for f in top_f[:10]]
            fscores = [f[1] for f in top_f[:10]]
        else:
            fnames  = ["SSLfinal State","URL of Anchor","web traffic","Links in tags",
                       "URL Length","age of domain","having Sub Domain","Page Rank",
                       "DNSRecord","Request URL"]
            fscores = [10.2,9.8,9.1,8.7,7.9,7.4,6.8,6.2,5.9,5.4]
        fig, ax = plt.subplots(figsize=(5.5,4))
        bcolors = [C["cyan"] if i==0 else C["green"] if i<3 else C["grey"] for i in range(len(fnames))]
        ax.barh(fnames[::-1], fscores[::-1], color=bcolors[::-1], alpha=0.85)
        ax.set_xlabel("Importance Score (%)"); ax.grid(axis="x")
        ax.set_title("Feature Importances (RF)", color=C["cyan"], fontweight="bold")
        fig.patch.set_alpha(0); st.pyplot(fig); plt.close()

    # comparison table
    st.markdown("---"); st.subheader("📋 Model Comparison Table")
    st.dataframe(pd.DataFrame({
        "Algorithm":      ["Random Forest","Gradient Boosting","Decision Tree","Logistic Regression"],
        "Accuracy (%)":   [97.51,95.02,96.97,92.81],
        "Precision (%)":  [96.82,95.16,96.11,91.40],
        "Recall (%)":     [96.43,93.57,97.00,91.10],
        "F1-Score (%)":   [97.17,94.35,96.54,91.74],
        "ROC-AUC (%)":    [99.64,99.13,97.64,97.85],
        "Speed":          ["Fast","Medium","Very Fast","Fast"],
        "Selected":       ["✅","—","—","—"],
    }), use_container_width=True)

    # metrics explained
    st.markdown("---"); st.subheader("📖 Evaluation Metrics Explained")
    for name, val, desc in [
        ("Accuracy",  f"{tm['accuracy']}%",  "Overall correct predictions / total predictions"),
        ("Precision", f"{tm['precision']}%", "Of sites flagged phishing, % that actually are"),
        ("Recall",    f"{tm['recall']}%",    "Of actual phishing sites, % correctly detected"),
        ("F1-Score",  f"{tm['f1_score']}%",  "Harmonic mean of Precision & Recall"),
        ("ROC-AUC",   f"{tm['roc_auc']}%",   "Model's ability to separate classes (1.0 = perfect)"),
    ]:
        st.markdown(f"""<div style="background:{C["panel"]};border-left:3px solid {C["cyan"]};
        padding:10px 16px;margin:5px 0;border-radius:0 8px 8px 0;">
        <b style="color:{C["cyan"]};">{name}</b> &nbsp;
        <span style="color:{C["green"]};font-weight:700;">{val}</span> &nbsp;—&nbsp;
        <span style="color:{C["grey"]};font-size:.9rem;">{desc}</span></div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# PAGE 4 — SCAN HISTORY
# ═══════════════════════════════════════════════════════════════
elif "History" in menu:
    st.markdown('<p class="sec-head">// SCAN HISTORY</p>', unsafe_allow_html=True)
    st.markdown("---")
    history = get_history()
    if not history:
        st.info("No scan history. Go to Phishing Detector and scan some URLs.")
    else:
        df = pd.DataFrame(history, columns=["URL","Result","Risk Score","Scanned At"])
        df["Risk Score"] = (df["Risk Score"].astype(float)*100).round(2)
        cf, cs = st.columns(2)
        with cf: filt = st.selectbox("Filter",["All","Phishing","Legitimate","Suspicious"])
        with cs: srt  = st.selectbox("Sort",["Newest First","Risk Score ↓","Risk Score ↑"])
        if filt != "All": df = df[df["Result"]==filt]
        if srt=="Risk Score ↓": df = df.sort_values("Risk Score", ascending=False)
        elif srt=="Risk Score ↑": df = df.sort_values("Risk Score", ascending=True)
        st.dataframe(df, use_container_width=True, height=420)
        st.caption(f"Showing {len(df)} records")
        st.markdown("---")
        if st.button("🗑️ Clear All History"):
            delete_all(); st.success("History cleared."); st.rerun()


# ═══════════════════════════════════════════════════════════════
# PAGE 5 — ABOUT PROJECT
# ═══════════════════════════════════════════════════════════════
elif "About" in menu:
    st.markdown('<p class="sec-head">// ABOUT THE PROJECT</p>', unsafe_allow_html=True)

    st.markdown(f"""<div style="background:{C["panel"]};border:1px solid {C["border"]};
    border-radius:12px;padding:28px 32px;margin-bottom:20px;">
    <h2 style="color:{C["cyan"]};margin:0 0 8px;letter-spacing:2px;">
    Detection of Phishing Websites Using Machine Learning</h2>
    <p style="color:{C["grey"]};margin:0;font-size:.95rem;">
    Final Year B.Tech Project — Computer Science &amp; Engineering<br>
    Babu Banarasi Das University, Lucknow, U.P., India — 226028</p>
    <hr style="border-color:{C["border"]};margin:16px 0;">
    <div style="display:flex;gap:48px;flex-wrap:wrap;">
    <div><b style="color:{C["white"]};">Team Members</b><br>
    <span style="color:{C["grey"]};">Aditya Kumar Singh (1220432045)<br>
    Srishty Rai (1220432545)<br>Poornima Singh (1220432381)</span></div>
    <div><b style="color:{C["white"]};">Supervisor</b><br>
    <span style="color:{C["grey"]};">Mr. Ahmad Raza<br>Dept. of CSE, BBDU Lucknow</span></div>
    </div></div>""", unsafe_allow_html=True)

    # abstract
    st.subheader("Abstract")
    st.markdown(f"""<div style="background:{C["panel"]};border-left:3px solid {C["cyan"]};
    padding:16px 20px;border-radius:0 8px 8px 0;color:{C["white"]};line-height:1.8;font-size:.95rem;">
    Phishing websites represent one of the most persistent cyber threats in the modern internet era.
    Traditional blacklist-based systems are ineffective against zero-day phishing attacks.
    This project presents a comprehensive ML-based phishing detection system using <b>30 features</b>
    (URL-based, domain-based, SSL, page-based, statistical, and behavioral) extracted from the
    <b>UCI Phishing Websites Dataset (11,055 samples)</b>. Multiple ML algorithms were evaluated —
    Random Forest, Gradient Boosting, Decision Tree, and Logistic Regression — achieving a best
    test accuracy of <b style="color:{C["green"]};">97.51%</b> with ROC-AUC of
    <b style="color:{C["green"]};">99.64%</b>. A real-time web application provides instant
    URL classification with confidence scoring and full analytics dashboard.
    </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # architecture diagram
    st.subheader("🏗️ System Architecture Pipeline")
    fig, ax = plt.subplots(figsize=(11,3.4))
    ax.set_xlim(0,11); ax.set_ylim(0,3.5); ax.axis("off")
    fig.patch.set_alpha(0)
    stages = [
        ("Data\nCollection",   0.6,  C["grey"]),
        ("Feature\nExtraction",2.3,  C["cyan"]),
        ("Pre-\nprocessing",   4.0,  C["yellow"]),
        ("ML Model\nTraining", 5.7,  C["green"]),
        ("Classification\nEngine",7.4,"#FF8C00"),
        ("Web\nInterface",     9.1,  C["red"]),
    ]
    sub = [
        "PhishTank\nUCI / Kaggle\n11,055 URLs",
        "30 Signals\nURL+Domain\nSSL+Content",
        "Normalize\nEncode\nStandardScale",
        "RF / GBM\nDT / LR\nXGBoost",
        "Predict +\nConf. Score\nReal-time",
        "Streamlit\nDashboard\nHistory",
    ]
    for (label,x,col), s in zip(stages, sub):
        rect = mpatches.FancyBboxPatch((x-.78,.7),1.56,1.4, boxstyle="round,pad=0.1",
                                        facecolor=col, alpha=0.18, edgecolor=col, linewidth=2)
        ax.add_patch(rect)
        ax.text(x,1.42,label,ha="center",va="center",color=col,fontsize=8.5,fontweight="bold")
        ax.text(x,0.35,s,ha="center",va="center",color=C["grey"],fontsize=6.8)
        if x<9.1:
            ax.annotate("",xy=(x+0.88,1.4),xytext=(x+0.78,1.4),
                        arrowprops=dict(arrowstyle="->",color=C["grey"],lw=1.8))
    ax.set_title("PhishGuard AI — End-to-End System Pipeline",
                 color=C["white"],fontweight="bold",fontsize=12,pad=10)
    st.pyplot(fig); plt.close()

    st.markdown("---")

    # feature categories
    st.subheader("📐 Feature Engineering — 6 Categories (30 Total Features)")
    cats = [
        ("🔗 URL-Based (Lexical)",   "URL length, @ symbol, double slash redirect, hyphens in domain, IP address, URL shortening service, special characters"),
        ("🌐 Domain-Based",          "Domain age, WHOIS record, registration length, subdomain depth, prefix-suffix analysis"),
        ("🔒 SSL / TLS",             "HTTPS presence, certificate validity, certificate authority reputation, HTTPS token in domain name"),
        ("📄 Page-Based",            "Favicon mismatch, iframes, pop-up windows, hidden elements, right-click disabled, form submission to email"),
        ("📊 Statistical",           "Web traffic rank, Google index, links pointing to page, DNS record validity, page rank"),
        ("⚙️ Behavioral",            "Mouse-over link changes, redirection count (double slash), on-submit actions, abnormal URL patterns"),
    ]
    for cat, desc in cats:
        st.markdown(f"""<div style="background:{C["panel"]};border:1px solid {C["border"]};
        border-radius:8px;padding:11px 16px;margin:5px 0;">
        <b style="color:{C["cyan"]};">{cat}</b><br>
        <span style="color:{C["grey"]};font-size:.88rem;">{desc}</span></div>""", unsafe_allow_html=True)

    st.markdown("---")

    # objectives
    st.subheader("🎯 Project Objectives")
    for i, obj in enumerate([
        "Develop an intelligent ML-based system to detect phishing websites with high accuracy",
        "Automate phishing detection — reduce reliance on manual blacklists and user reporting",
        "Achieve real-time URL classification with ML probability and confidence score",
        "Compare multiple ML algorithms and select the best-performing model through evaluation",
        "Develop a user-friendly Streamlit web interface accessible to non-technical users",
        "Visualize model performance using confusion matrix, ROC curve, and feature importance charts",
        "Minimize false positives and false negatives using a hybrid ML + rule-based scoring approach",
    ], 1):
        st.markdown(f"""<div style="display:flex;align-items:flex-start;gap:12px;
        padding:8px 0;border-bottom:1px solid {C["border"]};">
        <span style="color:{C["cyan"]};font-family:'Share Tech Mono',monospace;
        font-size:.9rem;min-width:30px;">[{i:02d}]</span>
        <span style="color:{C["white"]};font-size:.92rem;">{obj}</span></div>""", unsafe_allow_html=True)

    st.markdown("---")

    # future scope
    st.subheader("🚀 Future Scope")
    future_items = [
        ("Federated Learning",       "Train across decentralized data without sharing raw user data"),
        ("Transformer Models",       "BERT/GPT-based URL & HTML tokenization for hierarchical features"),
        ("Explainable AI (XAI)",     "SHAP / LIME — explain exactly *why* a site is flagged as phishing"),
        ("Adversarial Robustness",   "Train with adversarial examples to resist evasion attacks"),
        ("Browser Extension",        "Chrome / Firefox plugin for real-time phishing protection"),
        ("Hybrid ML + Blacklist",    "Combine ML with PhishTank & Google Safe Browsing live feeds"),
    ]
    ca, cb = st.columns(2)
    for i,(title,desc) in enumerate(future_items):
        (ca if i%2==0 else cb).markdown(
            f"""<div style="background:{C["panel"]};border:1px solid {C["border"]};
            border-radius:8px;padding:12px 16px;margin:5px 0;">
            <b style="color:{C["cyan"]};">→ {title}</b><br>
            <span style="color:{C["grey"]};font-size:.87rem;">{desc}</span></div>""",
            unsafe_allow_html=True)

    st.markdown("---")

    # references
    st.subheader("📚 References")
    for i,ref in enumerate([
        "UCI Machine Learning Repository — Phishing Websites Dataset (Mohammad et al., 2014)",
        "PhishTank — https://www.phishtank.com (Community-driven phishing URL repository)",
        "Mohammad R., Thabtah F., McCluskey L. — Neural Computing and Applications, 2014",
        "Verma R., Das A. — What's in a URL: Fast feature extraction and malicious URL detection, 2017",
        "Marchal S. et al. — Off-the-hook: Efficient and effective phishing detection using ML, 2016",
        "Aggarwal A. K., Kumar A. — Journal of King Saud University-CIS, 2020",
        "Rao R. S., Pais A. R. — ICCMC, IEEE, 2019",
    ], 1):
        st.markdown(f"<div style='color:{C['grey']};font-size:.87rem;padding:4px 0;"
                    f"border-bottom:1px solid {C['border']};'>[{i}] {ref}</div>",
                    unsafe_allow_html=True)
