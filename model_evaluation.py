"""
model_evaluation.py — Standalone Testing & Accuracy Report
PhishGuard AI | Final Year Project | BBDU Lucknow

Run:  python model_evaluation.py
Out:  eval_results.json  (auto-read by app.py)
      phishing_model_rf.pkl
      scaler_rf.pkl
"""

import json, pickle, warnings, os
import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix,
                              classification_report, roc_curve)

warnings.filterwarnings("ignore")

# ── Permanent path fix ──────────────────────────────────────────
# __file__ is always this script's own location, regardless of
# which directory you run the terminal from.
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
ARFF_PATH  = os.path.join(BASE_DIR, "Training_Dataset.arff")
MODEL_OUT  = os.path.join(BASE_DIR, "phishing_model_rf.pkl")
SCALER_OUT = os.path.join(BASE_DIR, "scaler_rf.pkl")
EVAL_OUT   = os.path.join(BASE_DIR, "eval_results.json")
# ────────────────────────────────────────────────────────────────

print("=" * 62)
print("  PhishGuard AI — Model Evaluation Report")
print("=" * 62)

# 1. Load dataset
print("\n[1/5] Loading UCI Phishing Dataset...")
print(f"      Looking for: {ARFF_PATH}")

if not os.path.exists(ARFF_PATH):
    print(f"\n  ERROR: Training_Dataset.arff not found at:\n  {ARFF_PATH}")
    print("  Make sure Training_Dataset.arff is in the same folder as this script.")
    raise SystemExit(1)

data = arff.loadarff(ARFF_PATH)
df   = pd.DataFrame(data[0])
df   = df.map(lambda x: int(x.decode()) if isinstance(x, bytes) else int(x))

X = df.drop("Result", axis=1)
y = df["Result"].map({-1: 1, 1: 0})  # phishing=1, legit=0

print(f"      Total samples  : {len(df)}")
print(f"      Features       : {X.shape[1]}")
print(f"      Legitimate URLs: {(y==0).sum()} ({(y==0).mean()*100:.1f}%)")
print(f"      Phishing URLs  : {(y==1).sum()} ({(y==1).mean()*100:.1f}%)")

# 2. Preprocess
print("\n[2/5] Preprocessing — StandardScaler + 80-20 split...")
sc = StandardScaler()
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_tr_s = sc.fit_transform(X_tr)
X_te_s  = sc.transform(X_te)
print(f"      Train: {len(X_tr)} | Test: {len(X_te)}")

# 3. Train & compare models
print("\n[3/5] Training & Evaluating All Algorithms...")
print(f"  {'Algorithm':28} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'AUC':>7}")
print("  " + "-" * 60)

models_cfg = {
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, random_state=42),
    "Decision Tree":       DecisionTreeClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
}

comparison = {}
best_acc   = 0

for name, m in models_cfg.items():
    m.fit(X_tr_s, y_tr)
    yp   = m.predict(X_te_s)
    yprob = m.predict_proba(X_te_s)[:, 1]
    acc  = accuracy_score(y_te, yp)
    prec = precision_score(y_te, yp)
    rec  = recall_score(y_te, yp)
    f1   = f1_score(y_te, yp)
    auc  = roc_auc_score(y_te, yprob)
    comparison[name] = {"accuracy": round(acc*100,2), "precision": round(prec*100,2),
                        "recall": round(rec*100,2), "f1": round(f1*100,2), "auc": round(auc*100,2)}
    star = " ★ BEST" if acc > best_acc else ""
    print(f"  {name:28} {acc*100:6.2f}% {prec*100:6.2f}% {rec*100:6.2f}% {f1*100:6.2f}% {auc*100:6.2f}%{star}")
    if acc > best_acc:
        best_acc = acc

# 4. Detailed RF evaluation
print("\n[4/5] Detailed Random Forest Evaluation...")
rf = models_cfg["Random Forest"]
yp_rf    = rf.predict(X_te_s)
yprob_rf = rf.predict_proba(X_te_s)[:, 1]
cm       = confusion_matrix(y_te, yp_rf)

print("\n  Classification Report:")
rpt = classification_report(y_te, yp_rf, target_names=["Legitimate", "Phishing"])
for line in rpt.split("\n"):
    print("    " + line)

print(f"\n  Confusion Matrix:")
print(f"                  Pred Legit  Pred Phishing")
print(f"  Actual Legit  :  {cm[0,0]:6d}      {cm[0,1]:6d}")
print(f"  Actual Phishing: {cm[1,0]:6d}      {cm[1,1]:6d}")

# 5-fold CV
print("\n  5-Fold Cross-Validation...")
Xs_full = sc.transform(X)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_acc = cross_val_score(rf, Xs_full, y, cv=cv, scoring="accuracy")
cv_f1  = cross_val_score(rf, Xs_full, y, cv=cv, scoring="f1")
cv_roc = cross_val_score(rf, Xs_full, y, cv=cv, scoring="roc_auc")
print(f"  Fold accuracies: {[f'{a*100:.2f}%' for a in cv_acc]}")
print(f"  Mean Accuracy  : {cv_acc.mean()*100:.2f}% ± {cv_acc.std()*100:.2f}%")
print(f"  Mean F1-Score  : {cv_f1.mean()*100:.2f}%")
print(f"  Mean ROC-AUC   : {cv_roc.mean()*100:.2f}%")

# Feature importances
fi = rf.feature_importances_
top12 = sorted(zip(list(X.columns), fi.tolist()), key=lambda x: -x[1])[:12]
print("\n  Top 12 Feature Importances:")
for feat, score in top12:
    bar = "█" * int(score * 200)
    print(f"    {feat:35}: {score*100:5.2f}%  {bar}")

# ROC points
fpr, tpr, _ = roc_curve(y_te, yprob_rf)
idx = np.linspace(0, len(fpr)-1, 25, dtype=int)

# 5. Save
print("\n[5/5] Saving model, scaler, and evaluation results...")
pickle.dump(rf, open(MODEL_OUT, "wb"))
pickle.dump(sc, open(SCALER_OUT, "wb"))

eval_data = {
    "model_type": "RandomForestClassifier",
    "dataset": {"total": len(df), "phishing": int((y==1).sum()), "legitimate": int((y==0).sum())},
    "n_features": int(rf.n_features_in_),
    "rf_test_metrics": {
        "accuracy":  round(accuracy_score(y_te, yp_rf)*100, 2),
        "precision": round(precision_score(y_te, yp_rf)*100, 2),
        "recall":    round(recall_score(y_te, yp_rf)*100, 2),
        "f1_score":  round(f1_score(y_te, yp_rf)*100, 2),
        "roc_auc":   round(roc_auc_score(y_te, yprob_rf)*100, 2),
    },
    "cv_metrics": {
        "acc_mean": round(cv_acc.mean()*100, 2), "acc_std": round(cv_acc.std()*100, 2),
        "f1_mean":  round(cv_f1.mean()*100,  2), "roc_mean": round(cv_roc.mean()*100, 2),
    },
    "confusion_matrix":  cm.tolist(),
    "top_features":      [[k, round(v*100, 2)] for k, v in top12],
    "roc_curve":         {"fpr": fpr[idx].tolist(), "tpr": tpr[idx].tolist()},
    "model_comparison":  comparison,
    "cv_fold_accs":      cv_acc.round(4).tolist(),
    "best_model":        "Random Forest",
}
json.dump(eval_data, open(EVAL_OUT, "w"), indent=2)

print(f"\n  Saved: {MODEL_OUT}")
print(f"  Saved: {SCALER_OUT}")
print(f"  Saved: {EVAL_OUT}")
print("\n" + "=" * 62)
print("  FINAL ACCURACY SUMMARY")
print("=" * 62)
m = eval_data["rf_test_metrics"]
print(f"  Accuracy  : {m['accuracy']}%")
print(f"  Precision : {m['precision']}%")
print(f"  Recall    : {m['recall']}%")
print(f"  F1-Score  : {m['f1_score']}%")
print(f"  ROC-AUC   : {m['roc_auc']}%")
print(f"  CV Acc    : {eval_data['cv_metrics']['acc_mean']}% ± {eval_data['cv_metrics']['acc_std']}%")
print("=" * 62)
print("\n  Next: streamlit run app.py")
