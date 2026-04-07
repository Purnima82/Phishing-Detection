<div align="center">

# 🛡️ PhishGuard AI
### Real-Time Phishing Website Detection System

[![Python](https://img.shields.io/badge/Python-3.9+-3776ab?style=flat&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-ff4b4b?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-f7931e?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Accuracy](https://img.shields.io/badge/Accuracy-97.51%25-00e676?style=flat)](#model-performance)
[![ROC--AUC](https://img.shields.io/badge/ROC--AUC-99.64%25-00c8f0?style=flat)](#model-performance)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat)](LICENSE)

**A machine learning-powered web application that detects phishing websites in real-time using 30 URL and domain features, achieving 97.51% accuracy with a Random Forest classifier trained on 11,055 samples.**

[🔍 Try the App](#getting-started) · [📊 Model Performance](#model-performance) · [🚀 Features](#features) · [📁 Project Structure](#project-structure)

---

</div>

## 📌 Overview

Phishing attacks are one of the most common cyber threats, with over 700,000 phishing sites reported in 2025 alone. Traditional blacklist-based systems fail against zero-day attacks. **PhishGuard AI** uses machine learning to detect phishing websites based on structural and behavioral URL features — no blacklists needed.

This project was developed as a Final Year B.Tech project at **Babu Banarasi Das University, Lucknow** by the CSE Department.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔍 **Real-Time URL Scanner** | Paste any URL and get instant Phishing / Suspicious / Legitimate verdict |
| 📊 **Analytics Dashboard** | Live charts of scan history, risk trends, and result distributions |
| 🎯 **Model Performance Page** | ROC curve, confusion matrix, algorithm comparison, feature importances |
| 📜 **Scan History** | SQLite-backed persistent history with filter and sort |
| 🧠 **30-Feature Analysis** | Full breakdown of every signal used in the prediction |
| 🛡️ **Chrome Extension** | Real-time protection in your browser with badge alerts and warning banners |

---

## 🧠 Model Performance

The system was trained and evaluated on the **UCI Phishing Websites Dataset** (11,055 samples, 30 features).

| Algorithm | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| ✅ **Random Forest** | **97.51%** | **97.93%** | **96.43%** | **97.17%** | **99.64%** |
| Decision Tree | 96.97% | 97.60% | 95.51% | 96.54% | 97.64% |
| Gradient Boosting | 95.02% | 94.94% | 93.78% | 94.35% | 99.13% |
| Logistic Regression | 92.81% | 93.44% | 90.10% | 91.74% | 97.85% |

**5-Fold Cross-Validation:** 96.98% ± 0.29%

### Top Features (by importance)
1. `SSLfinal_State` — 32.19%
2. `URL_of_Anchor` — 25.53%
3. `web_traffic` — 7.15%
4. `having_Sub_Domain` — 6.53%
5. `Links_in_tags` — 4.31%

---

## 🏗️ System Architecture

```
User Input (URL)
      │
      ▼
Feature Extraction (30 signals)
  ├── URL-based:    length, @ symbol, hyphens, IP address, shorteners
  ├── Domain-based: age, WHOIS, subdomain depth, registration length
  ├── SSL/TLS:      HTTPS presence, certificate validity
  ├── Page-based:   favicon, iframes, popups, right-click disabled
  └── Statistical:  traffic rank, Google index, DNS record, page rank
      │
      ▼
StandardScaler (normalization)
      │
      ▼
Random Forest Classifier (100 trees)
      │
      ▼
Rule-Based Risk Boost (+0–40%)
  ├── No HTTPS → +10%
  ├── Phishing keywords → +4% each (max +20%)
  └── Suspicious domain patterns → +10%
      │
      ▼
Final Risk Score → Verdict
  ├── ≥ 70% → 🚨 PHISHING
  ├── 30–70% → ⚠️ SUSPICIOUS
  └── ≤ 30% → ✅ LEGITIMATE
```

---

## 📁 Project Structure

```
PhishGuard-AI/
│
├── app.py                    # Main Streamlit web application (5 pages)
├── feature_engineering.py    # 30-feature URL extraction + rule-based scoring
├── database.py               # SQLite scan history persistence
├── model_evaluation.py       # Model training, evaluation & .pkl generation
│
├── Training_Dataset.arff     # UCI Phishing Websites Dataset (11,055 samples)
├── phishing_model_rf.pkl     # Trained Random Forest model
├── scaler_rf.pkl             # Fitted StandardScaler
├── eval_results.json         # Evaluation metrics (auto-read by app.py)
│
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9 or higher
- pip

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/PhishGuard-AI.git
cd PhishGuard-AI
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train / regenerate the model (optional)

If you get a pickle version mismatch error, regenerate the model from scratch:

```bash
python model_evaluation.py
```

This will produce `phishing_model_rf.pkl`, `scaler_rf.pkl`, and `eval_results.json`.

### 5. Run the app

```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501**

---

## 🔬 Feature Categories

The system extracts **30 features** across 6 categories:

| Category | Features |
|---|---|
| 🔗 URL-Based | URL length, @ symbol, double slash redirect, hyphens, IP address, shortening service |
| 🌐 Domain-Based | Domain age, WHOIS record, registration length, subdomain depth, prefix-suffix |
| 🔒 SSL / TLS | HTTPS presence, certificate validity, HTTPS token in domain name |
| 📄 Page-Based | Favicon mismatch, iframes, pop-ups, right-click disabled, form submission to email |
| 📊 Statistical | Web traffic rank, Google index, links to page, DNS record, page rank |
| ⚙️ Behavioral | Mouse-over changes, redirect count, on-submit actions, abnormal URL patterns |

---

## 🛡️ Chrome Extension

A companion browser extension is included that:
- Auto-scans every URL you visit in real-time
- Shows a **colored badge**: ✓ green (safe) · ? yellow (suspicious) · ! red (phishing)
- Injects a **warning banner** on phishing/suspicious pages
- Sends **desktop notifications** when phishing is detected
- Maintains scan history and stats in the popup

**To install:**
1. Open `chrome://extensions`
2. Enable **Developer Mode** (top-right)
3. Click **Load unpacked** → select the `chrome-extension/` folder

---

## 📦 Dependencies

```
streamlit>=1.32.0
scikit-learn>=1.4.0
pandas>=2.0.0
numpy>=1.26.0
scipy>=1.11.0
matplotlib>=3.7.0
requests>=2.31.0
python-whois>=0.9.4
```

---

## 👥 Team

| Name | Enrollment No. |
|---|---|
| Aditya Kumar Singh | 1220432045 |
| Srishty Rai | 1220432545 |
| Poornima Singh | 1220432381 |

**Supervisor:** Mr. Ahmad Raza, Dept. of CSE, BBDU Lucknow

---

## 📚 References

1. UCI Machine Learning Repository — [Phishing Websites Dataset](https://archive.ics.uci.edu/dataset/327/phishing+websites) (Mohammad et al., 2014)
2. PhishTank — https://www.phishtank.com
3. Mohammad R., Thabtah F., McCluskey L. — *Neural Computing and Applications*, 2014
4. Verma R., Das A. — *What's in a URL: Fast feature extraction and malicious URL detection*, 2017

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">
Made with ❤️ at BBDU Lucknow · CSE Department · 2024–25
</div>
