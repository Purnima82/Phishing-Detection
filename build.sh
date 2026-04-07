#!/bin/bash
set -e
echo "=== PhishGuard AI — Build Step ==="

pip install -r requirements_full.txt

# Only retrain if the model doesn't already exist (saves build time on redeploys)
if [ ! -f "phishing_model_rf.pkl" ]; then
    echo "Training ML model..."
    python model_evaluation.py
else
    echo "Model already exists, skipping training."
fi

echo "=== Build complete ==="
