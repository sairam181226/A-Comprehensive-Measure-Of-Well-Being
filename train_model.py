"""
train_model.py
---------------
Trains a classifier to predict a country's HDI tier (Very High / High /
Medium / Low) directly from the four raw development indicators:
    - life_expectancy
    - mean_years_schooling
    - expected_years_schooling
    - gni_per_capita

Compares two models:
    1. Logistic Regression (baseline, linear)
    2. Random Forest (handles the non-linear log/geometric-mean structure
       of the true HDI formula better)

Outputs:
    - console classification report + accuracy for both models
    - figures/confusion_matrix.png
    - figures/feature_importance.png
    - models/hdi_classifier.joblib  (best model, plus the label encoder)
"""
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
)

FEATURES = [
    "life_expectancy", "mean_years_schooling",
    "expected_years_schooling", "gni_per_capita",
]
TARGET = "hdi_tier"
TIER_ORDER = ["Low", "Medium", "High", "Very High"]

data_path = os.path.join(BASE_DIR, "data", "countries_dataset.csv")
df = pd.read_csv(data_path)

X = df[FEATURES].copy()
# log-transform income: this is the single most important preprocessing
# step, since HDI itself uses log(GNI) -- a model given raw GNI has to
# rediscover this non-linearity from scratch.
X["gni_per_capita"] = np.log(X["gni_per_capita"])

le = LabelEncoder()
le.fit(TIER_ORDER)
y = le.transform(df[TARGET])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

models = {
    "Logistic Regression": Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000)),
    ]),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, max_depth=6, random_state=42
    ),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    results[name] = (model, acc, preds)
    print(f"\n=== {name} ===")
    print(f"Accuracy: {acc:.3f}")
    print(classification_report(y_test, preds, target_names=le.classes_, zero_division=0))

# Pick the better model
best_name = max(results, key=lambda k: results[k][1])
best_model, best_acc, best_preds = results[best_name]
print(f"\nBest model: {best_name} (accuracy={best_acc:.3f})")

# --- Confusion matrix figure ---
cm = confusion_matrix(y_test, best_preds)
fig, ax = plt.subplots(figsize=(5.5, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title(f"Confusion Matrix — {best_name}\nTest Accuracy: {best_acc:.1%}")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "confusion_matrix.png"), dpi=150)
plt.close()

# --- Feature importance (Random Forest only) ---
rf_model, rf_acc, _ = results["Random Forest"]
importances = rf_model.feature_importances_
order = np.argsort(importances)
fig, ax = plt.subplots(figsize=(6, 4))
labels = ["Life expectancy", "Mean years\nschooling", "Expected years\nschooling", "GNI per capita\n(log)"]
ax.barh(np.array(labels)[order], importances[order], color="#3f7f6b")
ax.set_xlabel("Importance")
ax.set_title("Random Forest — Feature Importance")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "feature_importance.png"), dpi=150)
plt.close()

# --- Save best model + encoder + the log-transform convention ---
joblib.dump(
    {"model": best_model, "label_encoder": le, "features": FEATURES},
    os.path.join(MODELS_DIR, "hdi_classifier.joblib")
)