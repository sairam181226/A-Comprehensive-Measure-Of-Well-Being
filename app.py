import os
import joblib
import numpy as np
from flask import Flask, jsonify, render_template, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "hdi_classifier.joblib")

app = Flask(__name__, static_folder=".", template_folder=".")

with open(MODEL_PATH, "rb") as f:
    model_bundle = joblib.load(f)

model = model_bundle["model"]
label_encoder = model_bundle["label_encoder"]
FEATURES = model_bundle["features"]


def compute_hdi(life_expectancy, mean_years_schooling, expected_years_schooling, gni_per_capita):
    life_expectancy_index = np.clip((life_expectancy - 20) / (85 - 20), 0, 1)
    mys_index = np.clip(mean_years_schooling / 15, 0, 1)
    eys_index = np.clip(expected_years_schooling / 18, 0, 1)
    education_index = (mys_index + eys_index) / 2
    income_index = np.clip((np.log(gni_per_capita) - np.log(100)) / (np.log(75000) - np.log(100)), 0, 1)
    hdi = float(np.cbrt(life_expectancy_index * education_index * income_index))
    return {
        "life_expectancy_index": round(float(life_expectancy_index), 4),
        "education_index": round(float(education_index), 4),
        "income_index": round(float(income_index), 4),
        "hdi": round(hdi, 4),
    }


def tier_for(hdi):
    if hdi >= 0.800:
        return "Very High"
    if hdi >= 0.700:
        return "High"
    if hdi >= 0.550:
        return "Medium"
    return "Low"


@app.route("/")
def index():
    return app.send_static_file("hdi_explorer.html")


@app.route("/api/predict", methods=["POST"])
def predict():
    payload = request.get_json(force=True)
    try:
        values = [float(payload[k]) for k in FEATURES]
    except (KeyError, ValueError, TypeError):
        return jsonify({"error": "Request must include numeric values for all features: life_expectancy, mean_years_schooling, expected_years_schooling, gni_per_capita."}), 400

    X = np.array([np.log(values[3]) if i == 3 else values[i] for i in range(len(values))]).reshape(1, -1)
    predicted_class = model.predict(X)[0]
    tier = label_encoder.inverse_transform([predicted_class])[0]
    hdi_data = compute_hdi(*values)

    return jsonify({
        "tier": tier,
        "hdi": hdi_data["hdi"],
        "sub_indices": {
            "life_expectancy_index": hdi_data["life_expectancy_index"],
            "education_index": hdi_data["education_index"],
            "income_index": hdi_data["income_index"],
        },
        "model": "HDI tier classifier",
        "accuracy": None,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
