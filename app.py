"""
Fake News Detector - web app.

Serves a simple UI where you paste a headline (or a whole article) and get a
REAL / FAKE verdict with a confidence score, powered by the model trained by
train_model.py (word + char TF-IDF -> calibrated LinearSVC).

Run:  py app.py          (then open http://127.0.0.1:5000)
"""

import json
import os

import joblib
from flask import Flask, jsonify, render_template, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "model", "metrics.json")

MAX_INPUT_CHARS = 20_000  # guard against absurd payloads; articles fit easily

app = Flask(__name__)

_bundle = None


def get_bundle():
    """Lazily load the trained model bundle (featurizer + classifier)."""
    global _bundle
    if _bundle is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                "model/model.pkl not found. Run `py train_model.py` first."
            )
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


def predict_real_probability(text: str) -> float:
    b = get_bundle()
    clf = b["classifier"]
    features = b["featurizer"].transform([text])
    proba = clf.predict_proba(features)[0]
    classes = list(clf.classes_)  # ['FAKE', 'REAL']
    return float(proba[classes.index("REAL")])


def load_metrics():
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


MODEL_DISPLAY_NAMES = {
    "linear_svc_calibrated": "Calibrated Linear SVC",
    "logreg_tuned": "Tuned Logistic Regression",
    "logreg_baseline": "Logistic Regression",
}


@app.route("/")
def index():
    metrics = load_metrics()
    best = metrics.get("models", {}).get(metrics.get("best_model", ""), {})
    accuracy = best.get("accuracy")
    return render_template(
        "index.html",
        model_name=MODEL_DISPLAY_NAMES.get(
            metrics.get("best_model", ""), metrics.get("best_model", "unknown")
        ),
        accuracy=f"{accuracy * 100:.1f}%" if accuracy is not None else None,
        n_train=metrics.get("n_train"),
    )


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()

    if not text:
        return jsonify({"error": "Please enter a headline or some article text."}), 400
    if len(text) > MAX_INPUT_CHARS:
        return jsonify({"error": "Text is too long (20,000 character limit)."}), 400

    p_real = predict_real_probability(text)
    label = "REAL" if p_real >= 0.5 else "FAKE"
    confidence = p_real if label == "REAL" else 1.0 - p_real

    return jsonify(
        {
            "label": label,
            "confidence": round(confidence * 100, 1),
            "probability_real": round(p_real, 4),
            "probability_fake": round(1.0 - p_real, 4),
            "model": MODEL_DISPLAY_NAMES.get(
                load_metrics().get("best_model", ""), "unknown"
            ),
        }
    )


@app.route("/health")
def health():
    try:
        get_bundle()
        return jsonify({"status": "ok", "model_loaded": True})
    except Exception as exc:  # noqa: BLE001 - report any load failure to the caller
        return jsonify({"status": "error", "detail": str(exc)}), 500


if __name__ == "__main__":
    # Load the model once at startup so the first request is fast.
    get_bundle()
    app.run(host="127.0.0.1", port=5000, debug=False)
