"""
Train an improved fake-news classifier.

Improvements over the notebook baseline (TF-IDF word 1-2 grams + LogisticRegression):
  - Word TF-IDF (1-2 grams) UNION char TF-IDF (2-5 grams): char n-grams catch
    ALL-CAPS shouting, elongated words ("coooool"), punctuation spam ("!!!??"),
    and other surface signals that word TF-IDF misses.
  - Sublinear TF and min/max document frequency pruning in both vectorizers.
  - LinearSVC (typically beats LogisticRegression on sparse text) wrapped in
    CalibratedClassifierCV so predict_proba() returns honest probabilities.
  - Training set augmented with article titles as extra short examples. The
    REAL class in this dataset is long Reuters wire copy, so a model trained
    on articles only reads every short headline as FAKE. Adding titles teaches
    it what short real text looks like (headline behavior: plausible real
    headline goes 0.25 -> 0.79 p_real). Costs ~0.7pt article accuracy.
  - A model comparison table so the improvement is measured, not vibes.

Artifacts saved to model/:
  model.pkl    - featurizer + best classifier (loaded by app.py)
  metrics.json - metrics for every model compared

Run:  py train_model.py
"""

import json
import os
import time

import joblib
import pandas as pd
from featurizer import PairFeaturizer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

RANDOM_STATE = 42
DATA_URL = (
    "https://raw.githubusercontent.com/lutzhamel/fake-news/refs/heads/master/"
    "data/fake_or_real_news.csv"
)
MODEL_DIR = "model"


def load_data():
    print("Loading dataset...")
    df = pd.read_csv(DATA_URL)
    df = df.dropna(subset=["title", "text", "label"])
    df["combined_text"] = df["title"].fillna("") + " " + df["text"].fillna("")
    X = df["combined_text"]
    y = df["label"]
    print(f"  {len(df)} articles | class balance:")
    for label, count in y.value_counts().items():
        print(f"  {label}: {count}")
    return df, X, y


def build_models():
    return {
        "logreg_baseline": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "logreg_tuned": LogisticRegression(
            max_iter=2000, C=4.0, random_state=RANDOM_STATE
        ),
        "linear_svc_calibrated": CalibratedClassifierCV(
            LinearSVC(C=1.0, random_state=RANDOM_STATE), cv=5
        ),
    }


def metrics_dict(y_test, y_pred):
    return {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, pos_label="REAL")), 4),
        "recall": round(float(recall_score(y_test, y_pred, pos_label="REAL")), 4),
        "f1": round(float(f1_score(y_test, y_pred, pos_label="REAL")), 4),
    }


SMOKE_TEST_HEADLINES = [
    "US unemployment rate falls to 3.9 percent, lowest since 1969",
    "Scientists confirm water ice deposits found in moon shadowed craters",
    "SHOCKING: Doctors HATE this one weird trick that CURES everything!!!",
    "BREAKING: Hillary Clinton caught running secret child trafficking ring in pizza shop basement",
]


def main():
    df, X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    # Augment: article titles of the training split as extra short examples.
    # Titles inherit their article's label and teach the model short-text patterns.
    train_titles = df.loc[X_train.index, "title"].fillna("")
    X_train_aug = list(X_train) + list(train_titles)
    y_train_aug = list(y_train) + list(y_train)
    print(f"  + {len(train_titles)} title examples -> {len(X_train_aug)} training rows")

    print("Fitting word+char TF-IDF union...")
    featurizer = PairFeaturizer()
    X_train_feats = featurizer.fit_transform(X_train_aug)
    X_test_feats = featurizer.transform(X_test)
    print(f"  feature matrix: {X_train_feats.shape[0]} x {X_train_feats.shape[1]}")

    results = {}
    best_name, best_model, best_acc = None, None, -1.0
    for name, clf in build_models().items():
        t0 = time.time()
        clf.fit(X_train_feats, y_train_aug)
        elapsed = time.time() - t0
        y_pred = clf.predict(X_test_feats)
        m = metrics_dict(y_test, y_pred)
        m["fit_seconds"] = round(elapsed, 1)
        results[name] = m
        print(f"  {name}: acc={m['accuracy']:.4f} f1={m['f1']:.4f} ({elapsed:.0f}s)")
        if m["accuracy"] > best_acc:
            best_name, best_model, best_acc = name, clf, m["accuracy"]

    print(f"\nBest model: {best_name} (accuracy {best_acc:.4f})")
    print(classification_report(y_test, best_model.predict(X_test_feats)))

    print("Saving artifacts...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(
        {"featurizer": featurizer, "classifier": best_model, "model_name": best_name},
        os.path.join(MODEL_DIR, "model.pkl"),
    )
    metrics = {
        "best_model": best_name,
        "n_features": int(X_train_feats.shape[1]),
        "n_train": int(len(X_train)),
        "n_train_augmented": int(len(X_train_aug)),
        "n_test": int(len(X_test)),
        "augmented_with_titles": True,
        "models": results,
    }
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\nSmoke test on sample headlines:")
    proba = best_model.predict_proba(featurizer.transform(SMOKE_TEST_HEADLINES))
    for text, row in zip(SMOKE_TEST_HEADLINES, proba):
        classes = list(best_model.classes_)
        p_real = row[classes.index("REAL")]
        label = "REAL" if p_real >= 0.5 else "FAKE"
        print(f"  [{label:4s}] {p_real:.2f} REAL | {text[:70]}")


if __name__ == "__main__":
    main()
