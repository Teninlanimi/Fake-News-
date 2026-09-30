"""
Stress test: evaluate the fake-news model (trained on long news articles) on
the LIAR dataset - short PolitiFact statements it has never seen.

Checks whether the model learned "fake vs real news" or just "what Reuters
wire copy looks like", and breaks the failures down by original LIAR label,
statement length, and topic.

Run:  py stress_test.py   (writes stress_results.json + prints analysis)
"""

import csv
import json
import os
import urllib.request
from collections import Counter, defaultdict

import joblib

# LIAR six-way labels -> binary
LABEL_MAP = {
    "true": "REAL",
    "mostly-true": "REAL",
    "half-true": None,  # genuinely ambiguous; excluded from scoring
    "barely-true": "FAKE",
    "false": "FAKE",
    "pants-fire": "FAKE",
}

TEST_FILE = "liar_test.tsv"
LIAR_URL = "https://raw.githubusercontent.com/tfs4/liar_dataset/master/test.tsv"
RESULTS_FILE = "stress_results.json"


def ensure_test_file():
    if not os.path.exists(TEST_FILE):
        print(f"Downloading LIAR test set -> {TEST_FILE}")
        urllib.request.urlretrieve(LIAR_URL, TEST_FILE)


def load_liar():
    rows = []
    with open(TEST_FILE, encoding="utf-8") as f:
        for line in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            if len(line) < 3:
                continue
            binary = LABEL_MAP.get(line[1].strip())
            if binary is None:
                continue
            rows.append(
                {
                    "label": binary,
                    "liar_label": line[1].strip(),
                    "statement": line[2],
                    "subject": line[3] if len(line) > 3 else "",
                }
            )
    return rows


def main():
    ensure_test_file()
    print("Loading model...")
    bundle = joblib.load("model/model.pkl")
    clf, fz = bundle["classifier"], bundle["featurizer"]
    classes = list(clf.classes_)

    rows = load_liar()
    print(f"Loaded {len(rows)} binary-labeled LIAR statements")
    print(f"  balance: {dict(Counter(r['label'] for r in rows))}")

    texts = [r["statement"] for r in rows]
    proba = clf.predict_proba(fz.transform(texts))
    p_reals = [row[classes.index("REAL")] for row in proba]

    for r, p_real in zip(rows, p_reals):
        r["p_real"] = round(p_real, 4)
        r["pred"] = "REAL" if p_real >= 0.5 else "FAKE"
        r["correct"] = r["pred"] == r["label"]

    n = len(rows)
    acc = sum(r["correct"] for r in rows) / n
    real_rows = [r for r in rows if r["label"] == "REAL"]
    fake_rows = [r for r in rows if r["label"] == "FAKE"]
    acc_real = sum(r["correct"] for r in real_rows) / len(real_rows)
    acc_fake = sum(r["correct"] for r in fake_rows) / len(fake_rows)

    print("\n=== OVERALL ===")
    print(f"  accuracy: {acc:.1%}  ({acc_real:.1%} on REAL, {acc_fake:.1%} on FAKE)")

    print("\n=== BY ORIGINAL LIAR LABEL ===")
    by_liar = defaultdict(list)
    for r in rows:
        by_liar[r["liar_label"]].append(r)
    liar_table = {}
    for lbl in ["true", "mostly-true", "barely-true", "false", "pants-fire"]:
        grp = by_liar[lbl]
        acc_g = sum(r["correct"] for r in grp) / len(grp)
        avg_p = sum(r["p_real"] for r in grp) / len(grp)
        liar_table[lbl] = {
            "n": len(grp),
            "accuracy": round(acc_g, 3),
            "avg_p_real": round(avg_p, 3),
        }
        print(f"  {lbl:12s} n={len(grp):4d}  acc={acc_g:6.1%}  avg p_real={avg_p:.3f}")

    print("\n=== BY STATEMENT LENGTH ===")
    buckets = [("short (<15 words)", lambda w: w < 15), ("medium (15-30)", lambda w: 15 <= w < 30), ("long (30+)", lambda w: w >= 30)]
    len_table = {}
    for name, pred_fn in buckets:
        grp = [r for r in rows if pred_fn(len(r["statement"].split()))]
        if not grp:
            continue
        acc_g = sum(r["correct"] for r in grp) / len(grp)
        avg_p = sum(r["p_real"] for r in grp) / len(grp)
        len_table[name] = {"n": len(grp), "accuracy": round(acc_g, 3), "avg_p_real": round(avg_p, 3)}
        print(f"  {name:18s} n={len(grp):4d}  acc={acc_g:6.1%}  avg p_real={avg_p:.3f}")

    print("\n=== BY TOPIC (top subjects) ===")
    subj_groups = defaultdict(list)
    for r in rows:
        for s in r["subject"].split(","):
            s = s.strip()
            if s:
                subj_groups[s].append(r)
    top = sorted(subj_groups.items(), key=lambda kv: -len(kv[1]))[:8]
    topic_table = {}
    for subj, grp in top:
        acc_g = sum(r["correct"] for r in grp) / len(grp)
        avg_p = sum(r["p_real"] for r in grp) / len(grp)
        topic_table[subj] = {"n": len(grp), "accuracy": round(acc_g, 3), "avg_p_real": round(avg_p, 3)}
        print(f"  {subj:18s} n={len(grp):4d}  acc={acc_g:6.1%}  avg p_real={avg_p:.3f}")

    # Failure exemplars
    print("\n=== WORST CONFIDENT MISTAKES (real statements called FAKE) ===")
    bad_real = sorted((r for r in real_rows if not r["correct"]), key=lambda r: r["p_real"])
    for r in bad_real[:5]:
        print(f"  p_real={r['p_real']:.2f} [{r['liar_label']:11s}] {r['statement'][:80]}")

    print("\n=== WORST CONFIDENT MISTAKES (false statements called REAL) ===")
    bad_fake = sorted((r for r in fake_rows if not r["correct"]), key=lambda r: -r["p_real"])
    for r in bad_fake[:5]:
        print(f"  p_real={r['p_real']:.2f} [{r['liar_label']:11s}] {r['statement'][:80]}")

    results = {
        "dataset": "LIAR test.tsv (PolitiFact short statements, unseen)",
        "n_scored": n,
        "n_excluded_half_true": 1283 - n,
        "accuracy": round(acc, 4),
        "accuracy_real": round(acc_real, 4),
        "accuracy_fake": round(acc_fake, 4),
        "by_liar_label": liar_table,
        "by_length": len_table,
        "by_topic": topic_table,
        "confident_wrong_real": bad_real[:10],
        "confident_wrong_fake": bad_fake[:10],
    }
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {RESULTS_FILE}")


if __name__ == "__main__":
    main()
