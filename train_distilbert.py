"""
Fine-tune DistilBERT on the same data as train_model.py and compare it
against the TF-IDF model on:
  1. the in-domain held-out test set (articles, same split), and
  2. the LIAR stress set (short PolitiFact statements, unseen domain).

The question: does a transformer with real language understanding generalize
better, or does it learn the same "wire copy vs tabloid" stylistic shortcut?

Run:  py train_distilbert.py   (CPU-friendly: ~20-40 min on a modern laptop)
"""

import faulthandler
import json
import os
import time

faulthandler.enable()  # print a native traceback on hard crashes (segfaults)

import joblib
import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from stress_test import ensure_test_file, load_liar
from train_model import DATA_URL, RANDOM_STATE, SMOKE_TEST_HEADLINES, load_data

MODEL_NAME = "distilbert-base-uncased"
OUT_DIR = "model/distilbert"
METRICS_PATH = "model/distilbert_metrics.json"
MAX_LENGTH = 256  # title + lede carries the signal; keeps CPU training sane
BATCH_SIZE = 32
EPOCHS = 2
SEED = RANDOM_STATE

LABEL2ID = {"FAKE": 0, "REAL": 1}
ID2LABEL = {0: "FAKE", 1: "REAL"}


def to_dataset(texts, labels, tokenizer):
    enc = tokenizer(
        list(texts), truncation=True, max_length=MAX_LENGTH, padding=False
    )
    ds = {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "labels": [LABEL2ID[l] for l in labels],
    }
    return ds


class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, enc):
        self.enc = enc

    def __len__(self):
        return len(self.enc["labels"])

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = int(item["labels"])
        return item


class Collator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        batch = {
            "input_ids": [f["input_ids"] for f in features],
            "attention_mask": [f["attention_mask"] for f in features],
            "labels": [f["labels"] for f in features],
        }
        return self.tokenizer.pad(batch, padding=True, return_tensors="pt")


def predict_proba(texts, tokenizer, model):
    """Return p(REAL) for each text."""
    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(texts), 32):
            batch = texts[i : i + 32]
            enc = tokenizer(
                list(batch),
                truncation=True,
                max_length=MAX_LENGTH,
                padding=True,
                return_tensors="pt",
            )
            logits = model(**enc).logits
            p = torch.softmax(logits, dim=-1)[:, LABEL2ID["REAL"]]
            probs.extend(p.tolist())
    return np.array(probs)


def acc(y_true, y_pred):
    return float(np.mean(np.array(y_true) == np.array(y_pred)))


def f1_real(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    tp = np.sum((y_pred == "REAL") & (y_true == "REAL"))
    fp = np.sum((y_pred == "REAL") & (y_true == "FAKE"))
    fn = np.sum((y_pred == "FAKE") & (y_true == "REAL"))
    return float(tp / (tp + 0.5 * (fp + fn)))


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("=== Loading data (identical split + augmentation as the TF-IDF model) ===")
    df, X, y = load_data()
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    train_titles = df.loc[X_train.index, "title"].fillna("")
    X_train_aug = list(X_train) + list(train_titles)
    y_train_aug = list(y_train) + list(y_train)
    print(f"  train: {len(X_train_aug)} (articles + titles) | test: {len(X_test)}")

    print("\n=== Tokenizer & model ===")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2, id2label=ID2LABEL, label2id=LABEL2ID
    )

    train_ds = SimpleDataset(to_dataset(X_train_aug, y_train_aug, tokenizer))

    args = TrainingArguments(
        output_dir=OUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_steps=50,
        save_strategy="no",
        report_to=[],
        seed=SEED,
        disable_tqdm=True,
        dataloader_drop_last=False,
        dataloader_pin_memory=False,  # pin_memory hard-crashes on CPU-only Windows
        use_cpu=True,
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, data_collator=Collator(tokenizer))

    steps = int(np.ceil(len(train_ds) / BATCH_SIZE)) * EPOCHS
    print(f"\n=== Fine-tuning: {steps} steps (batch={BATCH_SIZE}, len<={MAX_LENGTH}) ===", flush=True)
    trainer.train()

    print("\n=== In-domain evaluation (held-out article test set) ===")
    p_real = predict_proba(list(X_test), tokenizer, model)
    pred_test = np.where(p_real >= 0.5, "REAL", "FAKE")
    in_acc = acc(list(y_test), pred_test)
    in_f1 = f1_real(list(y_test), pred_test)
    print(f"  DistilBERT: accuracy={in_acc:.4f} f1={in_f1:.4f}")

    print("\n=== Out-of-domain evaluation (LIAR) ===")
    ensure_test_file()
    rows = load_liar()
    p_liar = predict_proba([r["statement"] for r in rows], tokenizer, model)
    for r, p in zip(rows, p_liar):
        r["p_real"] = float(p)
        r["pred"] = "REAL" if p >= 0.5 else "FAKE"
        r["correct"] = r["pred"] == r["label"]
    liar_acc = float(np.mean([r["correct"] for r in rows]))
    real_rows = [r for r in rows if r["label"] == "REAL"]
    fake_rows = [r for r in rows if r["label"] == "FAKE"]
    liar_acc_real = float(np.mean([r["correct"] for r in real_rows]))
    liar_acc_fake = float(np.mean([r["correct"] for r in fake_rows]))
    avg_p_by_label = {}
    for lbl in ["true", "mostly-true", "barely-true", "false", "pants-fire"]:
        grp = [r["p_real"] for r in rows if r["liar_label"] == lbl]
        avg_p_by_label[lbl] = round(float(np.mean(grp)), 3)
    print(f"  DistilBERT: accuracy={liar_acc:.4f} (REAL {liar_acc_real:.3f} / FAKE {liar_acc_fake:.3f})")
    print(f"  avg p_real by LIAR label: {avg_p_by_label}")

    print("\n=== Short-headline smoke test (same as TF-IDF) ===")
    p_head = predict_proba(SMOKE_TEST_HEADLINES, tokenizer, model)
    for t, p in zip(SMOKE_TEST_HEADLINES, p_head):
        print(f"  [{('REAL' if p >= 0.5 else 'FAKE'):4s}] {p:.2f} REAL | {t[:70]}")

    print("\n=== Saving ===")
    os.makedirs(OUT_DIR, exist_ok=True)
    model.save_pretrained(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)

    tfidf_in = joblib.load("model/metrics.json")
    tfidf_liar = json.load(open("stress_results.json", encoding="utf-8"))
    comparison = {
        "distilbert": {
            "in_domain": {"accuracy": round(in_acc, 4), "f1": round(in_f1, 4)},
            "liar": {
                "accuracy": round(liar_acc, 4),
                "accuracy_real": round(liar_acc_real, 4),
                "accuracy_fake": round(liar_acc_fake, 4),
                "avg_p_real_by_liar_label": avg_p_by_label,
            },
            "headline_smoke_p_real": [round(float(p), 3) for p in p_head],
        },
        "tfidf": {
            "in_domain": {
                k: tfidf_in["models"][tfidf_in["best_model"]][k]
                for k in ("accuracy", "f1")
            },
            "liar": {
                "accuracy": tfidf_liar["accuracy"],
                "accuracy_real": tfidf_liar["accuracy_real"],
                "accuracy_fake": tfidf_liar["accuracy_fake"],
                "avg_p_real_by_liar_label": {
                    k: v["avg_p_real"] for k, v in tfidf_liar["by_liar_label"].items()
                },
            },
        },
        "settings": {
            "base_model": MODEL_NAME,
            "max_length": MAX_LENGTH,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "train_rows": len(X_train_aug),
        },
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)
    print(f"  saved {OUT_DIR} + {METRICS_PATH}")

    print("\n=== HEAD TO HEAD ===")
    print(f"  {'':12s} {'in-domain':>10s} {'LIAR':>8s}")
    print(f"  {'TF-IDF':12s} {tfidf_in['models'][tfidf_in['best_model']]['accuracy']:>10.4f} {tfidf_liar['accuracy']:>8.4f}")
    print(f"  {'DistilBERT':12s} {in_acc:>10.4f} {liar_acc:>8.4f}")


if __name__ == "__main__":
    main()
