# Fake-News-

[![CI](https://github.com/Teninlanimi/Fake-News-/actions/workflows/ci.yml/badge.svg)](https://github.com/Teninlanimi/Fake-News-/actions/workflows/ci.yml)

An AI-powered fake news detection system using NLP and machine learning to classify news articles and provide confidence based predictions.

Every push retrains the models and runs the accuracy gate (94% minimum on the held-out test set) in [GitHub Actions](.github/workflows/ci.yml); PRs run a fast smoke-test mode. The out-of-domain LIAR stress test runs too, as an informational check.

## What's inside

- [News_detector.ipynb](News_detector.ipynb) - the learning notebook: data exploration, TF-IDF, Logistic Regression baseline, transformer pipelines, and a final "leveling up" section that loads the improved model.
- [train_model.py](train_model.py) - trains and compares three models and saves the winner:
  - baseline: word TF-IDF + Logistic Regression (the notebook's original approach)
  - tuned: stronger regularization for sparse text
  - **best: word + character TF-IDF -> LinearSVC wrapped in `CalibratedClassifierCV`** for honest probabilities
  - the training set is augmented with article titles so the model also handles short headline-style input
- [featurizer.py](featurizer.py) - the shared word+char TF-IDF featurizer (imported by both training and serving, which is what makes the saved model loadable)
- [app.py](app.py) + [templates/index.html](templates/index.html) - a Flask web app: paste a headline, get REAL/FAKE with a confidence score

## Results (held-out test set, 1,267 articles)

| Model | Accuracy | F1 |
|---|---|---|
| Logistic Regression (baseline) | 94.6% | 0.946 |
| Logistic Regression (tuned C) | 95.9% | 0.959 |
| **Calibrated Linear SVC (+title augmentation)** | **95.6%** | **0.956** |

The SVC wins on raw accuracy (96.3% without augmentation), but augmentation trades ~0.7pt of article-level accuracy for dramatically better behavior on short headlines — a plausible real headline goes from 25% to 79% "REAL" confidence, which matters most in the app.

## Run it

```bash
# 1. install dependencies
py -m pip install -r requirements.txt

# 2. train + save the model (writes model/model.pkl, ~19 MB, not committed)
py train_model.py

# 3. start the web app
py app.py
# open http://127.0.0.1:5000
```

> Note: training downloads the dataset (~6,300 articles) from the web on each run.

## Honest caveats

This is a pattern matcher trained on a single dataset, not a ground-truth fact checker. It learns stylistic patterns (wire-service phrasing vs. sensationalist writing) from one corpus, so it can be confidently wrong on topics or styles it hasn't seen. The footer in the app says the same.

How confidently wrong? We stress-tested it on the [LIAR dataset](https://www.cs.ucsb.edu/~william/data/liar_dataset) (short PolitiFact statements it never saw): **accuracy drops from ~96% to 48.8% — a coin flip**. The model learned "what wire copy looks like," not "what's true." Full breakdown in [STRESS_TEST_REPORT.md](STRESS_TEST_REPORT.md); rerun it with `py stress_test.py`.
