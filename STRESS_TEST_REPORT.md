# Stress Test: The Model Falls Apart Out-of-Distribution

**TL;DR: 96% accuracy on the dataset it was trained on, 48.8% on LIAR — a coin flip.**
The model didn't learn "fake vs. real news." It learned "what Reuters wire copy looks like vs. what tabloid-style articles look like."

## Setup

- **Model under test:** the deployed `model/model.pkl` (word+char TF-IDF → calibrated LinearSVC, trained only on `fake_or_real_news.csv`)
- **Stress set:** [LIAR](https://www.cs.ucsb.edu/~william/data/liar_dataset) — 1,016 scored PolitiFact statements from 2016-era politics (267 `half-true` excluded as genuinely ambiguous). Short spoken claims, not articles. Completely different source, format, and era.
- **Mapping:** `true`/`mostly-true` → REAL, `barely-true`/`false`/`pants-fire` → FAKE
- Run it yourself: `py stress_test.py` → [stress_results.json](stress_results.json)

## Results

### Overall: 48.8% accuracy (worse than always guessing FAKE, which scores 54.7%)

| Subset | n | Accuracy | Avg p(REAL) |
|---|---|---|---|
| All statements | 1016 | 48.8% | 0.524 |
| REAL only | 460 | 50.9% | — |
| FAKE only | 556 | 47.1% | — |

### By original LIAR label — the ordering is *backwards*

| LIAR label | n | Accuracy | Avg p(REAL) |
|---|---|---|---|
| true | 211 | 51.7% | 0.513 |
| mostly-true | 249 | 50.2% | 0.511 |
| barely-true | 214 | 39.7% | **0.561** ⚠️ |
| false | 250 | 50.4% | 0.519 |
| pants-fire | 92 | 55.4% | 0.470 |

`barely-true` statements get the *highest* average REAL score — the model's confidence is anti-correlated with truthfulness in the middle of the scale. It has zero signal here.

### By length — no rescue from longer text

| Length | n | Accuracy | Avg p(REAL) |
|---|---|---|---|
| short (<15 words) | 383 | 49.6% | 0.547 |
| medium (15–30) | 555 | 47.7% | 0.506 |
| long (30+) | 78 | 52.6% | 0.491 |

Note this *despite* our title augmentation — the model handles short text without collapsing to FAKE anymore (headline smoke tests still pass), but "handles" ≠ "classifies correctly."

### By topic — nothing above 61%

Taxes (41.1%), health-care (45.0%), and federal-budget (44.7%) — the biggest topics — score worst. State-budget (60.3%) and education (52.2%) are best, likely by chance in n≈70 samples.

## The failure modes, in the model's own words

**True statements it confidently calls FAKE (p_real ≈ 0.01):**
- "The Fed created $1.2 trillion out of nothing, gave it to banks…" (labeled `true`)
- "On my first day in office, I ordered a review of every regulation…" (`mostly-true`)

**False statements it confidently calls REAL (p_real ≈ 1.00):**
- "The White House is won in the swing states, and I am winning the swing states." (labeled `false`)
- "McCain 'hasn't held executive responsibility.'" (`false`)

Both directions fail *confidently* — calibration does not save you when the features themselves are meaningless for the new distribution.

## Why it breaks down

1. **Style is not truth.** The training signal was stylistic: Reuters wire copy (REAL) vs. tabloid-style articles (FAKE). LIAR statements are neutral, sourced, political speech — the style gap the model relies on doesn't exist.
2. **Domain + era shift.** 2016 politics, spoken claims, partisan framing — none of these patterns appear in the 2014–2016 news articles it trained on.
3. **Different task.** Fact-checking a specific claim's veracity ≠ detecting fake *articles*. LIAR's `barely-true` is about misleading framing, which no bag-of-ngrams model can catch.

## What this means for the app

The app is honest within its niche: paste in a news-article-style text and it measures "does this read like tabloid-style or wire-style writing" at ~96%. It is **not** a lie detector, and on political one-liners it's a random guess. The footer disclaimer is doing real work.

## What would actually generalize

- Train on multiple *domains* (articles + statements + social posts) with domain-adversarial techniques or domain labels as features.
- Use models with actual world knowledge (fine-tuned transformers, LLMs) that can reason about the claim's content, not just its wording.
- Accept the task decomposition: "style mimicry detection" and "claim verification" are different problems.

## Files

- [stress_test.py](stress_test.py) — reproducible evaluation script
- [stress_results.json](stress_results.json) — raw numbers + all failure exemplars
- `liar_test.tsv` — the test data (downloaded from the tfs4/liar_dataset GitHub mirror)
