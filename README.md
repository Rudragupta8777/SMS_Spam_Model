# ML Pipeline

Trains the on-device spam classifier and exports a quantized `.tflite` for the Android app.

## Quick start

```bash
python download_datasets.py     # fetch + cache the multilingual corpus
python preprocess.py            # build train/test/real_test
python train.py                 # train + export models/spam_detector.tflite
python evaluate.py              # honest metrics, incl. the real-data slice
python export_parity_vectors.py # refresh the Kotlin parity fixture
```

Then copy `models/spam_detector.tflite` and `models/model_meta.json` into the app's
`app/src/main/assets/`, and run the app's `TextFeaturizerParityTest`.

## Current results

| Slice | Precision | Recall | F1 |
|---|---|---|---|
| Held-out test split (contains synthetic → optimistic) | 0.9504 | 0.9309 | 0.9405 |
| **Real Indian scam holdout** (never trained on) | **1.0000** | **0.9500** | **0.9744** |

Per-script recall on test spam: Latin 0.940, Devanagari 0.909, Bengali 0.908, Arabic/Urdu 0.907,
Gurmukhi 0.860, Tamil 1.000. Inference failures: **0**.

Read the real-data row, not the test row. The test split contains generated messages, so a model
can score well there by memorising templates — `benchmark_archs.py` found variants where test F1
and real F1 moved in *opposite* directions.

**Caveat:** the real holdout is 120 messages (60 spam). It is the only genuinely real code-mixed
Indian data available, and metrics on 60 positives carry wide error bars. Treat 0.95 recall as
"clearly working", not as a precise figure.

## What was wrong before, and what fixed it

The previous pipeline reported ~99% accuracy while being broken in production.

| Problem | Fix |
|---|---|
| `tokenizer.json` had 20,823 words but the embedding had 10,000 rows. Keras dropped over-cap words; Kotlin passed them through, so TFLite threw `gather index out of bounds` on **21.5%** of real messages. | `text_features.py` — features are FNV-1a **hashed** into a fixed bucket count. Every id is `< NUM_BUCKETS` by construction, and there is no vocabulary file to drift. |
| The app's `[^a-z0-9 ]` cleanup deleted every Devanagari/Tamil character, so Hindi spam reached the model as an empty string. | `normalize()` keeps Unicode Letters, **Marks** and Numbers of any script. Marks matter: `ा` in `आपका` is a combining mark, not a letter — keeping them took real Hindi recall from 0.762 to ~0.95. |
| Corpus was **1.66% spam**, so "98.3% accuracy" was just predicting ham every time. | Multilingual spam + synthetic code-mixed spam + per-source ham caps → **33.5% spam**. |
| 124 of 133 test spam messages were English. The code-mixed case was effectively untested. | 14 translated languages + generated Hinglish/Hindi/Tamil/Tanglish spam. |
| 34,884 ham lines carried a `"Rohan: "` speaker prefix that no real SMS has — a giveaway the model could learn instead of spam semantics. | Stripped in `preprocess.py`. |
| A random split let a message's English original and its near-identical Punjabi translation land on opposite sides, leaking the answer. | `GroupShuffleSplit` keyed on the source message id. |

## Files

| File | Role |
|---|---|
| `text_features.py` | **The contract.** Text → hashed feature ids. Mirrored by the app's `TextFeaturizer.kt`. |
| `download_datasets.py` | Fetches `SMS_Spam_Multilingual_Collection_Dataset` (UCI spam/ham in 22 languages). |
| `generate_synthetic.py` | Template generator for code-mixed Indian SMS + hard-negative legit messages. |
| `preprocess.py` | Builds `train.csv` / `test.csv` / `real_test.csv`. |
| `train.py` | Trains, tunes the threshold, exports `.tflite` + `model_meta.json`. |
| `evaluate.py` | Honest metrics: real-data slice, per-script recall, crash check, spot probes. |
| `benchmark_archs.py` | Compares pooling strategies and embedding sizes. |
| `export_parity_vectors.py` | Writes the golden fixture the Kotlin parity test asserts against. |
| `fetch_telemetry.py` | Pulls field-reported spam back in, closing the retrain loop. |

## Datasets

- **UCI SMS Spam Collection** — 747 spam / 4,825 ham, English.
- **`dbarbedillo/SMS_Spam_Multilingual_Collection_Dataset`** — the same UCI messages machine-translated
  into 22 languages. Used for Hindi, Bengali, Punjabi, Marathi, Urdu + 9 others. Its *imperfect*
  translation is a feature here: it leaves English fragments in place
  (`मुक्त प्रवेश 2 a wkly comp विजेता FA कप फाइनल`), which is realistic code-mixing. Both spam **and**
  ham are taken from it — importing only foreign spam would teach the model "Devanagari ⇒ spam".
- **`indian_scam.csv`** — 60 real scam / 60 real legit. **Never trained on**; this is `real_test.csv`.
- **`hinglish_conversations/`** — real Hinglish chat, used as ham after prefix stripping.
- **Synthetic** (`generate_synthetic.py`) — train-only. Covers KYC freeze, customs parcel, lottery,
  UPI/OTP theft, loans, electricity disconnection, Aadhaar/SIM and fake jobs in English, Hindi,
  Hinglish, Tamil and Tanglish. Roughly half its output is *legit* hard negatives (real OTPs,
  delivery updates, payment receipts) which share the surface features of scams — without those
  the model learns "contains a rupee amount ⇒ spam" and flags every genuine bank SMS.

Rejected: `Ngadou/Spam_SMS` — it is Enron **email**, not SMS, and would pull the model off-domain.

`hinglish_general.csv` is retained but nearly worthless: 25,000 rows containing only **40 unique
sentences**, and it is a toxicity corpus rather than a spam one (abusive ≠ spam, so it is all ham).

## Changing the feature contract

`text_features.py` and `TextFeaturizer.kt` are two implementations of one contract. If they drift,
the model silently misbehaves on-device — which is exactly how the 21.5% crash shipped. So:

1. Edit `text_features.py`.
2. `python train.py` (a contract change invalidates the existing model).
3. `python export_parity_vectors.py`.
4. Mirror the change in `TextFeaturizer.kt` and run `./gradlew :app:testDebugUnitTest`.

`SpamClassifier.kt` also logs an error at load time if `model_meta.json`'s `num_buckets` /
`max_features` disagree with the app's constants.
