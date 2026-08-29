"""
Builds the training corpus from data/raw/ into data/processed/.

WHAT CHANGED AND WHY
--------------------
The previous corpus was 1.66% spam (666 spam vs 39,455 ham), so "98.3% accuracy" was just the
predict-everything-ham baseline, and 124 of its 133 test spam messages were English UCI. It was
an English spam detector wearing a code-mixed label. Three things are fixed here.

1. CLASS BALANCE. Ham is subsampled per source and spam is expanded across languages, landing
   near 1:2 instead of 1:60. Accuracy stops being a meaningless metric.

2. THE SPEAKER-PREFIX LEAK. hinglish_conversations lines look like "Rohan: Hey Radhika!". Every
   one of those 34,884 ham lines carried a "Name:" prefix that no real SMS ever has, so the
   model could score well by learning an artifact that vanishes in production. They are stripped.

3. GROUPED SPLIT. The multilingual corpus is the same 5,572 UCI messages rendered 14 times, and
   several translations leave most of the English intact. A random split would put the English
   original in train and its near-identical Punjabi twin in test, leaking the answer and
   inflating the score. Rows are therefore grouped by source message id and split by GROUP, so
   every translation of a message lands on the same side.

REAL-DATA HOLDOUT
-----------------
indian_scam.csv is the only genuinely real code-mixed Indian data in the project (60 scam / 60
legit). It is written to real_test.csv and NEVER trained on, so evaluate.py can report an honest
real-world number separately from the synthetic-inflated one.
"""
import os
import re

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from download_datasets import INDIAN_LANGS, OTHER_LANGS, MULTILINGUAL_PATH
from generate_synthetic import generate

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

# Per-source ham caps. Ham is abundant and highly redundant; spam is the scarce class.
HAM_CAP_PER_LANG = 800
HAM_CAP_CONVERSATIONS = 8000
HAM_CAP_HINGLISH_GENERAL = 6000
SEED = 42

# "Rohan: ", "Radhika : " - dialogue speaker labels, an artifact of the conversation corpus.
SPEAKER_PREFIX = re.compile(r"^\s*[A-Z][a-zA-Z]{1,20}\s*:\s*")


def clean_text(text) -> str:
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text).strip()


def strip_speaker(line: str) -> str:
    return SPEAKER_PREFIX.sub("", line)


def _add(records, texts, label, group_prefix, groups_start=0):
    """Append rows carrying an explicit group id so the split can keep related rows together."""
    texts = [clean_text(t) for t in texts]
    df = pd.DataFrame({"text": texts, "label": label})
    df["group"] = [f"{group_prefix}:{i + groups_start}" for i in range(len(df))]
    records.append(df)
    return df


def build_corpus():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    records = []
    rng_state = SEED

    # ---------- 1. UCI English base ----------
    uci_path = os.path.join(RAW_DIR, "SMSSpamCollection")
    if os.path.exists(uci_path):
        uci = pd.read_csv(uci_path, sep="\t", header=None, names=["label", "text"])
        uci["label"] = (uci["label"].str.strip().str.lower() == "spam").astype(int)
        # Group by row index; the multilingual file reuses these same indices below.
        uci_spam = uci[uci.label == 1]
        uci_ham = uci[uci.label == 0]
        for sub, lab in ((uci_spam, 1), (uci_ham, 0)):
            df = pd.DataFrame({"text": sub.text.map(clean_text), "label": lab})
            df["group"] = [f"uci:{i}" for i in sub.index]
            records.append(df)
        print(f"UCI English            : {len(uci_spam)} spam / {len(uci_ham)} ham")

    # ---------- 2. Multilingual translations of the SAME UCI messages ----------
    if os.path.exists(MULTILINGUAL_PATH):
        ml = pd.read_csv(MULTILINGUAL_PATH, encoding="utf-8")
        ml["is_spam"] = (ml["labels"].astype(str).str.strip().str.lower() == "spam").astype(int)
        n_spam = n_ham = 0
        for lang in INDIAN_LANGS + OTHER_LANGS:
            col = f"text_{lang}"
            if col not in ml.columns:
                continue
            sub = ml[[col, "is_spam"]].dropna(subset=[col])

            spam = sub[sub.is_spam == 1]
            ham = sub[sub.is_spam == 0].sample(
                n=min(HAM_CAP_PER_LANG, (sub.is_spam == 0).sum()), random_state=rng_state)

            for part, lab in ((spam, 1), (ham, 0)):
                df = pd.DataFrame({"text": part[col].map(clean_text), "label": lab})
                # CRITICAL: group by the ORIGINAL uci row index, shared across all languages.
                df["group"] = [f"uci:{i}" for i in part.index]
                records.append(df)
            n_spam += len(spam)
            n_ham += len(ham)
        print(f"Multilingual (14 langs): {n_spam} spam / {n_ham} ham "
              f"[grouped with their English originals to prevent leakage]")

    # ---------- 3. Hinglish conversations -> ham (speaker prefix stripped) ----------
    conv_dir = os.path.join(RAW_DIR, "hinglish_conversations")
    if os.path.isdir(conv_dir):
        lines = []
        for fname in sorted(os.listdir(conv_dir)):
            if not fname.endswith(".txt"):
                continue
            with open(os.path.join(conv_dir, fname), encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    line = strip_speaker(clean_text(line))
                    if len(line) > 5:
                        lines.append(line)
        conv = pd.Series(lines).drop_duplicates()
        if len(conv) > HAM_CAP_CONVERSATIONS:
            conv = conv.sample(n=HAM_CAP_CONVERSATIONS, random_state=rng_state)
        _add(records, conv.tolist(), 0, "conv")
        print(f"Hinglish conversations : {len(conv)} ham (speaker prefixes stripped)")

    # ---------- 4. Hinglish general (a toxicity corpus; abusive != spam, so all ham) ----------
    hg_path = os.path.join(RAW_DIR, "hinglish_general.csv")
    if os.path.exists(hg_path):
        hg = pd.read_csv(hg_path)
        col = "Text" if "Text" in hg.columns else hg.columns[0]
        texts = hg[col].dropna().drop_duplicates()
        if len(texts) > HAM_CAP_HINGLISH_GENERAL:
            texts = texts.sample(n=HAM_CAP_HINGLISH_GENERAL, random_state=rng_state)
        _add(records, texts.tolist(), 0, "hinglish_gen")
        print(f"Hinglish general       : {len(texts)} ham (personal/abusive msgs, not spam)")

    # ---------- 5. Field telemetry: real spam + user-corrected false positives ----------
    # Written by fetch_telemetry.py. These are the highest-value rows in the corpus: real
    # messages from real phones, and in the label-0 case, the exact mistakes the live model is
    # making. Both labels are honoured - forcing them all to spam (as an earlier version did)
    # would train the model to repeat its own false positives.
    telemetry_path = os.path.join(RAW_DIR, "telemetry_export.csv")
    if os.path.exists(telemetry_path):
        tel = pd.read_csv(telemetry_path, encoding="utf-8").dropna(subset=["text"])
        tel["label"] = tel["label"].astype(int)
        for lab in (0, 1):
            part = tel[tel.label == lab]
            if len(part):
                _add(records, part.text.tolist(), lab, f"telemetry_{lab}")
        n_spam_t = int((tel.label == 1).sum())
        n_ham_t = int((tel.label == 0).sum())
        print(f"Field telemetry        : {n_spam_t} spam / {n_ham_t} corrected false positives")

    # ---------- 6. Synthetic code-mixed Indian spam + hard-negative legit ----------
    syn_spam, syn_ham = generate(n_spam=4000, n_ham=3000, seed=SEED)
    _add(records, syn_spam, 1, "syn_spam")
    _add(records, syn_ham, 0, "syn_ham")
    print(f"Synthetic (train only) : {len(syn_spam)} spam / {len(syn_ham)} legit hard-negatives")

    full = pd.concat(records, ignore_index=True)
    full = full[full.text.str.len() > 0].drop_duplicates(subset=["text"])
    return full


def build_real_holdout():
    """indian_scam.csv - the only real code-mixed Indian data. Never trained on."""
    path = os.path.join(RAW_DIR, "indian_scam.csv")
    if not os.path.exists(path):
        return None
    d = pd.read_csv(path)
    out = pd.DataFrame({
        "text": d["message"].map(clean_text),
        "label": (d["label"].astype(str).str.strip().str.lower() == "scam").astype(int),
        "language": d["language"] if "language" in d.columns else "unknown",
    })
    return out[out.text.str.len() > 0]


def main():
    full = build_corpus()

    # Group-aware split: all translations of one source message stay on the same side.
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, test_idx = next(splitter.split(full, full.label, groups=full["group"]))
    train_df = full.iloc[train_idx].drop(columns=["group"])
    test_df = full.iloc[test_idx].drop(columns=["group"])

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    train_df.to_csv(os.path.join(PROCESSED_DIR, "train.csv"), index=False, encoding="utf-8")
    test_df.to_csv(os.path.join(PROCESSED_DIR, "test.csv"), index=False, encoding="utf-8")

    real = build_real_holdout()
    if real is not None:
        real.to_csv(os.path.join(PROCESSED_DIR, "real_test.csv"), index=False, encoding="utf-8")

    print("\n" + "=" * 62)
    print(f"train      : {len(train_df):6d} rows | spam {train_df.label.mean()*100:5.2f}%")
    print(f"test       : {len(test_df):6d} rows | spam {test_df.label.mean()*100:5.2f}%")
    if real is not None:
        print(f"real_test  : {len(real):6d} rows | spam {real.label.mean()*100:5.2f}%  "
              f"(REAL Indian scam, never trained on)")
    print("=" * 62)
    print(f"Previous corpus was 1.66% spam; predicting all-ham scored 98.3%.")
    print(f"Now spam is {train_df.label.mean()*100:.1f}% of train, so metrics mean something.")


if __name__ == "__main__":
    main()
