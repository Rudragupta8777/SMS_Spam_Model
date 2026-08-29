"""
Fetches the public datasets that preprocess.py expects into data/raw/ and caches them, so the
corpus can be rebuilt from scratch on a clean checkout.

The important addition over the original corpus is dbarbedillo/SMS_Spam_Multilingual_Collection_Dataset:
the UCI SMS Spam Collection machine-translated into 22 languages. It matters for two reasons.

1. It supplies spam AND ham in Hindi, Bengali, Punjabi, Marathi and Urdu. Adding only foreign
   spam would teach the model "Devanagari => spam"; having ham in the same scripts is what
   forces it to learn actual spam semantics instead of a script shortcut.
2. Its translations are imperfect and leave chunks of English in place
   ("मुक्त प्रवेश 2 a wkly comp विजेता FA कप फाइनल tkts"), which is a realistic stand-in for
   the code-mixing this project targets.

Caveat worth stating plainly: these are machine translations, not natively-written SMS. They
teach vocabulary and script patterns but are not a substitute for real native spam. That is why
evaluate.py keeps the (small, real) indian_scam.csv as a separate real-data-only test slice.
"""
import os

import pandas as pd

RAW_DIR = "data/raw"
MULTILINGUAL_PATH = os.path.join(RAW_DIR, "sms_spam_multilingual.csv")
MULTILINGUAL_URL = (
    "https://huggingface.co/api/datasets/dbarbedillo/"
    "SMS_Spam_Multilingual_Collection_Dataset/parquet/default/train/0.parquet"
)

# Indian languages are the project's target; the rest are kept so "works in any language" is
# more than a claim. Each column is the same 5,572 UCI messages rendered in that language.
INDIAN_LANGS = ["hi", "bn", "pa", "mr", "ur"]
OTHER_LANGS = ["es", "fr", "de", "ar", "ru", "zh", "pt", "id", "tr"]


def download_multilingual(force: bool = False) -> str:
    os.makedirs(RAW_DIR, exist_ok=True)
    if os.path.exists(MULTILINGUAL_PATH) and not force:
        print(f"Already cached: {MULTILINGUAL_PATH}")
        return MULTILINGUAL_PATH

    print("Downloading SMS_Spam_Multilingual_Collection_Dataset ...")
    df = pd.read_parquet(MULTILINGUAL_URL)

    keep = ["labels", "text"] + [f"text_{l}" for l in INDIAN_LANGS + OTHER_LANGS]
    keep = [c for c in keep if c in df.columns]
    df = df[keep]
    df.to_csv(MULTILINGUAL_PATH, index=False, encoding="utf-8")
    print(f"Saved {len(df)} rows x {len(keep)} language columns -> {MULTILINGUAL_PATH}")
    return MULTILINGUAL_PATH


if __name__ == "__main__":
    download_multilingual()
    print("\nDone. Next: python preprocess.py")
