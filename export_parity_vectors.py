r"""
Emits golden feature vectors that the Android unit test asserts against.

This is the guard rail for the bug that broke the previous model. Training and on-device
inference are two independent implementations of the same text->features contract, written in
two languages; nothing stops them drifting apart except a test that compares them on real input.
Last time they drifted, the result was a 21.5% on-device crash rate that every desktop metric
reported as 99% accuracy.

Run this whenever text_features.py changes, then run TextFeaturizerParityTest on the app side.
Probes deliberately cover Devanagari, Tamil, Bengali, Arabic, emoji, non-BMP "mathematical bold"
letters (a real spam evasion trick that lives outside the Basic Multilingual Plane and so needs
code-point-correct iteration in Kotlin), digits, punctuation and empty input.

Digit-masking probes specifically check: runs >= 3 digits masked, runs < 3 left alone, digits
fused to a word without a separator ("Rs50000"), digits split by punctuation ("50,000"), a run
attached on both sides ("call9876543210now"), and non-ASCII digits (Devanagari) - Python's re \d
and Kotlin's \p{Nd} must agree on all of these. mask_digits() is exercised here even though
FEATURE_VERSION did not change, because it's still used for the text uploaded to the backend
(see text_features.py's mask_digits docstring for why it's NOT applied before featurization).
"""
import json
import os

import text_features as tf_feat

# The Android project lives in its own repo, so its location is environment-specific. Override
# with SPAMSHIELD_APP_DIR when it is not at the default Android Studio path.
DEFAULT_APP_DIR = os.path.join(os.path.expanduser("~"), "AndroidStudioProjects", "SpamShield")
APP_DIR = os.environ.get("SPAMSHIELD_APP_DIR", DEFAULT_APP_DIR)
OUT = os.path.join(APP_DIR, "app", "src", "test", "resources", "parity_vectors.json")

PROBES = [
    "",
    "   ",
    "hello",
    "Bhai lottery jeet gaya, turant click karo bit.ly/xy12",
    "आपका बैंक खाता आज बंद हो जाएगा, तुरंत KYC पूरा करें",
    "உங்கள் கணக்கு முடக்கப்படும், உடனே சரிபார்க்கவும்",
    "আপনার অ্যাকাউন্ট বন্ধ হয়ে যাবে",
    "مفت انٹرویو میں جیتنے کے لئے",
    "C0ngratu1ati0ns! U w0n Rs 50,000 c1ick n0w",
    "Your SBI OTP is 448291. Valid 10 min. Do not share.",
    "Aapka SIM 24 ghante me band ho jayega, Aadhaar link kare",
    "MIXED आपका order #12345 ship ho gaya kal tak",
    "punctuation!!! ...   multiple   spaces\t\tand\ttabs",
    "𝐖𝐈𝐍 𝐁𝐈𝐆 now",           # non-BMP: surrogate pairs in UTF-16
    "emoji 🎉🎁 win prize",     # non-BMP, non-letter
    "123 456 7890",
    "a",
    "ab",
    "abc",
    "ThisIsAVeryLongMessageWithoutSpaces" * 12,  # forces MAX_FEATURES truncation
    # --- digit masking (FEATURE_VERSION 2) ---
    "top 10 in India",                              # 2-digit run: below threshold, NOT masked
    "meet at 5 - 6 pm",                              # single digits either side of a separator
    "Rs50000 cashback",                              # digits fused to a preceding word, no space
    "Rs 50,000 cashback offer",                      # comma-separated: "50" (2, kept) + "000" (3, masked)
    "call9876543210now",                             # 10-digit run fused on BOTH sides
    "आपका OTP ४४८२९१ है, साझा न करें",                 # Devanagari digits - \d vs \p{Nd} must agree
    "your PAN ABCDE1234F needs verification",        # digits inside an alphanumeric code (4 digits: masked)
]


def main():
    if not os.path.isdir(APP_DIR):
        raise SystemExit(
            f"Android project not found at {APP_DIR}.\n"
            f"Set SPAMSHIELD_APP_DIR to its location and re-run.")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    mask_probes = [
        "hi",                          # no digits at all
        "10",                          # 2 digits: below threshold, unchanged
        "100",                         # exactly 3 digits: masked
        "call 9876543210 now",
        "Rs50000",                     # fused to a word, no separator
        "50,000",                      # comma-separated digit groups
        "OTP ४४८२९१ valid",            # Devanagari digits
        "a1b2c3",                      # short digit runs interleaved with letters: none masked
    ]

    payload = {
        "contract": {
            "num_buckets": tf_feat.NUM_BUCKETS,
            "max_features": tf_feat.MAX_FEATURES,
            "feature_version": tf_feat.FEATURE_VERSION,
            "char_ngrams": list(tf_feat.CHAR_NGRAMS),
            "use_word_bigrams": tf_feat.USE_WORD_BIGRAMS,
        },
        # A few raw hashes too, so a hash mismatch is diagnosed directly instead of showing up
        # as an opaque whole-vector difference.
        "fnv1a": {s: tf_feat.fnv1a(s) for s in
                  ["", "a", "abc", "<hi>", "आपका", "hello world", "🎉"]},
        # Explicit input->output pairs for mask_digits(), so a Python/Kotlin disagreement on
        # digit masking is diagnosed directly rather than as an opaque feature-vector mismatch.
        "mask_digits": {s: tf_feat.mask_digits(s) for s in mask_probes},
        "cases": [
            {
                "text": t,
                "masked": tf_feat.mask_digits(t),
                "normalized": tf_feat.normalize(t),
                "features": tf_feat.featurize(t),
            }
            for t in PROBES
        ],
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    print(f"wrote {len(PROBES)} golden cases -> {os.path.abspath(OUT)}")


if __name__ == "__main__":
    main()
