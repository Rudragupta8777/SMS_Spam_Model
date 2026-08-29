"""
Shared text->feature contract between training (Python) and on-device inference (Kotlin).

WHY THIS EXISTS
---------------
The previous pipeline exported a 20,823-word `tokenizer.json` but sized the model's Embedding
at 10,000. Keras silently DROPS words whose index exceeds `num_words`; the Kotlin port looked
them up raw and fed indices up to 20,823 into a 10,000-row embedding. Result: TFLite threw
"gather index out of bounds" on 21.5% of real messages. The Kotlin regex `[^a-z0-9 ]` also
erased every Devanagari/Tamil character, so native-script spam arrived as an empty string.

Both bugs share one root cause: a vocabulary file is a second source of truth that can drift
from the model. So there is no vocabulary here. Features are HASHED into a fixed bucket count
with FNV-1a, which means:

  * No asset to ship, load, or keep in sync -> the drift bug class is structurally impossible.
  * Every hashed id is < NUM_BUCKETS by construction -> an out-of-range gather cannot happen.
  * Script-agnostic: Devanagari, Tamil, Latin and code-mixed text all hash identically well.
  * Char n-grams give robustness to the deliberate misspellings spam relies on
    ("l0ttery", "clikc", "w1n") and to Hinglish spelling variation ("kare"/"karo"/"karein").

Anything changed in this file MUST be mirrored in the Kotlin TextFeaturizer, which is enforced
by parity_vectors.json + TextFeaturizerParityTest (see export_parity_vectors.py).
"""

import re
import unicodedata

# --- Feature contract. Changing any of these invalidates a trained model. ---
NUM_BUCKETS = 20000   # hashed feature space; also the model's embedding vocab size
MAX_FEATURES = 256    # fixed input length fed to the model
PAD_ID = 0            # reserved, never emitted by the hash
CHAR_NGRAMS = (3, 4)  # char n-gram sizes taken inside word boundaries
USE_WORD_BIGRAMS = True

# Bumped whenever tokenize()/normalize() semantics change, even if NUM_BUCKETS/MAX_FEATURES stay
# the same. That distinction matters: a model trained under the OLD tokenization loaded with the
# NEW code (or vice versa) would not throw - it would silently classify with different features
# than it was trained on. NUM_BUCKETS/MAX_FEATURES only catch a SHAPE mismatch; this would catch
# a SEMANTIC one. Currently 1 (unchanged) - see mask_digits' docstring for why digit masking
# stayed OUT of this pipeline despite being built and measured.
FEATURE_VERSION = 1

MASK_TOKEN = "XXXX"
MIN_DIGIT_RUN_TO_MASK = 3
_DIGIT_RUN = re.compile(r"\d{%d,}" % MIN_DIGIT_RUN_TO_MASK)


def mask_digits(text: str) -> str:
    """Replaces runs of MIN_DIGIT_RUN_TO_MASK+ consecutive digits with a fixed placeholder.

    Used ONLY for the text uploaded to the telemetry backend (TelemetryClient calls this
    directly before building the request), NOT inside normalize()/tokenize() below. That
    exclusion is deliberate and measured, not an oversight:

    Masking digits before feature extraction was implemented and evaluated. Two independent
    retrains both landed real-holdout F1 around 0.85-0.88, down from 0.9744, with the drop
    outside the model's own bootstrap confidence interval - a real regression, not noise. The
    false negatives it introduced mostly did not even contain digits, which means the harm
    wasn't "losing signal from this message's own numbers" - masking every digit-heavy spam
    template down to a shared "XXXX" token changed what the whole embedding table learns to
    weight, and that shift generalized worse to the small set of genuinely real spam messages
    this project has. So: raw digits ARE still fed to the classifier (unchanged from before this
    feature existed), but they never reach the backend - see TelemetryClient's privacy comment.

    The placeholder is a FIXED string regardless of the matched run's length, so a 4-digit OTP
    and a 10-digit phone number both become "XXXX" - the placeholder's length does not leak how
    many digits were redacted. Padded with spaces so "Rs50000" splits into "rs" + "XXXX" instead
    of merging into one unrecognizable token.

    Deliberately per contiguous run only, not separator-spanning: "98765-43210" masks to
    "XXXX-XXXX", not one blended placeholder. A regex greedy enough to span separators would
    also catch ordinary sentences like "meet at 5 - 6 pm", which this avoids.

    `\\d` matches any Unicode decimal digit (category Nd) for str patterns in Python 3, not just
    ASCII 0-9, so Devanagari/Arabic-indic digits are masked too - the Kotlin port uses `\\p{Nd}`
    for the same reason; see the parity fixture for a probe covering non-ASCII digits.
    """
    if not isinstance(text, str):
        return ""
    return _DIGIT_RUN.sub(f" {MASK_TOKEN} ", text)


_FNV_OFFSET = 0x811C9DC5
_FNV_PRIME = 0x01000193
_MASK32 = 0xFFFFFFFF


def fnv1a(text: str) -> int:
    """32-bit FNV-1a over UTF-8 bytes. Mirrored exactly in Kotlin."""
    h = _FNV_OFFSET
    for byte in text.encode("utf-8"):
        h ^= byte
        h = (h * _FNV_PRIME) & _MASK32
    return h


def bucket(token: str) -> int:
    """Map a token into [1, NUM_BUCKETS). 0 stays reserved for padding."""
    return 1 + (fnv1a(token) % (NUM_BUCKETS - 1))


def _is_kept(ch: str) -> bool:
    """Keep Unicode Letters, Marks and Numbers; drop everything else.

    Marks (category M*) matter enormously here. Devanagari, Tamil and Bengali write vowels as
    combining marks - the 'ा' in 'आपका' is category Mc, not a letter - so testing `isalpha()`
    silently strips them and reduces 'आपका' to 'आपक'. That throws away most of the vowel
    information in every Indic message.

    Categories L, M and N are exactly Java's Character.getType() constants 1..11, which is what
    lets the Kotlin port match this byte for byte.
    """
    return unicodedata.category(ch)[0] in ("L", "M", "N")


def normalize(text: str) -> str:
    """Lowercases and keeps only letters/marks/digits of ANY script, collapsing everything else
    to a single space. Unlike the old `[^a-z0-9 ]` regex this preserves Devanagari/Tamil/Arabic.

    Deliberately does NOT mask digits (see mask_digits' docstring) - that was tried and measurably
    hurt real-holdout recall, so raw digits still reach the classifier here."""
    if not isinstance(text, str):
        return ""
    out = []
    prev_space = True  # leading spaces get collapsed away
    for ch in text.lower():
        if _is_kept(ch):
            out.append(ch)
            prev_space = False
        elif not prev_space:
            out.append(" ")
            prev_space = True
    return "".join(out).strip()


def tokenize(text: str) -> list:
    return normalize(text).split()


def extract_features(text: str) -> list:
    """Produce the ordered list of hashed feature ids for one message.

    Order matters only for truncation: word-level features are emitted first so that a very
    long message keeps its words and loses only trailing char n-grams.
    """
    words = tokenize(text)
    if not words:
        return []

    feats = [bucket(w) for w in words]

    if USE_WORD_BIGRAMS:
        # Space separator: normalized text is only letters/digits/spaces and a word never
        # contains a space, so the bigram "a b" cannot collide with the single word "ab".
        feats.extend(bucket(words[i] + " " + words[i + 1]) for i in range(len(words) - 1))

    # Word-boundary markers so a prefix/suffix n-gram is distinct from a mid-word one.
    for w in words:
        padded = "<" + w + ">"
        for n in CHAR_NGRAMS:
            if len(padded) >= n:
                feats.extend(bucket(padded[i:i + n]) for i in range(len(padded) - n + 1))

    return feats


def featurize(text: str) -> list:
    """Fixed-length, zero-padded feature vector ready for the model."""
    feats = extract_features(text)[:MAX_FEATURES]
    return feats + [PAD_ID] * (MAX_FEATURES - len(feats))


def featurize_batch(texts) -> "np.ndarray":
    import numpy as np
    return np.array([featurize(t) for t in texts], dtype=np.float32)
