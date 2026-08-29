"""
Decides whether a freshly trained model is allowed to reach users. Exits non-zero to block a
release.

This is the most important script in the automated loop. Everything else in the OTA pipeline is
plumbing; this is the part that stops a bad retrain from silently degrading spam detection on
every installed device. Without it, "retrain automatically and auto-publish" means "ship whatever
the last training run happened to produce".

CHECKS
  1. Runs at all             - the .tflite loads and scores every probe without throwing. The
                               previous generation of this model crashed on 21.5% of real
                               messages while reporting excellent offline accuracy, so "does it
                               execute" is a real check, not a formality.
  2. Feature contract        - model_meta.json still matches text_features.py. A mismatch here
                               means the app would feed the model garbage.
  3. No regression           - real-holdout F1 must not fall more than TOLERANCE below the
                               currently published model. The real holdout is frozen and never
                               trained on.
  4. Absolute floors         - real-holdout precision and recall must clear minimums, so a model
                               cannot pass by being uselessly conservative or trigger-happy.
  5. Behavioural probes      - a handful of messages that must always classify correctly
                               (obvious spam stays spam, a real bank OTP stays ham). Catches
                               catastrophic label flips that aggregate metrics can hide.

Note on the comparison baseline: the live model's metrics come from the backend manifest when
reachable, otherwise from models/published_metrics.json. If neither exists (first ever release)
the regression check is skipped and only the absolute floors apply.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import f1_score, precision_score, recall_score

import text_features as tf_feat

MODELS_DIR = "models"
PROCESSED_DIR = "data/processed"

TOLERANCE = 0.02        # how much real-holdout F1 may drop vs the live model
MIN_PRECISION = 0.90    # a spam blocker that hides real messages is worse than one that misses
MIN_RECALL = 0.80

# Messages whose verdict must never regress, whatever the aggregate numbers say.
PROBES = [
    ("Bhai lottery jeet gaya, turant click karo bit.ly/xy12", 1),
    ("आपका बैंक खाता आज बंद हो जाएगा, तुरंत KYC पूरा करें", 1),
    ("உங்கள் வங்கிக் கணக்கு இன்று முடக்கப்படும், உடனே KYC முடிக்கவும்", 1),
    ("C0ngratu1ati0ns! U w0n Rs 50,000 c1ick n0w", 1),
    ("Your SBI OTP is 448291. Valid 10 min. Do not share.", 0),
    ("Aapka Amazon order ship ho gaya, kal tak pahunchega", 0),
    ("आपका ऑर्डर भेज दिया गया है, कल पहुंचेगा", 0),
    ("yaar kal milte hain coffee ke liye", 0),
]


class GateFailure(Exception):
    pass


def load_candidate():
    path = os.path.join(MODELS_DIR, "spam_detector.tflite")
    meta_path = os.path.join(MODELS_DIR, "model_meta.json")
    if not os.path.exists(path) or not os.path.exists(meta_path):
        raise GateFailure("candidate model or model_meta.json is missing - did train.py run?")

    meta = json.load(open(meta_path, encoding="utf-8"))
    interp = tf.lite.Interpreter(model_path=path)
    interp.allocate_tensors()
    return interp, interp.get_input_details()[0], interp.get_output_details()[0], meta


def predict(interp, inp, out, texts):
    probs = []
    for t in texts:
        interp.set_tensor(inp["index"], np.array([tf_feat.featurize(t)], dtype=np.int32))
        interp.invoke()
        probs.append(float(interp.get_tensor(out["index"])[0][0]))
    return np.array(probs)


def live_metrics(server_url, api_key):
    """Metrics for the model currently in users' hands."""
    if server_url and api_key:
        try:
            import requests
            r = requests.get(f"{server_url.rstrip('/')}/api/model/latest",
                             headers={"x-api-key": api_key}, timeout=15)
            if r.status_code == 200:
                m = r.json().get("metrics")
                if m and "realF1" in m:
                    print(f"  baseline: live model v{r.json().get('version')} "
                          f"(realF1 {m['realF1']:.4f})")
                    return m
            elif r.status_code == 404:
                print("  baseline: none published yet - regression check skipped")
                return None
        except Exception as e:
            print(f"  baseline: backend unreachable ({type(e).__name__}), falling back to local")

    local = os.path.join(MODELS_DIR, "published_metrics.json")
    if os.path.exists(local):
        m = json.load(open(local, encoding="utf-8"))
        print(f"  baseline: published_metrics.json (realF1 {m.get('realF1', float('nan')):.4f})")
        return m
    print("  baseline: none available - regression check skipped")
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=os.path.join(MODELS_DIR, "gate_report.json"))
    args = ap.parse_args()

    server_url = os.environ.get("TELEMETRY_SERVER_URL")
    api_key = os.environ.get("TELEMETRY_API_KEY")
    failures = []

    print("=" * 68)
    print("QUALITY GATE")
    print("=" * 68)

    interp, inp, out, meta = load_candidate()

    # --- 2. feature contract ---
    if meta.get("num_buckets") != tf_feat.NUM_BUCKETS or \
       meta.get("max_features") != tf_feat.MAX_FEATURES:
        failures.append(
            f"feature contract mismatch: model_meta says "
            f"{meta.get('num_buckets')}/{meta.get('max_features')}, "
            f"text_features.py says {tf_feat.NUM_BUCKETS}/{tf_feat.MAX_FEATURES}")
    else:
        print(f"  [ok] feature contract {tf_feat.NUM_BUCKETS} buckets / "
              f"{tf_feat.MAX_FEATURES} features")

    threshold = float(meta["threshold"])

    # --- 1 + 3 + 4. real holdout ---
    real_path = os.path.join(PROCESSED_DIR, "real_test.csv")
    if not os.path.exists(real_path):
        raise GateFailure("real_test.csv missing - cannot evaluate honestly, refusing to pass")

    real = pd.read_csv(real_path, encoding="utf-8")
    texts = real.text.fillna("").astype(str).tolist()
    try:
        probs = predict(interp, inp, out, texts)
    except Exception as e:
        raise GateFailure(f"model threw during inference on real holdout: {e}")

    pred = (probs > threshold).astype(int)
    y = real.label.values
    p = precision_score(y, pred, zero_division=0)
    r = recall_score(y, pred, zero_division=0)
    f1 = f1_score(y, pred, zero_division=0)
    print(f"  [--] real holdout: precision {p:.4f}  recall {r:.4f}  F1 {f1:.4f}  (n={len(y)})")

    if p < MIN_PRECISION:
        failures.append(f"precision {p:.4f} below floor {MIN_PRECISION}")
    if r < MIN_RECALL:
        failures.append(f"recall {r:.4f} below floor {MIN_RECALL}")

    baseline = live_metrics(server_url, api_key)
    if baseline and "realF1" in baseline:
        drop = baseline["realF1"] - f1
        if drop > TOLERANCE:
            failures.append(
                f"real-holdout F1 regressed by {drop:.4f} "
                f"({baseline['realF1']:.4f} -> {f1:.4f}, tolerance {TOLERANCE})")
        else:
            print(f"  [ok] no regression (delta {-drop:+.4f}, tolerance {TOLERANCE})")

    # --- 5. behavioural probes ---
    probe_probs = predict(interp, inp, out, [t for t, _ in PROBES])
    bad = [(t, exp, float(pr)) for (t, exp), pr in zip(PROBES, probe_probs)
           if int(pr > threshold) != exp]
    if bad:
        for t, exp, pr in bad:
            failures.append(f"probe failed (expected {exp}, p={pr:.3f}): {t[:50]}")
    else:
        print(f"  [ok] all {len(PROBES)} behavioural probes correct")

    report = {
        "passed": not failures,
        "realPrecision": p, "realRecall": r, "realF1": f1,
        "threshold": threshold,
        "featureContract": {"numBuckets": tf_feat.NUM_BUCKETS,
                            "maxFeatures": tf_feat.MAX_FEATURES},
        "baselineF1": baseline.get("realF1") if baseline else None,
        "failures": failures,
    }
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("=" * 68)
    if failures:
        print("GATE FAILED - this model will NOT be published:")
        for f_ in failures:
            print(f"  * {f_}")
        print("=" * 68)
        sys.exit(1)

    print(f"GATE PASSED - safe to publish (real F1 {f1:.4f})")
    print("=" * 68)


if __name__ == "__main__":
    try:
        main()
    except GateFailure as e:
        print(f"\nGATE FAILED: {e}")
        sys.exit(1)
