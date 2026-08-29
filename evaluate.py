"""
Evaluates the exported .tflite the way the phone will actually run it.

Reports four things the old pipeline never did:
  1. Whether inference SURVIVES every message. The previous model threw
     "gather index out of bounds" on 21.5% of inputs, which no accuracy number would reveal.
  2. Precision / recall / F1, not just accuracy. On the old 1.66%-spam corpus, "98.3% accuracy"
     was literally the score for predicting ham every single time.
  3. A REAL-DATA slice (indian_scam.csv, never trained on) reported separately from the
     synthetic-inflated test split, because the two disagree - see benchmark_archs.py.
  4. A per-script breakdown, since Hindi and Tamil were previously erased to empty strings by
     the app's `[^a-z0-9 ]` regex and so were undetectable by construction.
"""
import json
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)

import text_features as tf_feat

MODELS_DIR = "models"
PROCESSED_DIR = "data/processed"


def load_runtime():
    interp = tf.lite.Interpreter(model_path=os.path.join(MODELS_DIR, "spam_detector.tflite"))
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    meta = json.load(open(os.path.join(MODELS_DIR, "model_meta.json"), encoding="utf-8"))
    return interp, inp, out, meta


def predict(interp, inp, out, texts):
    """Runs one message at a time, exactly like the Android app does."""
    probs, crashes = [], 0
    for t in texts:
        vec = np.array([tf_feat.featurize(t)], dtype=np.int32)
        try:
            interp.set_tensor(inp["index"], vec)
            interp.invoke()
            probs.append(float(interp.get_tensor(out["index"])[0][0]))
        except Exception:
            crashes += 1
            probs.append(0.0)
    return np.array(probs), crashes


def report(name, y, probs, thr):
    pred = (probs > thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    print(f"\n--- {name} (n={len(y)}, spam={int(y.sum())}) ---")
    print(f"  accuracy  {accuracy_score(y,pred):.4f}   precision {precision_score(y,pred,zero_division=0):.4f}")
    print(f"  recall    {recall_score(y,pred,zero_division=0):.4f}   F1        {f1_score(y,pred,zero_division=0):.4f}")
    if len(np.unique(y)) > 1:
        print(f"  ROC-AUC   {roc_auc_score(y,probs):.4f}")
    print(f"  TN={tn} FP={fp} FN={fn} TP={tp}  (FN = spam that slipped through)")
    return f1_score(y, pred, zero_division=0)


def script_of(text):
    for ch in str(text):
        o = ord(ch)
        if 0x0900 <= o <= 0x097F: return "Devanagari"
        if 0x0B80 <= o <= 0x0BFF: return "Tamil"
        if 0x0980 <= o <= 0x09FF: return "Bengali"
        if 0x0A00 <= o <= 0x0A7F: return "Gurmukhi"
        if 0x0600 <= o <= 0x06FF: return "Arabic/Urdu"
    return "Latin"


def main():
    interp, inp, out, meta = load_runtime()
    thr = meta["threshold"]
    print(f"model: spam_detector.tflite  ({os.path.getsize(os.path.join(MODELS_DIR,'spam_detector.tflite'))/1024:.0f} KB)")
    print(f"tuned decision threshold: {thr:.4f}")

    test = pd.read_csv(os.path.join(PROCESSED_DIR, "test.csv"), encoding="utf-8")
    te_texts = test.text.fillna("").astype(str).tolist()
    te_probs, te_crash = predict(interp, inp, out, te_texts)
    report("Held-out test split (contains synthetic -> optimistic)", test.label.values, te_probs, thr)

    real_path = os.path.join(PROCESSED_DIR, "real_test.csv")
    if os.path.exists(real_path):
        real = pd.read_csv(real_path, encoding="utf-8")
        r_texts = real.text.fillna("").astype(str).tolist()
        r_probs, r_crash = predict(interp, inp, out, r_texts)
        report("REAL Indian scam holdout - never trained on", real.label.values, r_probs, thr)
        if "language" in real.columns:
            print("\n  by language (real data):")
            for lang, grp in real.groupby("language"):
                idx = grp.index
                p = r_probs[[real.index.get_loc(i) for i in idx]]
                pred = (p > thr).astype(int)
                print(f"    {lang:<10} n={len(grp):<4} recall={recall_score(grp.label,pred,zero_division=0):.3f} "
                      f"precision={precision_score(grp.label,pred,zero_division=0):.3f}")
    else:
        r_crash = 0

    # --- The headline fix: does inference survive every input? ---
    print("\n" + "=" * 68)
    print("ROBUSTNESS")
    print("=" * 68)
    total_crash = te_crash + r_crash
    print(f"  inference failures across all evaluated messages: {total_crash}")
    print("  (the previous model threw 'gather index out of bounds' on 21.5% of them)")

    print("\n  per-script recall on test spam:")
    spam_rows = test[test.label == 1].copy()
    spam_probs = te_probs[[test.index.get_loc(i) for i in spam_rows.index]]
    spam_rows["script"] = spam_rows.text.map(script_of)
    for sc, grp in spam_rows.groupby("script"):
        p = spam_probs[[spam_rows.index.get_loc(i) for i in grp.index]]
        print(f"    {sc:<12} n={len(grp):<5} recall={(p>thr).mean():.3f}")

    # Obfuscated / code-mixed probes. Not a benchmark - a sanity check that the char n-grams do
    # what they are supposed to do on the tricks real spam uses.
    probes = [
        ("Bhai lottery jeet gaya, turant click karo bit.ly/xy12", 1),
        ("Aapka SIM 24 ghante me band ho jayega, Aadhaar link kare", 1),
        ("आपका बैंक खाता आज बंद हो जाएगा, तुरंत KYC पूरा करें", 1),
        ("C0ngratu1ati0ns! U w0n Rs 50,000 c1ick n0w", 1),
        ("உங்கள் வங்கிக் கணக்கு இன்று முடக்கப்படும், உடனே KYC முடிக்கவும்", 1),
        ("Ungal SIM 24 mani nerathil block agum, Aadhaar link pannunga", 1),
        ("Your SBI OTP is 448291. Valid 10 min. Do not share.", 0),
        ("Aapka Amazon order ship ho gaya, kal tak pahunchega", 0),
        ("आपका ऑर्डर भेज दिया गया है, कल पहुंचेगा", 0),
        ("உங்கள் ஆர்டர் அனுப்பப்பட்டது, நாளை வந்துவிடும்", 0),
        ("yaar kal milte hain coffee ke liye", 0),
    ]
    print("\n  spot checks (expected -> predicted):")
    pp, _ = predict(interp, inp, out, [t for t, _ in probes])
    for (txt, exp), p in zip(probes, pp):
        got = int(p > thr)
        print(f"    [{'OK ' if got==exp else 'MISS'}] p={p:.3f} exp={exp} got={got}  {txt[:52]}")


if __name__ == "__main__":
    main()
