"""
Puts the shipped model next to the classical approaches most SMS-spam work actually uses, on the
identical splits, so "is this any good?" has a measured answer instead of a vibe.

Baselines:
  * MultinomialNB + word TF-IDF        - the textbook SMS spam classifier
  * LogisticRegression + word TF-IDF   - strong linear baseline
  * LogisticRegression + char TF-IDF   - the fair fight: char n-grams also survive code-mixing
                                          and obfuscation, so this isolates how much the neural
                                          model adds over "same features, linear model"
  * LinearSVC + char TF-IDF

Deployment footprint is reported too, because a classical model is only "smaller" if you forget
that its fitted vocabulary/IDF table has to ship with it. That table is the thing that broke the
previous version of this project, so it is not a footnote.

Every model is tuned on the SAME validation split and scored on the SAME held-out and real slices.
"""
import os
import pickle
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

import text_features as tf_feat

PROCESSED_DIR = "data/processed"
MODELS_DIR = "models"
SEED = 42


def load_splits():
    tr = pd.read_csv(os.path.join(PROCESSED_DIR, "train.csv"), encoding="utf-8")
    te = pd.read_csv(os.path.join(PROCESSED_DIR, "test.csv"), encoding="utf-8")
    real = pd.read_csv(os.path.join(PROCESSED_DIR, "real_test.csv"), encoding="utf-8")
    for d in (tr, te, real):
        d["text"] = d.text.fillna("").astype(str)
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(tr))
    nval = int(0.15 * len(tr))
    return tr.iloc[perm[nval:]], tr.iloc[perm[:nval]], te, real


def best_threshold(y, scores):
    best_t, best_f1 = 0.5, -1.0
    for t in np.quantile(scores, np.linspace(0.01, 0.99, 99)):
        f = f1_score(y, (scores > t).astype(int), zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t
    return best_t


def score(name, y, pred, extra=""):
    return dict(model=name,
                P=precision_score(y, pred, zero_division=0),
                R=recall_score(y, pred, zero_division=0),
                F1=f1_score(y, pred, zero_division=0), note=extra)


def main():
    tr, val, te, real = load_splits()
    rows_test, rows_real, sizes, latencies = [], [], {}, {}

    specs = [
        ("NB + word tfidf", Pipeline([
            ("v", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            ("c", MultinomialNB())])),
        ("LogReg + word tfidf", Pipeline([
            ("v", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            ("c", LogisticRegression(max_iter=2000, class_weight="balanced"))])),
        ("LogReg + char tfidf", Pipeline([
            ("v", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), min_df=2,
                                  sublinear_tf=True)),
            ("c", LogisticRegression(max_iter=2000, class_weight="balanced"))])),
        ("LinearSVC + char tfidf", Pipeline([
            ("v", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), min_df=2,
                                  sublinear_tf=True)),
            ("c", LinearSVC(class_weight="balanced"))])),
    ]

    for name, pipe in specs:
        t0 = time.perf_counter()
        pipe.fit(tr.text, tr.label)
        fit_s = time.perf_counter() - t0

        def raw(X):
            if hasattr(pipe, "predict_proba"):
                return pipe.predict_proba(X)[:, 1]
            return pipe.decision_function(X)

        thr = best_threshold(val.label.values, raw(val.text))

        t0 = time.perf_counter()
        te_scores = raw(te.text)
        per_msg_ms = (time.perf_counter() - t0) / len(te) * 1000
        latencies[name] = per_msg_ms

        rows_test.append(score(name, te.label.values, (te_scores > thr).astype(int)))
        rows_real.append(score(name, real.label.values, (raw(real.text) > thr).astype(int)))

        blob = pickle.dumps(pipe)
        vocab = len(pipe.named_steps["v"].vocabulary_)
        sizes[name] = (len(blob) / 1024, vocab, fit_s)

    # --- the shipped TFLite model, run exactly as the phone runs it ---
    interp = tf.lite.Interpreter(model_path=os.path.join(MODELS_DIR, "spam_detector.tflite"))
    interp.allocate_tensors()
    inp, outp = interp.get_input_details()[0], interp.get_output_details()[0]
    import json
    thr = json.load(open(os.path.join(MODELS_DIR, "model_meta.json"), encoding="utf-8"))["threshold"]

    def tflite_scores(texts, time_it=False):
        out = []
        t0 = time.perf_counter()
        for t in texts:
            interp.set_tensor(inp["index"], np.array([tf_feat.featurize(t)], dtype=np.int32))
            interp.invoke()
            out.append(float(interp.get_tensor(outp["index"])[0][0]))
        if time_it:
            latencies["THIS MODEL (tflite)"] = (time.perf_counter() - t0) / len(texts) * 1000
        return np.array(out)

    te_s = tflite_scores(te.text.tolist(), time_it=True)
    rows_test.append(score("THIS MODEL (tflite)", te.label.values, (te_s > thr).astype(int)))
    rows_real.append(score("THIS MODEL (tflite)", real.label.values,
                           (tflite_scores(real.text.tolist()) > thr).astype(int)))
    sizes["THIS MODEL (tflite)"] = (
        os.path.getsize(os.path.join(MODELS_DIR, "spam_detector.tflite")) / 1024, 0, float("nan"))

    def table(title, rows):
        print(f"\n{title}")
        print(f"{'model':<24}{'precision':>11}{'recall':>9}{'F1':>9}")
        print("-" * 53)
        for r in sorted(rows, key=lambda r: -r["F1"]):
            print(f"{r['model']:<24}{r['P']:>11.4f}{r['R']:>9.4f}{r['F1']:>9.4f}")

    table("HELD-OUT TEST SPLIT (contains synthetic -> optimistic for everyone)", rows_test)
    table("REAL INDIAN SCAM HOLDOUT (never trained on) <- the one that matters", rows_real)

    print("\nDEPLOYMENT FOOTPRINT")
    print(f"{'model':<24}{'artifact KB':>13}{'vocab entries':>15}{'ms/msg (desktop)':>19}")
    print("-" * 71)
    for name, (kb, vocab, _) in sorted(sizes.items(), key=lambda kv: kv[1][0]):
        v = f"{vocab:,}" if vocab else "none (hashed)"
        print(f"{name:<24}{kb:>13.0f}{v:>15}{latencies.get(name, float('nan')):>19.3f}")

    print("\nThe vocab column is the point: every classical model has to ship a fitted vocabulary")
    print("as a second artifact that must stay in lockstep with the weights. A mismatch there is")
    print("exactly the bug that made the previous model crash on 21.5% of real messages.")


if __name__ == "__main__":
    main()
