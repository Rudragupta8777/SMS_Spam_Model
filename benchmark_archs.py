"""
Compares pooling strategies + embedding sizes so train.py's architecture is an evidence-based
choice rather than a guess, and confirms each variant converts to TFLite with BUILTIN ops only
(anything needing SELECT_TF_OPS would fail to load on the phone).

Run: python benchmark_archs.py
"""
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import f1_score, precision_score, recall_score
from tensorflow.keras import layers, models

import text_features as tf_feat

SEED = 42
PROCESSED_DIR = "data/processed"


def pooled_model(mode, dim, num_buckets=tf_feat.NUM_BUCKETS, maxlen=tf_feat.MAX_FEATURES):
    inp = layers.Input(shape=(maxlen,), dtype="int32")
    emb = layers.Embedding(num_buckets, dim)(inp)
    mask = layers.Lambda(lambda t: tf.cast(tf.not_equal(t, 0), tf.float32)[..., None])(inp)
    masked = layers.Multiply()([emb, mask])

    parts = []
    if mode in ("avg", "both"):
        summed = layers.Lambda(lambda t: tf.reduce_sum(t, axis=1))(masked)
        counts = layers.Lambda(lambda m: tf.maximum(tf.reduce_sum(m, axis=1), 1.0))(mask)
        parts.append(layers.Lambda(lambda x: x[0] / x[1])([summed, counts]))
    if mode in ("max", "both"):
        neg = layers.Lambda(lambda x: x[0] + (x[1] - 1.0) * 1e9)([emb, mask])
        parts.append(layers.Lambda(lambda t: tf.reduce_max(t, axis=1))(neg))

    x = parts[0] if len(parts) == 1 else layers.Concatenate()(parts)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation="sigmoid")(x)

    m = models.Model(inp, out)
    m.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(1e-3),
              metrics=[tf.keras.metrics.AUC(curve="PR", name="pr_auc")])
    return m


def tflite_ok(model):
    """True if the model converts using BUILTIN ops only (i.e. will load on-device)."""
    try:
        c = tf.lite.TFLiteConverter.from_keras_model(model)
        c.optimizations = [tf.lite.Optimize.DEFAULT]
        c.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        blob = c.convert()
        return True, len(blob) / 1024
    except Exception as e:
        return False, str(e)[:90]


def main():
    tf.random.set_seed(SEED); np.random.seed(SEED)
    tr = pd.read_csv(os.path.join(PROCESSED_DIR, "train.csv"), encoding="utf-8")
    te = pd.read_csv(os.path.join(PROCESSED_DIR, "test.csv"), encoding="utf-8")
    real = pd.read_csv(os.path.join(PROCESSED_DIR, "real_test.csv"), encoding="utf-8")

    F = lambda s: np.array([tf_feat.featurize(t) for t in s], dtype=np.int32)
    X, y = F(tr.text.fillna("").astype(str)), tr.label.values.astype(np.float32)
    Xte, yte = F(te.text.fillna("").astype(str)), te.label.values.astype(np.float32)
    Xr, yr = F(real.text.fillna("").astype(str)), real.label.values.astype(np.float32)

    rng = np.random.default_rng(SEED); perm = rng.permutation(len(X))
    nval = int(0.15 * len(X)); vi, ti = perm[:nval], perm[nval:]
    pos = y[ti].sum(); neg = len(ti) - pos
    cw = {0: len(ti) / (2 * neg), 1: len(ti) / (2 * pos)}

    print(f"{'variant':<18}{'test F1':>9}{'test P':>9}{'test R':>9}{'REAL F1':>9}"
          f"{'REAL R':>9}{'KB':>8}  builtin-only")
    print("-" * 80)

    results = []
    for mode in ("avg", "max", "both"):
        for dim in (32, 48):
            tag = f"{mode}-d{dim}"
            m = pooled_model(mode, dim)
            m.fit(X[ti], y[ti], validation_data=(X[vi], y[vi]), epochs=30, batch_size=128,
                  class_weight=cw, verbose=0,
                  callbacks=[tf.keras.callbacks.EarlyStopping(
                      monitor="val_pr_auc", mode="max", patience=4, restore_best_weights=True)])

            vp = m.predict(X[vi], verbose=0).ravel()
            # threshold tuned on validation only
            best_t, best_f1 = 0.5, -1
            for t in np.linspace(0.05, 0.95, 91):
                f = f1_score(y[vi], (vp > t).astype(int), zero_division=0)
                if f > best_f1: best_f1, best_t = f, t

            tp = m.predict(Xte, verbose=0).ravel() > best_t
            rp = m.predict(Xr, verbose=0).ravel() > best_t
            ok, info = tflite_ok(m)

            row = (tag, f1_score(yte, tp, zero_division=0), precision_score(yte, tp, zero_division=0),
                   recall_score(yte, tp, zero_division=0), f1_score(yr, rp, zero_division=0),
                   recall_score(yr, rp, zero_division=0), info if ok else -1, ok)
            results.append(row)
            kb = f"{info:.0f}" if ok else "FAIL"
            print(f"{tag:<18}{row[1]:>9.4f}{row[2]:>9.4f}{row[3]:>9.4f}{row[4]:>9.4f}"
                  f"{row[5]:>9.4f}{kb:>8}  {'yes' if ok else info}")

    print("\nREAL F1/R = the 120 real Indian scam messages, never trained on. That column is the")
    print("one that actually predicts field behaviour; test F1 is inflated by synthetic data.")
    best = max(results, key=lambda r: (r[7], r[4], r[1]))
    print(f"\nBest by real-data F1: {best[0]}")


if __name__ == "__main__":
    main()
