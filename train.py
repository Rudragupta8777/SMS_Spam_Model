"""
Trains the on-device spam classifier and exports a quantized .tflite.

DESIGN NOTES
------------
Input is the hashed feature vector from text_features.py (word unigrams + bigrams + char 3/4-grams,
FNV-1a hashed into 20k buckets). Because features are hashed there is no vocabulary asset, so the
"tokenizer says 20,823 but the embedding has 10,000 rows" crash that broke the previous model
cannot recur, and Devanagari/Tamil survive normalization instead of being erased.

The architecture is fastText-style: embed every feature, then average-pool. Convolution is
deliberately not used: the feature vector is ordered by feature TYPE (words, then bigrams, then
char-grams), not by position in the sentence, so a 1D kernel sliding over it has no meaningful
locality to exploit.

Average pooling (dim 32) was chosen from benchmark_archs.py, and the reason is worth recording
because it is counter-intuitive. Max pooling scores BETTER on the held-out test split
(F1 0.9515 vs 0.9357) but WORSE on the 120 real Indian scam messages (F1 0.8571 / recall 0.75 vs
0.9189 / 0.85). Concatenating avg+max is worse still on real data (F1 0.80). Max pooling lets one
strong feature decide the output, which is an easy win on templated synthetic spam and a poor
proxy for messages a generator never produced. Averaging forces evidence to accumulate across the
whole message and transfers better. Selecting on test F1 here would have shipped the weaker model,
which is exactly why evaluate.py reports the real-data slice separately.

Every op here stays inside the TFLite BUILTIN set on purpose. Pulling in SELECT_TF_OPS would
require the Flex delegate, which the app's `com.google.ai.edge.litert:litert` dependency does not
bundle - the model would convert fine on the desktop and then fail to load on the phone.

Training details that matter:
  * class_weight balances the remaining 1:2 skew so the loss cannot ignore spam.
  * Early stopping tracks validation PR-AUC, which is informative under class imbalance in a way
    that accuracy is not.
  * The decision threshold is tuned for best F1 on validation rather than assumed to be 0.5, and
    written to model_meta.json for the app to read.
"""
import json
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import f1_score, precision_recall_curve
from tensorflow.keras import layers, models

import text_features as tf_feat

PROCESSED_DIR = "data/processed"
MODELS_DIR = "models"
SEED = 42
EMBED_DIM = 32  # benchmark_archs.py: best real-data F1, and half the size of dim 48
EPOCHS = 30
BATCH_SIZE = 128


def build_model(num_buckets=tf_feat.NUM_BUCKETS, maxlen=tf_feat.MAX_FEATURES, dim=EMBED_DIM):
    inp = layers.Input(shape=(maxlen,), dtype="int32", name="features")

    # mask_zero=False keeps the op set TFLite-friendly; padding is handled by the explicit mask
    # below so zero-padded slots cannot drag the average toward the pad embedding.
    emb = layers.Embedding(num_buckets, dim, name="embedding")(inp)

    mask = layers.Lambda(
        lambda t: tf.cast(tf.not_equal(t, 0), tf.float32)[..., None],
        name="pad_mask")(inp)
    masked = layers.Multiply(name="apply_mask")([emb, mask])

    # Masked mean: sum only the real features and divide by how many there were, so a short
    # message is not diluted toward the padding embedding by its ~200 empty slots.
    summed = layers.Lambda(lambda t: tf.reduce_sum(t, axis=1), name="sum_pool")(masked)
    counts = layers.Lambda(
        lambda m: tf.maximum(tf.reduce_sum(m, axis=1), 1.0), name="token_count")(mask)
    x = layers.Lambda(lambda x: x[0] / x[1], name="avg_pool")([summed, counts])

    x = layers.Dense(64, activation="relu", name="dense_1")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = models.Model(inp, x, name="spam_detector")
    model.compile(
        loss="binary_crossentropy",
        optimizer=tf.keras.optimizers.Adam(1e-3),
        metrics=[tf.keras.metrics.AUC(curve="PR", name="pr_auc"),
                 tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")],
    )
    return model


def featurize(texts):
    return np.array([tf_feat.featurize(t) for t in texts], dtype=np.int32)


def tune_threshold(y_true, probs):
    """Pick the probability cut that maximises F1, rather than assuming 0.5."""
    precision, recall, thresholds = precision_recall_curve(y_true, probs)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(precision), where=(precision + recall) > 0)
    best = int(np.argmax(f1[:-1])) if len(thresholds) else 0
    return float(thresholds[best]) if len(thresholds) else 0.5, float(f1[best])


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    train_df = pd.read_csv(os.path.join(PROCESSED_DIR, "train.csv"), encoding="utf-8")
    test_df = pd.read_csv(os.path.join(PROCESSED_DIR, "test.csv"), encoding="utf-8")

    print("Featurizing (hashed n-grams, no vocabulary file) ...")
    X = featurize(train_df.text.fillna("").astype(str).tolist())
    y = train_df.label.values.astype(np.float32)
    Xte = featurize(test_df.text.fillna("").astype(str).tolist())
    yte = test_df.label.values.astype(np.float32)

    # Carve a validation slice out of train for early stopping + threshold tuning, so the
    # reported test number is never used to make a modelling decision.
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(X))
    n_val = int(0.15 * len(X))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    Xtr, ytr, Xval, yval = X[tr_idx], y[tr_idx], X[val_idx], y[val_idx]

    pos, neg = ytr.sum(), len(ytr) - ytr.sum()
    class_weight = {0: len(ytr) / (2 * neg), 1: len(ytr) / (2 * pos)}
    print(f"train {len(Xtr)} (spam {pos/len(ytr)*100:.1f}%) | val {len(Xval)} | test {len(Xte)}")
    print(f"class_weight: {class_weight}")

    model = build_model()
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_pr_auc", mode="max", patience=4,
                                         restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_pr_auc", mode="max", factor=0.5,
                                             patience=2, min_lr=1e-5, verbose=1),
    ]

    model.fit(Xtr, ytr, validation_data=(Xval, yval), epochs=EPOCHS, batch_size=BATCH_SIZE,
              class_weight=class_weight, callbacks=callbacks, verbose=2)

    val_probs = model.predict(Xval, verbose=0).ravel()
    threshold, val_f1 = tune_threshold(yval, val_probs)
    print(f"\nTuned threshold {threshold:.4f} (val F1 {val_f1:.4f}); "
          f"F1 at the naive 0.5 would be {f1_score(yval, (val_probs>0.5).astype(int)):.4f}")

    te_probs = model.predict(Xte, verbose=0).ravel()
    print(f"held-out test F1 @tuned: {f1_score(yte, (te_probs>threshold).astype(int)):.4f}")

    # ---------- Export ----------
    keras_path = os.path.join(MODELS_DIR, "spam_detector.keras")
    model.save(keras_path)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # BUILTINS only - see the module docstring. If this ever raises a "requires SELECT_TF_OPS"
    # error, fix the layer that caused it rather than widening the op set, or the model will
    # load on the desktop and fail on the phone.
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    tflite_model = converter.convert()

    tflite_path = os.path.join(MODELS_DIR, "spam_detector.tflite")
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    meta = {
        "threshold": threshold,
        "num_buckets": tf_feat.NUM_BUCKETS,
        "max_features": tf_feat.MAX_FEATURES,
        "char_ngrams": list(tf_feat.CHAR_NGRAMS),
        "use_word_bigrams": tf_feat.USE_WORD_BIGRAMS,
        "embed_dim": EMBED_DIM,
    }
    with open(os.path.join(MODELS_DIR, "model_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    size_kb = os.path.getsize(tflite_path) / 1024
    print(f"\nExported {tflite_path} ({size_kb:.0f} KB) + model_meta.json")
    print("No tokenizer.json is emitted any more - features are hashed, so there is no "
          "vocabulary that can drift from the model.")
    print("\nNext: python evaluate.py")


if __name__ == "__main__":
    main()
