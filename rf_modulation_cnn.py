"""Train a 1D CNN modulation classifier on the RadioML 2016.10A dataset.

Dataset: DeepSig RadioML 2016.10A (RML2016.10a_dict.pkl)
  - 220,000 frames, 128 complex samples each (I/Q), 11 modulations, SNR -20..18 dB
  - pickle dict: {(modulation, snr): ndarray of shape (1000, 2, 128)}

Usage:
    python rf_modulation_cnn.py --data RML2016.10a_dict.pkl
"""
import argparse
import json
import pickle

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 0
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ---------------------------------------------------------------- data
def load_radioml(path):
    """Loads the dataset and makes a stratified 60/20/20 split per (mod, snr)."""
    # NOTE: only unpickle files you trust (pickle can execute code).
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="latin1")

    mods = sorted({k[0] for k in d})
    rng = np.random.default_rng(SEED)

    X_list, y_list, snr_list, split_list = [], [], [], []
    for (mod, snr) in sorted(d.keys()):
        arr = np.asarray(d[(mod, snr)], dtype=np.float32)  # (n, 2, 128)
        arr = arr.transpose(0, 2, 1)                       # (n, 128, 2)
        n = len(arr)

        perm = rng.permutation(n)
        n_tr, n_va = int(0.6 * n), int(0.2 * n)
        split = np.empty(n, dtype=np.int8)
        split[perm[:n_tr]] = 0              # train
        split[perm[n_tr:n_tr + n_va]] = 1   # validation
        split[perm[n_tr + n_va:]] = 2       # test

        X_list.append(arr)
        y_list.append(np.full(n, mods.index(mod), dtype=np.int64))
        snr_list.append(np.full(n, snr, dtype=np.int64))
        split_list.append(split)

    X = np.concatenate(X_list)
    y = np.concatenate(y_list)
    snrs = np.concatenate(snr_list)
    split = np.concatenate(split_list)
    return X, y, snrs, split, mods


def normalize(X):
    """Per-frame power normalization (same function is used in the app)."""
    return X / (np.sqrt(np.mean(X ** 2, axis=(1, 2), keepdims=True)) + 1e-8)


# ---------------------------------------------------------------- model
def build_1d_cnn(input_shape, num_classes):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv1D(64, 7, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Conv1D(128, 5, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Conv1D(128, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ---------------------------------------------------------------- evaluation
def plot_snr_accuracy(preds, y_true, snrs, path="snr_robustness.png"):
    snr_vals = np.unique(snrs)
    accs = [float(np.mean(preds[snrs == s] == y_true[snrs == s])) for s in snr_vals]
    for s, a in zip(snr_vals, accs):
        print(f"SNR {s:>4} dB: acc={a:.3f}")

    plt.figure(figsize=(7, 4))
    plt.plot(snr_vals, accs, marker="o")
    plt.title("Classification Accuracy vs. SNR")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_confusion(preds, y_true, snrs, classes, min_snr=0,
                   path="confusion_matrix.png"):
    mask = snrs >= min_snr
    k = len(classes)
    cm = np.zeros((k, k), dtype=int)
    for t, p in zip(y_true[mask], preds[mask]):
        cm[t, p] += 1
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    plt.figure(figsize=(8, 7))
    plt.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    plt.xticks(range(k), classes, rotation=60, ha="right")
    plt.yticks(range(k), classes)
    for i in range(k):
        for j in range(k):
            plt.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center",
                     fontsize=7,
                     color="white" if cm_norm[i, j] > 0.5 else "black")
    plt.title(f"Confusion Matrix (SNR >= {min_snr} dB)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_demo_samples(X, y, snrs, split, per_combo=5, path="demo_samples.npz"):
    """Saves a few raw test frames per (class, SNR) for the Streamlit demo."""
    sel = []
    for c in np.unique(y):
        for s in np.unique(snrs):
            idx = np.where((y == c) & (snrs == s) & (split == 2))[0][:per_combo]
            sel.extend(idx.tolist())
    sel = np.array(sel)
    np.savez_compressed(path, X=X[sel], y=y[sel], snr=snrs[sel])
    print(f"Saved {len(sel)} demo frames to {path}")


# ---------------------------------------------------------------- main
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="RML2016.10a_dict.pkl")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    X, y, snrs, split, classes = load_radioml(args.data)
    print(f"Loaded {len(X)} frames, classes: {classes}")

    Xn = normalize(X)
    tr, va, te = split == 0, split == 1, split == 2

    model = build_1d_cnn(Xn.shape[1:], len(classes))
    model.summary()

    model.fit(
        Xn[tr], y[tr],
        validation_data=(Xn[va], y[va]),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[
            callbacks.EarlyStopping(monitor="val_loss", patience=8,
                                    restore_best_weights=True),
            callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                        patience=3),
        ],
    )

    test_loss, test_acc = model.evaluate(Xn[te], y[te], batch_size=1024, verbose=0)
    print(f"\nOverall test accuracy: {test_acc:.3f}\n")

    model.save("rf_model.keras")
    with open("classes.json", "w") as f:
        json.dump(classes, f)

    preds = np.argmax(model.predict(Xn[te], batch_size=1024, verbose=0), axis=1)
    plot_snr_accuracy(preds, y[te], snrs[te])
    plot_confusion(preds, y[te], snrs[te], classes, min_snr=0)
    save_demo_samples(X, y, snrs, split)
