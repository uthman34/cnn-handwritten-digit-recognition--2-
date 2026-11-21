"""
Convolutional Neural Network (CNN) Training & Browser Export Script (EMNIST Balanced)
Final Year CS Project: Handwritten Character & Digit Recognition (47 Classes)

Downloads the official NIST EMNIST Balanced dataset (112,800 train / 18,800 test
across digits 0-9, uppercase A-Z, and 11 distinct lowercase letters), trains a
47-class sequential CNN with TensorFlow/Keras, and exports the browser-ready
TensorFlow.js LayersModel to ../public/tfjs_model.
"""

import gzip
import json
import struct
import subprocess
import sys
import time
import types
import zlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tensorflow as tf

# TensorFlow.js imports this package while loading converter modules, but this
# project exports a regular Keras Sequential model and never uses TF Decision Forests.
if "tensorflow_decision_forests" not in sys.modules:
    sys.modules.setdefault(
        "tensorflow_decision_forests",
        types.ModuleType("tensorflow_decision_forests"),
    )

import tensorflowjs as tfjs

ROOT_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = Path(__file__).resolve().parent / ".cache" / "emnist"
OUTPUT_MODEL_DIR = ROOT_DIR / "public" / "emnist_model"

EMNIST_ZIP_URL = "https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip"
EMNIST_HEAD_BYTES = 33_100_000  # First ~33.1 MB covers all 5 emnist-balanced files

IMAGE_HEIGHT = 28
IMAGE_WIDTH = 28
IMAGE_CHANNELS = 1
NUM_CLASSES = 47


def ensure_emnist_balanced_files():
    """Downloads and extracts the 5 EMNIST Balanced files via HTTP range request."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    required = [
        "emnist-balanced-mapping.txt",
        "emnist-balanced-train-images-idx3-ubyte.gz",
        "emnist-balanced-train-labels-idx1-ubyte.gz",
        "emnist-balanced-test-images-idx3-ubyte.gz",
        "emnist-balanced-test-labels-idx1-ubyte.gz",
    ]
    if all((CACHE_DIR / name).exists() for name in required):
        return

    head_zip = CACHE_DIR / "emnist_balanced_head.zip"
    if not head_zip.exists():
        print(f"Downloading EMNIST Balanced split from {EMNIST_ZIP_URL} ...")
        subprocess.run(
            ["curl", "-sL", "-r", f"0-{EMNIST_HEAD_BYTES}", "-o", str(head_zip), EMNIST_ZIP_URL],
            check=True,
        )

    blob = head_zip.read_bytes()
    pos = 0
    extracted = 0
    while pos + 30 <= len(blob) and extracted < len(required):
        if blob[pos : pos + 4] != b"PK\x03\x04":
            break
        _, _, comp_method, _, _, _, comp_size, _, fn_len, ex_len = struct.unpack(
            "<HHHHHIIIHH", blob[pos + 4 : pos + 30]
        )
        fname = blob[pos + 30 : pos + 30 + fn_len].decode("utf-8")
        data_start = pos + 30 + fn_len + ex_len
        raw = blob[data_start : data_start + comp_size]
        out = zlib.decompress(raw, -zlib.MAX_WBITS) if comp_method == 8 else raw
        base = Path(fname).name
        if base in required:
            (CACHE_DIR / base).write_bytes(out)
            extracted += 1
        pos = data_start + comp_size


def read_idx_images(gz_path: Path) -> np.ndarray:
    """Reads IDX3 gzip images, normalizes to [0, 1], and transposes EMNIST to upright orientation."""
    with gzip.open(gz_path, "rb") as f:
        magic, count, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Unexpected IDX3 magic number: {magic}")
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape(count, rows, cols)
    # EMNIST images are stored column-major; transpose (rows, cols) -> (cols, rows)
    upright = np.transpose(data, (0, 2, 1)).astype("float32") / 255.0
    return np.expand_dims(upright, axis=-1)


def read_idx_labels(gz_path: Path) -> np.ndarray:
    """Reads IDX1 gzip labels."""
    with gzip.open(gz_path, "rb") as f:
        magic, _ = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Unexpected IDX1 magic number: {magic}")
        return np.frombuffer(f.read(), dtype=np.uint8)


def load_emnist_balanced():
    """Loads 47-class EMNIST Balanced dataset tensors and character labels."""
    ensure_emnist_balanced_files()

    mapping_lines = (CACHE_DIR / "emnist-balanced-mapping.txt").read_text(encoding="utf-8").splitlines()
    class_labels = [chr(int(line.split()[1])) for line in mapping_lines if line.strip()]

    x_train = read_idx_images(CACHE_DIR / "emnist-balanced-train-images-idx3-ubyte.gz")
    y_train_raw = read_idx_labels(CACHE_DIR / "emnist-balanced-train-labels-idx1-ubyte.gz")
    x_test = read_idx_images(CACHE_DIR / "emnist-balanced-test-images-idx3-ubyte.gz")
    y_test_raw = read_idx_labels(CACHE_DIR / "emnist-balanced-test-labels-idx1-ubyte.gz")

    y_train = tf.keras.utils.to_categorical(y_train_raw, len(class_labels))
    y_test = tf.keras.utils.to_categorical(y_test_raw, len(class_labels))

    return (x_train, y_train), (x_test, y_test), class_labels


def build_cnn_model(num_classes: int = NUM_CLASSES):
    """Builds the 47-class Sequential CNN for handwritten character & digit recognition."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)),
            tf.keras.layers.Conv2D(
                filters=32,
                kernel_size=(3, 3),
                padding="same",
                activation="relu",
                kernel_initializer="he_normal",
                name="conv2d_1",
            ),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="max_pooling2d_1"),
            tf.keras.layers.Conv2D(
                filters=64,
                kernel_size=(3, 3),
                padding="same",
                activation="relu",
                kernel_initializer="he_normal",
                name="conv2d_2",
            ),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="max_pooling2d_2"),
            tf.keras.layers.Flatten(name="flatten"),
            tf.keras.layers.Dense(
                units=128,
                activation="relu",
                kernel_initializer="he_normal",
                name="dense_relu",
            ),
            tf.keras.layers.Dropout(rate=0.25, name="dropout"),
            tf.keras.layers.Dense(units=num_classes, activation="softmax", name="dense_softmax"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_and_export():
    print("=== Python Keras EMNIST Balanced CNN Training & Browser Export ===")
    (x_train, y_train), (x_test, y_test), class_labels = load_emnist_balanced()
    print(f"Classes ({len(class_labels)}): {' '.join(class_labels)}")
    print(f"Training samples: {x_train.shape[0]} | Test samples: {x_test.shape[0]}")

    model = build_cnn_model(len(class_labels))
    model.summary()

    epochs = 6
    batch_size = 256

    start_time = time.perf_counter()
    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        shuffle=True,
        verbose=1,
    )
    duration_sec = round(time.perf_counter() - start_time, 2)

    test_loss, test_acc = model.evaluate(x_test, y_test, batch_size=256, verbose=0)
    print(f"\nTest Loss: {test_loss:.4f} | Test Accuracy: {test_acc * 100:.2f}%")

    OUTPUT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    tfjs.converters.save_keras_model(model, str(OUTPUT_MODEL_DIR))
    print(f"Exported TensorFlow.js browser model to {OUTPUT_MODEL_DIR}")

    epoch_history = []
    for i in range(epochs):
        epoch_history.append(
            {
                "epoch": i + 1,
                "loss": round(float(history.history["loss"][i]), 4),
                "accuracy": round(float(history.history["accuracy"][i]) * 100, 2),
                "valLoss": round(float(history.history["val_loss"][i]), 4),
                "valAccuracy": round(float(history.history["val_accuracy"][i]) * 100, 2),
            }
        )

    metrics = {
        "dataset": "EMNIST Balanced",
        "trainedAt": datetime.now(timezone.utc).isoformat(),
        "trainSamples": int(x_train.shape[0]),
        "testSamples": int(x_test.shape[0]),
        "numClasses": len(class_labels),
        "classLabels": class_labels,
        "epochs": epochs,
        "batchSize": batch_size,
        "optimizer": "Adam (lr=0.001)",
        "lossFunction": "categorical_crossentropy",
        "trainDurationSec": duration_sec,
        "testLoss": round(float(test_loss), 4),
        "testAccuracy": round(float(test_acc) * 100, 2),
        "totalParams": int(model.count_params()),
        "history": epoch_history,
    }

    metrics_path = OUTPUT_MODEL_DIR / "training_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved training metrics to {metrics_path}")


if __name__ == "__main__":
    train_and_export()
