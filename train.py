import argparse
import os
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

LABELS = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")

def prepare(example):
    image = tf.cast(example["image"], tf.float32) / 255.0
    image = tf.image.resize(image, (28, 28))
    image = tf.reshape(image, (28, 28, 1))
    label = tf.cast(example["label"], tf.int64)
    return image, label

def main():
    parser = argparse.ArgumentParser(description="Train an EMNIST ByClass character recognition CNN.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--dataset", default="emnist/byclass")
    args = parser.parse_args()

    # EMNIST ByClass contains 62 classes: 10 digits, 26 uppercase and 26 lowercase letters.
    (train_ds, test_ds), info = tfds.load(
        args.dataset, split=["train", "test"], as_supervised=False,
        with_info=True, data_dir="data"
    )
    train_ds = train_ds.map(prepare, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.map(prepare, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.shuffle(20000).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(32, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(62, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()
    model.fit(train_ds, validation_data=test_ds, epochs=args.epochs)
    os.makedirs("models", exist_ok=True)
    model.save("models/emnist_model.keras")
    print("Saved model to models/emnist_model.keras")

if __name__ == "__main__":
    main()
