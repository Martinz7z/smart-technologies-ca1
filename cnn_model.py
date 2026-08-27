import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
)


def create_baseline_model():
    model = Sequential([
        Input(shape=(32, 32, 3)),

        Conv2D(
            32,
            (3, 3),
            activation="relu",
        ),

        MaxPooling2D(
            pool_size=(2, 2),
        ),

        Conv2D(
            64,
            (3, 3),
            activation="relu",
        ),

        MaxPooling2D(
            pool_size=(2, 2),
        ),

        Flatten(),

        Dense(
            128,
            activation="relu",
        ),

        Dropout(0.5),

        Dense(
            24,
            activation="softmax",
        ),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def create_augmented_model():
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.1),
    ])

    model = Sequential([
        Input(shape=(32, 32, 3)),

        data_augmentation,

        Conv2D(
            32,
            (3, 3),
            padding="same",
            activation="relu",
        ),

        MaxPooling2D(
            pool_size=(2, 2),
        ),

        Conv2D(
            64,
            (3, 3),
            padding="same",
            activation="relu",
        ),

        MaxPooling2D(
            pool_size=(2, 2),
        ),

        Conv2D(
            128,
            (3, 3),
            padding="same",
            activation="relu",
        ),

        MaxPooling2D(
            pool_size=(2, 2),
        ),

        Flatten(),

        Dense(
            128,
            activation="relu",
        ),

        Dropout(0.5),

        Dense(
            24,
            activation="softmax",
        ),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model