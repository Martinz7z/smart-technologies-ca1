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