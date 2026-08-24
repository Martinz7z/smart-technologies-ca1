import tensorflow as tf
import numpy as np


CIFAR10_CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

REQUIRED_CIFAR10_CLASSES = {
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    7: "horse",
    9: "truck",
}


def filter_cifar10(images, labels):
    labels = labels.flatten()

    selected_labels = list(REQUIRED_CIFAR10_CLASSES.keys())
    mask = np.isin(labels, selected_labels)

    filtered_images = images[mask]
    filtered_labels = labels[mask]

    return filtered_images, filtered_labels


def print_class_counts(labels, dataset_name):
    print(f"\n{dataset_name} class counts:")

    for class_id, class_name in REQUIRED_CIFAR10_CLASSES.items():
        count = np.sum(labels == class_id)
        print(f"{class_name}: {count}")


def main():
    print("Smart Technologies CA1")
    print("TensorFlow version:", tf.__version__)

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    print("\nOriginal CIFAR-10")
    print("Training images:", x_train.shape)
    print("Testing images:", x_test.shape)

    x_train_filtered, y_train_filtered = filter_cifar10(x_train, y_train)
    x_test_filtered, y_test_filtered = filter_cifar10(x_test, y_test)

    print("\nFiltered CIFAR-10")
    print("Training images:", x_train_filtered.shape)
    print("Training labels:", y_train_filtered.shape)
    print("Testing images:", x_test_filtered.shape)
    print("Testing labels:", y_test_filtered.shape)

    print_class_counts(y_train_filtered, "Training")
    print_class_counts(y_test_filtered, "Testing")


if __name__ == "__main__":
    main()