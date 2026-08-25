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


CIFAR100_FINE_CLASS_NAMES = [
    "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle",
    "bicycle", "bottle", "bowl", "boy", "bridge", "bus", "butterfly", "camel",
    "can", "castle", "caterpillar", "cattle", "chair", "chimpanzee", "clock",
    "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur",
    "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster",
    "house", "kangaroo", "keyboard", "lamp", "lawn_mower", "leopard", "lion",
    "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain", "mouse",
    "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree", "pear",
    "pickup_truck", "pine_tree", "plain", "plate", "poppy", "porcupine",
    "possum", "rabbit", "raccoon", "ray", "road", "rocket", "rose",
    "sea", "seal", "shark", "shrew", "skunk", "skyscraper", "snail", "snake",
    "spider", "squirrel", "streetcar", "sunflower", "sweet_pepper", "table",
    "tank", "telephone", "television", "tiger", "tractor", "train", "trout",
    "tulip", "turtle", "wardrobe", "whale", "willow_tree", "wolf", "woman",
    "worm",
]

REQUIRED_CIFAR100_CLASSES = [
    "cattle",
    "fox",
    "baby",
    "boy",
    "girl",
    "man",
    "woman",
    "rabbit",
    "squirrel",
    "bicycle",
    "bus",
    "motorcycle",
    "pickup_truck",
    "train",
    "lawn_mower",
    "tractor",
]

TREE_CLASSES = [
    "maple_tree",
    "oak_tree",
    "palm_tree",
    "pine_tree",
    "willow_tree",
]


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


def get_cifar100_class_ids(class_names):
    return [
        CIFAR100_FINE_CLASS_NAMES.index(class_name)
        for class_name in class_names
    ]


def filter_cifar100(images, labels):
    labels = labels.flatten()

    required_ids = get_cifar100_class_ids(REQUIRED_CIFAR100_CLASSES)
    tree_ids = get_cifar100_class_ids(TREE_CLASSES)

    selected_ids = required_ids + tree_ids
    mask = np.isin(labels, selected_ids)

    filtered_images = images[mask]
    filtered_labels = labels[mask]

    return filtered_images, filtered_labels


def print_cifar100_counts(labels, dataset_name):
    print(f"\n{dataset_name} CIFAR-100 class counts:")

    for class_name in REQUIRED_CIFAR100_CLASSES:
        class_id = CIFAR100_FINE_CLASS_NAMES.index(class_name)
        count = np.sum(labels == class_id)
        print(f"{class_name}: {count}")

    tree_count = 0

    for class_name in TREE_CLASSES:
        class_id = CIFAR100_FINE_CLASS_NAMES.index(class_name)
        count = np.sum(labels == class_id)
        tree_count += count
        print(f"{class_name}: {count}")

    print(f"trees combined: {tree_count}")


def main():
    print("Smart Technologies CA1")
    print("TensorFlow version:", tf.__version__)

    # Load CIFAR-10
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

    # Load CIFAR-100
    (x_train_100, y_train_100), (x_test_100, y_test_100) = (
        tf.keras.datasets.cifar100.load_data(label_mode="fine")
    )

    x_train_100_filtered, y_train_100_filtered = filter_cifar100(
        x_train_100,
        y_train_100,
    )

    x_test_100_filtered, y_test_100_filtered = filter_cifar100(
        x_test_100,
        y_test_100,
    )

    print("\nFiltered CIFAR-100")
    print("Training images:", x_train_100_filtered.shape)
    print("Training labels:", y_train_100_filtered.shape)
    print("Testing images:", x_test_100_filtered.shape)
    print("Testing labels:", y_test_100_filtered.shape)

    print_cifar100_counts(y_train_100_filtered, "Training")
    print_cifar100_counts(y_test_100_filtered, "Testing")


if __name__ == "__main__":
    main()