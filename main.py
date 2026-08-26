import tensorflow as tf
import numpy as np


FINAL_CLASS_NAMES = [
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "horse",
    "truck",
    "cattle",
    "fox",
    "baby",
    "boy",
    "girl",
    "man",
    "woman",
    "rabbit",
    "squirrel",
    "trees",
    "bicycle",
    "bus",
    "motorcycle",
    "pickup_truck",
    "train",
    "lawn_mower",
    "tractor",
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


def prepare_cifar10(images, labels):
    labels = labels.flatten()

    selected_ids = list(REQUIRED_CIFAR10_CLASSES.keys())
    mask = np.isin(labels, selected_ids)

    filtered_images = images[mask]
    filtered_labels = labels[mask]

    remapped_labels = np.array([
        FINAL_CLASS_NAMES.index(REQUIRED_CIFAR10_CLASSES[label])
        for label in filtered_labels
    ])

    return filtered_images, remapped_labels


def prepare_cifar100(images, labels):
    labels = labels.flatten()

    required_ids = [
        CIFAR100_FINE_CLASS_NAMES.index(name)
        for name in REQUIRED_CIFAR100_CLASSES
    ]

    tree_ids = [
        CIFAR100_FINE_CLASS_NAMES.index(name)
        for name in TREE_CLASSES
    ]

    selected_ids = required_ids + tree_ids
    mask = np.isin(labels, selected_ids)

    filtered_images = images[mask]
    filtered_labels = labels[mask]

    remapped_labels = []

    for label in filtered_labels:
        class_name = CIFAR100_FINE_CLASS_NAMES[label]

        if class_name in TREE_CLASSES:
            final_class_name = "trees"
        else:
            final_class_name = class_name

        remapped_labels.append(
            FINAL_CLASS_NAMES.index(final_class_name)
        )

    return filtered_images, np.array(remapped_labels)


def print_final_class_counts(labels, dataset_name):
    print(f"\n{dataset_name} final class counts:")

    for class_id, class_name in enumerate(FINAL_CLASS_NAMES):
        count = np.sum(labels == class_id)
        print(f"{class_id:2d} - {class_name}: {count}")


def main():
    print("Smart Technologies CA1")
    print("TensorFlow version:", tf.__version__)

    # CIFAR-10
    (x_train_10, y_train_10), (x_test_10, y_test_10) = (
        tf.keras.datasets.cifar10.load_data()
    )

    x_train_10, y_train_10 = prepare_cifar10(
        x_train_10,
        y_train_10,
    )

    x_test_10, y_test_10 = prepare_cifar10(
        x_test_10,
        y_test_10,
    )

    # CIFAR-100
    (x_train_100, y_train_100), (x_test_100, y_test_100) = (
        tf.keras.datasets.cifar100.load_data(label_mode="fine")
    )

    x_train_100, y_train_100 = prepare_cifar100(
        x_train_100,
        y_train_100,
    )

    x_test_100, y_test_100 = prepare_cifar100(
        x_test_100,
        y_test_100,
    )

    # Combine datasets
    x_train = np.concatenate(
        [x_train_10, x_train_100],
        axis=0,
    )

    y_train = np.concatenate(
        [y_train_10, y_train_100],
        axis=0,
    )

    x_test = np.concatenate(
        [x_test_10, x_test_100],
        axis=0,
    )

    y_test = np.concatenate(
        [y_test_10, y_test_100],
        axis=0,
    )

    print("\nCombined dataset")
    print("Training images:", x_train.shape)
    print("Training labels:", y_train.shape)
    print("Testing images:", x_test.shape)
    print("Testing labels:", y_test.shape)

    print_final_class_counts(y_train, "Training")
    print_final_class_counts(y_test, "Testing")


if __name__ == "__main__":
    main()