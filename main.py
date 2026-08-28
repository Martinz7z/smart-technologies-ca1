import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
)

from cnn_model import (
    create_baseline_model,
    create_augmented_model,
)

from preprocessing import (
    convert_to_grayscale,
    apply_gaussian_blur,
    equalize_histogram,
    normalize_image,
    resize_image,
)


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


def show_class_examples(images, labels):
    fig, axes = plt.subplots(4, 6, figsize=(12, 8))

    for class_id, ax in enumerate(axes.flat):
        image_index = np.where(labels == class_id)[0][0]

        ax.imshow(images[image_index])
        ax.set_title(FINAL_CLASS_NAMES[class_id], fontsize=9)
        ax.axis("off")

    plt.suptitle("Example Image from Each Class")
    plt.tight_layout()
    plt.savefig("class_examples.png")
    plt.show()


def show_class_distribution(labels):
    class_counts = [
        np.sum(labels == class_id)
        for class_id in range(len(FINAL_CLASS_NAMES))
    ]

    plt.figure(figsize=(14, 6))
    plt.bar(FINAL_CLASS_NAMES, class_counts)

    plt.title("Training Image Distribution")
    plt.xlabel("Class")
    plt.ylabel("Number of Images")
    plt.xticks(rotation=60, ha="right")

    plt.tight_layout()
    plt.savefig("class_distribution.png")
    plt.show()


def show_preprocessing_examples(image):
    grayscale = convert_to_grayscale(image)
    blurred = apply_gaussian_blur(image)
    equalized = equalize_histogram(image)
    normalized = normalize_image(image)
    resized = resize_image(image)

    fig, axes = plt.subplots(2, 3, figsize=(10, 7))

    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Original")

    axes[0, 1].imshow(grayscale, cmap="gray")
    axes[0, 1].set_title("Grayscale")

    axes[0, 2].imshow(blurred)
    axes[0, 2].set_title("Gaussian Blur")

    axes[1, 0].imshow(equalized, cmap="gray")
    axes[1, 0].set_title("Histogram Equalised")

    axes[1, 1].imshow(normalized)
    axes[1, 1].set_title("Normalised")

    axes[1, 2].imshow(resized)
    axes[1, 2].set_title("Resized 64x64")

    for ax in axes.flat:
        ax.axis("off")

    plt.suptitle("Image Preprocessing Examples")
    plt.tight_layout()
    plt.savefig("preprocessing_examples.png")
    plt.show()


def train_augmented_model(x_train, y_train, x_test, y_test):
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    model = create_augmented_model()

    print("\nAugmented CNN - Experiment 4")
    model.summary()

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True,
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=0.00001,
        verbose=1,
    )

    history = model.fit(
        x_train,
        y_train,
        validation_split=0.1,
        epochs=30,
        batch_size=64,
        shuffle=True,
        callbacks=[
            early_stopping,
            reduce_lr,
        ],
    )

    test_loss, test_accuracy = model.evaluate(
        x_test,
        y_test,
        verbose=0,
    )

    print("\nExperiment 4 results")
    print("Epochs completed:", len(history.history["loss"]))
    print("Test loss:", test_loss)
    print("Test accuracy:", test_accuracy)

    return model, history


def plot_experiment_four_history(history):
    plt.figure(figsize=(8, 5))

    plt.plot(
        history.history["accuracy"],
        label="Training Accuracy",
    )

    plt.plot(
        history.history["val_accuracy"],
        label="Validation Accuracy",
    )

    plt.title("Experiment 4 Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig("experiment4_accuracy.png")
    plt.show()

    plt.figure(figsize=(8, 5))

    plt.plot(
        history.history["loss"],
        label="Training Loss",
    )

    plt.plot(
        history.history["val_loss"],
        label="Validation Loss",
    )

    plt.title("Experiment 4 Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig("experiment4_loss.png")
    plt.show()


def main():
    print("Smart Technologies CA1")
    print("TensorFlow version:", tf.__version__)

    # Load CIFAR-10
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

    # Load CIFAR-100
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

    # Shuffle training data before validation split
    rng = np.random.default_rng(42)
    indices = rng.permutation(len(x_train))

    x_train = x_train[indices]
    y_train = y_train[indices]

    print("\nCombined dataset")
    print("Training images:", x_train.shape)
    print("Training labels:", y_train.shape)
    print("Testing images:", x_test.shape)
    print("Testing labels:", y_test.shape)

    print_final_class_counts(y_train, "Training")
    print_final_class_counts(y_test, "Testing")

    # Exploration plots already created
    # show_class_examples(x_train, y_train)
    # show_class_distribution(y_train)
    # show_preprocessing_examples(x_train[0])

    model, history = train_augmented_model(
        x_train,
        y_train,
        x_test,
        y_test,
    )

    plot_experiment_four_history(history)


if __name__ == "__main__":
    main()