import cv2
import numpy as np


def convert_to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def apply_gaussian_blur(image):
    return cv2.GaussianBlur(image, (3, 3), 0)


def equalize_histogram(image):
    grayscale = convert_to_grayscale(image)
    return cv2.equalizeHist(grayscale)


def normalize_image(image):
    return image.astype("float32") / 255.0


def resize_image(image, width=64, height=64):
    return cv2.resize(
        image,
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )