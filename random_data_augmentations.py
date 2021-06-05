from scipy import ndimage
import cv2
import numpy as np
import random
import os


# Loads all images from a directory
def load_images(images_dir):
    return [cv2.imread(os.path.join(images_dir, filename)) for filename in os.listdir(images_dir)]


# Flips the image horizontally with a probability.
def random_horizontal_flip(image, flip_prob):
    p = random.uniform(0, 1.0)

    if p < flip_prob:
        return image[:, ::-1]
    else:
        return image


# Crops a random region of the image.
def random_crop(image, scale):
    height, width = image.shape[0:2]

    x_min = int(width * scale)
    y_min = int(height * scale)

    x = random.randint(0, width - x_min)
    y = random.randint(0, height - y_min)

    cropped_image = image[y: y+y_min, x: x+x_min]
    return cv2.resize(cropped_image, (width, height))


# Randomly pads the image.
def random_padding(image, padding_range):
    height, width = image.shape[0:2]

    padding_pixels = random.randint(0, padding_range)
    padded_image = cv2.copyMakeBorder(image, padding_pixels, padding_pixels, padding_pixels, padding_pixels, cv2.BORDER_CONSTANT)
    return cv2.resize(padded_image, (width, height))


# Randomly adjusts the brightness to the image.
def random_brightness(image, min_range, max_range):
    brightness = random.randint(min_range, max_range)

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)

    if brightness >= 0:
        lim = 255 - brightness
        v[v > lim] = 255
        v[v <= lim] += brightness
    else:
        brightness = abs(brightness)
        lim = brightness
        v[v < lim] = 0
        v[v >= lim] -= brightness

    brightened_image = cv2.merge((h, s, v))
    return cv2.cvtColor(brightened_image, cv2.COLOR_HSV2RGB)


# Randomly adjusts the contrast to the image.
def random_contrast(image, min_range, max_range):
    contrast = random.randint(min_range, max_range)

    temp_img = np.int16(image)
    temp_img = temp_img * (contrast/127+1) - contrast
    temp_img = np.clip(temp_img, 0, 255)
    return np.uint8(temp_img)


# Randomly adjusts the saturation to the image.
def random_saturation(image, min_range, max_range):
    saturation = random.randint(min_range, max_range)

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)

    if saturation >= 0:
        lim = 255 - saturation
        v[v > lim] = 255
        v[v <= lim] += saturation
    else:
        saturation = abs(saturation)
        lim = saturation
        v[v < lim] = 0
        v[v >= lim] -= saturation

    saturated_image = cv2.merge((h, s, v))
    return cv2.cvtColor(saturated_image, cv2.COLOR_HSV2RGB)


# Randomly rotates the image.
def random_rotate(image, min_angle, max_angle):
    angle = random.randint(min_angle, max_angle)
    return ndimage.rotate(image, angle)


# Constructs a data augmentation pipeline.
def sequential(image, augmentation_prob):
    if random.uniform(0, 1.0) < augmentation_prob:
        image = random_horizontal_flip(image, flip_prob=0.5)
    if random.uniform(0, 1.0) < augmentation_prob:
        image = random_crop(image, scale=0.9)
    if random.uniform(0, 1.0) < augmentation_prob:
        image = random_padding(image, padding_range=20)
    if random.uniform(0, 1.0) < augmentation_prob:
        image = random_brightness(image, -20, 40)
    if random.uniform(0, 1.0) < augmentation_prob:
        image = random_saturation(image, -20, 40)
    if random.uniform(0, 1.0) < augmentation_prob:
        image = random_contrast(image, 0, 40)
    if random.uniform(0, 1.0) < augmentation_prob:
        image = random_rotate(image, -10, 10)

    return image


# Apply data augmentation pipeline.
def data_augmentations(images, num_of_augmentations, save_dir):
    counter = 0

    for image in images:
        for i in range(num_of_augmentations):
            random_image = sequential(image, augmentation_prob=0.8)
            cv2.imwrite(os.path.join(save_dir, str(counter)+'.jpg'), random_image)
            counter += 1


def main():
    images = load_images('images')
    data_augmentations(images, 10, 'augmentations')


main()
