import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngImageFile as PILformat1


def generate_black_rectangle_array(width, height):
    return np.zeros((width, height, 3), dtype='uint8')


def generate_mask_array(width, height, center, radius):
    width = float(width)
    height = float(height)

    x = np.arange(0.0, width, 1.0)
    y = np.arange(0.0, height, 1.0)
    yy, xx = np.meshgrid(y, x)
    # coords[i, j, :] = [i, j]
    coords = np.stack([xx, yy], axis=2)
    #for i in range(5):
    #    for j in range(5):
    #        print(f"{i}, {j}: {coords[i, j]}")
    # dist[i, j] = distance from (i, j) to center

    # Rescale center and radius
    center = np.array([width, height]) * center
    radius = height * radius

    dist = np.linalg.norm(coords - center, axis=2)
    # mask as True/False
    mask = dist > radius
    # mask as 255/0
    return (255 * mask).astype('uint8')


def array_to_image(array):
    array = np.flip(np.swapaxes(array, 0, 1), 0)
    format = 'RGB' if array.shape[2] == 3 else 'RGBA'
    return Image.fromarray(array, format)


def generate_mask_image(width, height, center, radius):
    rgb_array = generate_black_rectangle_array(width, height)
    mask_array = generate_mask_array(width, height, center, radius)
    mask_array = np.expand_dims(mask_array, axis=2)
    image_data = np.concatenate((rgb_array, mask_array), axis=2)
    return array_to_image(image_data)


def save_mask_image(width, height, center, radius, filename):
    image = generate_mask_image(width, height, center, radius)
    image.save(filename)


if __name__ == "__main__":
    width = 640
    height = 320
    # relative to width and height
    center = np.array([0.5, 0.5])
    # relative to height
    radius = 0.25

    save_mask_image(width, height, center, radius, 'mask.png')
