import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngImageFile as PILformat1


def generate_black_rectangle_array(width, height):
    return np.zeros((width, height, 3), dtype='uint8')


def generate_mask_array(width, height, center, radius):
    x = np.arange(0.0, float(width), 1.0)
    y = np.arange(0.0, float(height), 1.0)
    yy, xx = np.meshgrid(y, x)
    # coords[i, j, :] = [i, j]
    coords = np.stack([xx, yy], axis=2)
    #for i in range(5):
    #    for j in range(5):
    #        print(f"{i}, {j}: {coords[i, j]}")
    # dist[i, j] = distance from (i, j) to center
    dist = np.linalg.norm(coords - center, axis=2)
    # mask as True/False
    mask = dist > radius
    # mask as 255/0
    return (255 * mask).astype('uint8')


def array_to_image(array):
    array = np.flip(np.swapaxes(array, 0, 1), 0)
    format = 'RGB' if array.shape[2] == 3 else 'RGBA'
    return Image.fromarray(array, format)


if __name__ == "__main__":
    width = 10
    height = 5
    center = np.array([3.0, 2.0])
    radius = 2.0

    pixels = generate_black_rectangle_array(width, height)
    pixels[0, 0, :] = [255, 0, 0]
    pixels[3, 0, :] = [0, 255, 0]
    pixels[0, 3, :] = [0, 0, 255]
    print(pixels.shape)
    print(pixels)

    image = array_to_image(pixels)
    image.save('test_rgb.png')

    mask = generate_mask_array(width, height, center, radius)
    mask = np.expand_dims(mask, axis=2)
    print(mask.shape)
    print(mask)

    rgba_array = np.concatenate([pixels, mask], axis=2)
    print(rgba_array)

    image = array_to_image(rgba_array)
    image.save('test_rgba.png')
