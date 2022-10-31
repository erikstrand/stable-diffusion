import numpy as np
from PIL import Image
from pathlib import Path


class Mask:
    __slots__ = ["center", "radius"]

    def __init__(self, center, radius):
        self.center = np.array(center)
        assert(self.center.shape == (2,))
        self.radius = float(radius)

    def __str__(self):
        return f"Mask: center {self.center[0]}, {self.center[1]}, radius {self.radius}"


def generate_mask_array(width, height, circles):
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

    result = np.full(coords.shape[0:2], 255, dtype='uint8')

    for circle in circles:
        # Rescale center and radius.
        center = circle.center
        center = np.array([width, height]) * center
        radius = height * circle.radius

        # Generate the mask.
        dist = np.linalg.norm(coords - center, axis=2)
        # mask as True/False
        mask = dist > radius
        # mask as 255/0 (255 means opaque, 0 means transparent)
        mask = (255 * mask).astype('uint8')

        result = np.minimum(result, mask)

    return result


def generate_mask_image(width, height, circles, infile):
    # Load the RGB data from infile
    rgb_PIL = Image.open(infile)
    rgb_array = np.array(rgb_PIL)
    if rgb_array.shape[2] == 4:
        rgb_array = rgb_array[:, :, 0:3]

    mask_array = generate_mask_array(width, height, circles)
    mask_array = np.flip(np.swapaxes(mask_array, 0, 1), 0)
    mask_array = np.expand_dims(mask_array, axis=2)
    image_data = np.concatenate((rgb_array, mask_array), axis=2)
    return Image.fromarray(image_data, 'RGBA')


def save_mask_image(width, height, circles, infile, outfile):
    image = generate_mask_image(width, height, circles, infile)
    Path(outfile).parent.mkdir(parents=True, exist_ok=True)
    image.save(outfile)


# not used now but maybe useful in the future
def generate_black_rectangle_array(width, height):
    return np.zeros((width, height, 3), dtype='uint8')


# not used now but maybe useful in the future
def array_to_image(array):
    array = np.flip(np.swapaxes(array, 0, 1), 0)
    format = 'RGB' if array.shape[2] == 3 else 'RGBA'
    return Image.fromarray(array, format)


if __name__ == "__main__":
    # in pixels
    width = 640
    height = 360

    # relative to width and height
    center = np.array([0.5, 0.5])

    # relative to height
    radius = 0.25

    circles = [
        (center, radius),
        (np.array([0.7, 0.8]), 0.2)
    ]

    save_mask_image(width, height, circles, "../99fmermaid_1/frames_1/IM00200.jpg", 'mask.png')
