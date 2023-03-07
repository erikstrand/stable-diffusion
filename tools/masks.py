import numpy as np
from PIL import Image
from pathlib import Path


class Mask:
    __slots__ = ["center", "radius", "sigmoid_k", "invert"]

    def __init__(self, center, radius, sigmoid_k=0.2, invert=False):
        self.center = np.array(center)
        assert(self.center.shape == (2,))
        self.radius = float(radius)
        self.sigmoid_k = float(sigmoid_k)
        self.invert = bool(invert)

    @classmethod
    def from_dict(cls, dict):
        invert = False
        if "invert" in dict:
            if dict["invert"] == "true":
                invert = True
            else:
                assert(dict["invert"] == "false")
        return cls(dict["center"], dict["radius"], float(dict["sigmoid_k"]), bool(dict["invert"]))

    def __str__(self):
        result = f"Mask: center {self.center[0]}, {self.center[1]}, radius {self.radius}, sigmoid_k {self.sigmoid_k}"
        if self.invert:
            result += ", invert"
        return result


def generate_mask_array(width, height, masks):
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
    inverted = False

    for mask in masks:
        # Rescale center and radius.
        center = mask.center
        center = np.array([width, height]) * center
        radius = height * mask.radius

        # Generate the mask.
        dist = np.linalg.norm(coords - center, axis=2)
        # sigmoid
        mask_array = 255.0 / (1.0 + np.exp(-mask.sigmoid_k * (dist - radius)))
        # 255 means opaque, 0 means transparent
        mask_array = mask_array.astype('uint8')

        if mask.invert:
            #mask_array = 255 - mask_array
            inverted = True

        # Merge with previous masks.
        result = np.minimum(result, mask_array)
        #result = np.maximum(result, mask_array)

    if inverted:
        result = 255 - result

    return result


def generate_mask_image(width, height, masks, infile):
    # Load the RGB data from infile
    rgb_PIL = Image.open(infile)
    rgb_array = np.array(rgb_PIL)
    if rgb_array.shape[2] == 4:
        rgb_array = rgb_array[:, :, 0:3]

    mask_array = generate_mask_array(width, height, masks)
    mask_array = np.flip(np.swapaxes(mask_array, 0, 1), 0)
    mask_array = np.expand_dims(mask_array, axis=2)
    image_data = np.concatenate((rgb_array, mask_array), axis=2)
    return Image.fromarray(image_data, 'RGBA')


def save_mask_image(width, height, masks, infile, outfile):
    image = generate_mask_image(width, height, masks, infile)
    Path(outfile).parent.mkdir(parents=True, exist_ok=True)
    image.save(outfile)


# not used now but maybe useful in the future
def generate_black_rectangle_array(width, height):
    return np.zeros((width, height, 3), dtype='uint8')


# not used now but maybe useful in the future
def image_to_array(image):
    # convert the PIL image to a numpy array
    image_array = np.array(image)
    return np.swapaxes(np.flip(image_array, 0), 0, 1)


# not used now but maybe useful in the future
def array_to_image(array):
    array = np.flip(np.swapaxes(array, 0, 1), 0)
    format = 'RGB' if array.shape[2] == 3 else 'RGBA'
    return Image.fromarray(array, format)


if __name__ == "__main__":
    # in pixels
    width = 960
    height = 720

    masks = [
        Mask([0.92, 0.4], 0.25),
        Mask([0.92, 0.5], 0.25),
        Mask([0.92, 0.6], 0.25),
    ]

    image_file = "../alice/frames/rabbit_hole/IM0138.png"

    original = Image.open(image_file)
    original.crop((650, 30, 960, 690)).save("frame.png")

    image = generate_mask_image(width, height, masks, image_file)
    image.save('mask.png')
    image.crop((650, 30, 960, 690)).save('mask_crop.png')
