import numpy as np
from PIL import Image

def image_to_array(image):
    array = np.array(image)
    return np.swapaxes(np.flip(array, 0), 0, 1)

def array_to_image(array):
    array = np.flip(np.swapaxes(array, 0, 1), 0)
    format = 'RGB' if array.shape[2] == 3 else 'RGBA'
    return Image.fromarray(array, format)

if __name__ == "__main__":
    # Any image will do.
    image = Image.open("niku_1.jpeg")

    rgb_array = image_to_array(image)
    print(rgb_array.shape)
    print(rgb_array)

    image = array_to_image(rgb_array)
    image.save("niku_modified.png")
