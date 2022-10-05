import numpy as np
import cv2 as cv
from PIL import Image


def image_to_array(image):
    array = np.array(image)
    return np.swapaxes(np.flip(array, 0), 0, 1)


def array_to_image(array):
    array = np.flip(np.swapaxes(array, 0, 1), 0)
    format = 'RGB' if array.shape[2] == 3 else 'RGBA'
    return Image.fromarray(array, format)


def transform_image_array(image_array, zoom=1.0, angle=0.0, translation=(0.0, 0.0)):
    center = (0.5 * (image_array.shape[1] - 1), 0.5 * (image_array.shape[0] - 1))
    trans_mat = np.float32([
        [1.0, 0.0, translation[1]],
        [0.0, 1.0, translation[0]],
        [0.0, 0.0, 1.0]
    ])
    rot_mat = cv.getRotationMatrix2D(center, angle, zoom)
    rot_mat = np.vstack([rot_mat, [0.0, 0.0, 1.0]])
    xform = np.matmul(rot_mat, trans_mat)
    return cv.warpPerspective(
        image_array,
        xform,
        (image_array.shape[1], image_array.shape[0]),
        borderMode=cv.BORDER_REPLICATE
    )


def transform_image_file(infile, zoom=1.0, angle=0.0, translation=(0.0, 0.0)):
    image = Image.open(infile)
    image_array = image_to_array(image)
    return transform_image_array(image_array, zoom, angle, translation)
    #image = array_to_image(image_array)
    #image.save(outfile)


if __name__ == "__main__":
    # Any image will do.
    image = Image.open("niku_1.jpeg")
    array = image_to_array(image)
    array = transform(array, 1.2, -10.0, (15.0, 0.0))
    image = array_to_image(array)
    image.save("niku_modified.png")
