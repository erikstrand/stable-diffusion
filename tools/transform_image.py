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

def transform(image_array):
    angle = 45.0
    zoom = 1.0
    #translation_x = keys.translation_x_series[frame_idx]
    #translation_y = keys.translation_y_series[frame_idx]

    center = (0.5 * (image_array.shape[1] - 1), 0.5 * (image_array.shape[0] - 1))

    #trans_mat = np.float32([[1, 0, translation_x], [0, 1, translation_y]])
    rot_mat = cv.getRotationMatrix2D(center, angle, zoom)
    #trans_mat = np.vstack([trans_mat, [0,0,1]])
    #rot_mat = np.vstack([rot_mat, [0,0,1]])
    #xform = np.matmul(rot_mat, trans_mat)

    return cv.warpAffine(image_array, rot_mat, (image_array.shape[1], image_array.shape[0]))
    #return cv.warpPerspective(
    #    image_array,
    #    rot_mat,
    #    (image_array.shape[1], image_array.shape[0]),
    #    borderMode=cv.BORDER_REPLICATE
    #)

if __name__ == "__main__":
    img = cv.imread("niku_1.jpeg")
    print(type(img))
    print(img.shape)
    print(img.dtype)
    print(img)

    rows, cols = img.shape[0:2]
    # cols-1 and rows-1 are the coordinate limits.
    M = cv.getRotationMatrix2D(((cols-1)/2.0, (rows-1)/2.0), 45.0, 1)
    dst = cv.warpAffine(img,M,(cols,rows))
    cv.imwrite("niku_1_cv.png", dst)


    # Any image will do.
    image = Image.open("niku_1.jpeg")

    rgb_array = image_to_array(image)
    print(rgb_array.shape)
    print(rgb_array.dtype)
    print(rgb_array)

    rgb_array = transform(rgb_array)
    print(rgb_array.shape)
    print(rgb_array.dtype)
    print(rgb_array)

    image = array_to_image(rgb_array)
    image.save("niku_modified.png")
