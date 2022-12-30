import numpy as np
from PIL import Image
from pathlib import Path
from typing import List
from clipseg.models.clipseg import CLIPDensePredT
from PIL import Image
from torchvision import transforms
from matplotlib import pyplot as plt
import numpy as np
import cv2
import torch
from PIL import ImageOps

# load model
segmodel = CLIPDensePredT(version='ViT-B/16', reduce_dim=64)
segmodel.eval()

# non-strict, because we only stored decoder weights (not CLIP weights)
segmodel.load_state_dict(torch.load('../clipseg/weights/rd64-uni.pth', map_location=torch.device('cpu')), strict=False);


class Mask:
    __slots__ = ["center", "radius"]

    def __init__(self, center, radius):
        self.center = np.array(center)
        assert(self.center.shape == (2,))
        self.radius = float(radius)

    def __str__(self):
        return f"Mask: center {self.center[0]}, {self.center[1]}, radius {self.radius}"

class SegmentationMask:
    __slots__ = ["cls", "mask_val"]

    def __init__(self, cls:str, mask_val:bool):
        self.cls = cls
        self.mask_val = mask_val


def generate_mask_array(width, height, circles):
    width = float(width)
    height = float(height)

    x = np.arange(0.0, width, 1.0)
    y = np.arange(0.0, height, 1.0)
    yy, xx = np.meshgrid(x,y)
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
        center = np.array([height, width]) * center
        radius = height * circle.radius

        # Generate the mask.
        dist = np.linalg.norm(coords - center, axis=2)
        # mask as True/False
        mask = dist > radius
        # mask as 255/0 (255 means opaque, 0 means transparent)
        mask = (255 * mask).astype('uint8')

        result = np.minimum(result, mask)
    
    print(result.shape)

    im = Image.fromarray(result)
    im.show()

    return result

def run_class_segmentation(img_data, segmentation_class, width, height):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Resize((704, 704)),
        ])
    

    transform_grayscale = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Resize((352, 352)),
        transforms.Resize((704, 704)),
    ])

    img = transform(img_data).unsqueeze(0)

    img_data = ImageOps.grayscale(img_data)
    img_grayscale = transform_grayscale(img_data)
    print(img_grayscale.shape)
    Image.fromarray(np.asarray(img_grayscale.squeeze()).astype(np.int8))
    
    # predict
    with torch.no_grad():
        pred = segmodel(img.repeat(1,1,1,1), [segmentation_class.cls])[0][0]
        out_pred = []
        pred = np.asarray(np.squeeze(pred))
        print(pred.shape)
        kernel  = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(pred, kernel, iterations=1)
        if segmentation_class.mask_val:
            pred = (dilated < 0.5)
        else:
            pred = (dilated > 0.5)
        #pred_w_image = img_grayscale * pred
        pred_w_image = pred
        out_pred.append(pred_w_image)
        mask = Image.fromarray(np.asarray(pred_w_image * 255).astype(np.int8))
        #mask = Image.fromarray(np.asarray(pred_w_image.squeeze() * 255).astype(np.int8))
        #mask = Image.fromarray(np.asarray(pred) * 255)
        mask = mask.resize((width, height))
    
    # visualize prediction
    _, ax = plt.subplots(1, 5, figsize=(15, 4))
    [a.axis('off') for a in ax.flatten()]
    ax[0].imshow(img_data)
    #[ax[i+1].imshow(torch.sigmoid(out_preds[i])) for i in range(4)];
    ax[1].imshow(out_pred[0])
    
    mask.show()

    mask = np.asarray(mask)
    mask = (mask == 255)

    return mask


def generate_segmentation_array(rgb_array, segmentation_classes: List[SegmentationMask], width, height):
    masks = []
    for cls in segmentation_classes:
        res = run_class_segmentation(rgb_array, cls, width, height)
        masks.append(res)

    for mask in masks:
        # compile them all into one
        pass
    return masks[0]

def apply_segmentation_array(mask_array, segmentation_array):
    print(segmentation_array)
    new_arr = mask_array*segmentation_array
    return new_arr
    

def generate_mask_image(circles, segmentation_classes, infile):
    # Load the RGB data from infile
    rgb_PIL = Image.open(infile)
    width = rgb_PIL.width
    height = rgb_PIL.height
    rgb_array = np.array(rgb_PIL)
    if rgb_array.shape[2] == 4:
        rgb_array = rgb_array[:, :, 0:3]

    mask_array = generate_mask_array(width, height, circles)
    segmentation_array = generate_segmentation_array(rgb_PIL, segmentation_classes, width, height)
    mask_array = apply_segmentation_array(mask_array,segmentation_array)
    #mask_array = np.flip(np.swapaxes(mask_array, 0, 1), 0)
    mask_array = np.expand_dims(mask_array, axis=2)
    print(rgb_array.shape)
    print(mask_array.shape)
    image_data = np.concatenate((rgb_array, mask_array), axis=2)
    return Image.fromarray(image_data, 'RGBA')


def save_mask_image(circles, segmentation_classes, infile, outfile):
    image = generate_mask_image(circles, segmentation_classes, infile)
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

    # relative to width and height
    center = np.array([0.5, 0.5])

    # relative to height
    radius = 0.25

    circles = [
        Mask(center, radius),
        Mask(np.array([0.7, 0.8]), 0.2)
    ]

    print(circles[0])

    segmentation_classes = [SegmentationMask("a person", False)]

    save_mask_image(circles, segmentation_classes, "../seance/frames/IM0000.png", 'mask.png')
