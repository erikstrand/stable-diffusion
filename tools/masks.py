import numpy as np
from PIL import Image
from pathlib import Path
import os 


class Mask:
    __slots__ = ["center", "radius", "sigmoid_k", "invert", "refobject"]

    def __init__(self, center, radius, sigmoid_k=0.2, invert=False, refobject=None):
        self.center = np.array(center)
        assert(self.center.shape == (2,))
        self.radius = float(radius)
        self.sigmoid_k = float(sigmoid_k)
        self.invert = bool(invert)
        self.refobject = refobject 
        
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
            mask_array = 255 - mask_array

        # Merge with previous masks.
        result = np.minimum(result, mask_array)

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


def generate_image_with_object_inside_masked_region(width, height, masks, infile):
    # Load the RGB data from infile
    image = Image.open(infile)
    image = image.convert('RGBA') # need to convert to RGBA to prevent type error on object paste  

    for mask in masks:
        if mask.refobject is not None:
            # Grab path 
            objectfile = mask.refobject
            assert os.path.exists(objectfile),f"Refobject does not exist. Check path? {objectfile}"
            
            # Load the object 
            if objectfile.endswith('jpg') or objectfile.endswith('jpeg'):
                base = os.path.splitext(objectfile)[0]
                if not os.path.exists(base+".png"):
                    obj = Image.open(objectfile)
                    objectfile = base+".png"
                    obj.save(objectfile)
                else: 
                    objectfile = base+".png"
            
            # open ref object 
            obj = Image.open(objectfile)
            w,h = obj.size
                    
            # # resize obj to the radius of the mask (such that it fully fits in the mask) 
            assert not mask.invert, f"Cannot use inverted masks when pasting objects"
            d = mask.radius*2        
            d_im = int(height * d)        # diameter w.r.t to the image 
            #nh = int(d_im)                # use this if you the square height equals diameter  
            nh = int(np.sqrt(d_im**2/2))        # use this if you want to fit square enterily into the circle 
            nw = int((w/h)*nh)
            objr = obj.resize((nw,nh))
                    
            # crop to square 
            left = (nw - nh)/2
            top = 0
            right = (nw + nh)/2
            bottom = nh
            objrc = objr.crop((left,top,right,bottom))
            
            # paste into the image 
            wc,hc = list(mask.center)   # get center coordinates of the mask
            hc = 1-hc # when pasting over images with PIL, the coord system is from top-left corner (not bottom left, like our masks are defined)
            wc,hc, = int(width*wc), int(height*hc) # define in the image coord system 
            wc,hc = wc-int(nh/2),hc-int(nh/2) # when pasting over images with PIL, the count is from the edge of the object, not the center 
                
            objrc2 = objrc.convert('RGBA')
            image.paste(objrc2, (wc,hc))
            
    out = image.convert('RGB')
    return out

def save_mask_image(width, height, masks, infile, outfile):
    image = generate_mask_image(width, height, masks, infile)
    Path(outfile).parent.mkdir(parents=True, exist_ok=True)
    image.save(outfile)

def save_image_with_object_in_masked_region(width, height, masks, infile, outfile):
    image = generate_image_with_object_inside_masked_region(width, height, masks, infile)
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
    
    debug = "serge"
    
    if debug == "erik":
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
    elif debug == "serge":
        
        width = 960
        height = 576 
        
        # get image and masks
        object_file = "forest.jpeg"
        image_file = "seance/frames_r/seance7/IM0001.png"
        image = Image.open(image_file)     
        masks = [MagicMask([0.2, 0.2], radius = 0.4)] # lower left 
        
        
        # check file type 
        if object_file.endswith('jpg') or object_file.endswith('jpeg'):
            base = os.path.splitext(object_file)[0]
            if not os.path.exists(base+".png"):
                obj = Image.open(object_file)
                object_file = base+".png"
                obj.save(object_file)
            else: 
                object_file = base+".png"

        # open ref object 
        obj = Image.open(object_file)
        w,h = obj.size
        
        # # resize obj to the radius of the mask (such that it fully fits in the mask) 
        # r = masks[0].radius*2
        # r_im = int(height * r)        # diameter w.r.t to the image 
        # nw = int(r_im*2)              # 
        # nh = int((h/w)*nw)
        # objr = obj.resize((nw,nh))

        # # resize obj to the radius of the mask (such that it fully fits in the mask) 
        d = masks[0].radius*2
        d_im = int(height * d)        # diameter w.r.t to the image 
        #nh = int(d_im)                # use this if you the square height equals diameter  
        nh = int(np.sqrt(d_im**2/2))        # use this if you want to fit square enterily into the circle 
        nw = int((w/h)*nh)
        objr = obj.resize((nw,nh))
                
    
        # crop to square 
        left = (nw - nh)/2
        top = 0
        right = (nw + nh)/2
        bottom = nh
        objrc = objr.crop((left,top,right,bottom))
        
        # test saves 
        image_masked = generate_mask_image(width, height, masks, image_file)
        image_masked.save('test_with_mask.png')
        image.save("test.png")              
        objrc.save("test_forest_cropped.png")
                
        # paste into the image 
        wc,hc = list(masks[0].center)   # get center coordinates of the mask
        hc = 1-hc # when pasting over images with PIL, the coord system is from top-left corner (not bottom left, like our masks are defined)
        wc,hc, = int(width*wc), int(height*hc) # define in the image coord system 
        wc,hc = wc-int(nh/2),hc-int(nh/2) # when pasting over images with PIL, the count is from the edge of the object, not the center 
        
        image2 = image.convert('RGBA')
        image2.save("test_original.png")
        objrc2 = objrc.convert('RGBA')
        
        image2.paste(objrc2, (wc,hc))
        image2.save("test_with_forrest.png")
        
        

