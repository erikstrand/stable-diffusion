import os 
import sys 

import numpy as np 
from PIL import Image
from PIL.PngImagePlugin import PngImageFile as PILformat1
from PIL.JpegImagePlugin import JpegImageFile as PILformat2



def get_alpha_channel(image):
    assert isinstance(image,PILformat1) or isinstance(image,PILformat2), f"Input must be PIL format"
    im = np.array(image)
    assert im.ndim==3, f"Image must have 4 channels"
    alpha=im[:,:,-1]
    
    # invert it 
    mask = np.zeros_like(alpha)
    mask[alpha==0] = 1 
    return mask 

def mask_frame(image, mask):    
    # masks a standard image (with 3 channels, not 4)
    
    # some checks 
    assert isinstance(image,PILformat1) or isinstance(image,PILformat2), f"Input must be PIL format"
    assert isinstance(mask, np.ndarray), f"Mask must be numpy format"
    assert mask.ndim == 2 
    x1,y1=mask.shape[0:2]
    x2,y2=image.size
    assert x1==y2 and y1==x2
        
    # mask image 
    im = np.array(image)
    assert im.ndim==3, f"PIL images must have only 3 channels. Currently shape is: {im.shape}"
    mask_e=np.moveaxis(np.tile(mask,(3,1,1)), 0,-1)
    im[mask_e==1]=0
    
    #create 4th channel 
    alpha_mask = create_alpha_channel(mask)
    alpha_mask_e=np.expand_dims(alpha_mask,-1)
    
    # add together 
    final = np.concatenate((im,alpha_mask_e),-1)

    return final

def create_alpha_channel(mask):
    # create alpha channel from 2D mask image 
    assert mask.ndim==2
    
    # check that the only values are 0 and 1 
    assert np.unique(mask)[0]==0
    assert np.unique(mask)[1]==1    
    
    # flip values 
    masknew = np.zeros_like(mask)
    masknew.fill(255)
    masknew[mask==1] = 0 
    
    return masknew


def apply_mask_to_frame(ref_path,frame_path, savename):
    # assumes that masks exists as a physical png 
    
    assert os.path.exists(ref_path)
    assert os.path.exists(frame_path)
    
    # load and mask
    ref_PIL=Image.open(ref_path)
    mask_arr = get_alpha_channel(ref_PIL)
    frame_PIL=Image.open(frame_path)
    frame_masked_arr = mask_frame(frame_PIL, mask_arr)
    frame_masked_PIL=Image.fromarray(frame_masked_arr, 'RGBA')
    
    # save
    frame_masked_PIL.save(savename, format="png")
    
        
    
if __name__=='__main__':
    
    ref_path=sys.argv[1]
    frame_path = sys.argv[2]
    savename = sys.argv[3]
        
    apply_mask_to_frame(ref_path,frame_path, savename)