# Check image resolution 

import os 
import sys 
from PIL import Image

if __name__=='__main__':
    
    file=sys.argv[1]
    assert os.path.exists(file)
    print("Image size is")
    print(Image.open(file).size)