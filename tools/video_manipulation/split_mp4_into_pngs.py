import sys 
import os 

import cv2




if __name__ =='__main__':
    
    """python split_video.py <mp4 file> <folder>"""
    
    filename = sys.argv[1]
    savedir = sys.argv[2]
    
    # create dir 
    savedir = savedir +"/"
    os.makedirs(savedir, exist_ok=True)
    
    # read 
    vidcap = cv2.VideoCapture(filename)
    success,image = vidcap.read()

    # save 
    c = 1
    while success:
        if c < 10: 
            i="000"+str(c)
        elif c < 100:
            i="00"+str(c)
        elif c < 1000:
            i="0"+str(c)

        _ = cv2.imwrite(savedir+"IM"+str(i)+".png", image)     # save frame as JPEG file      
        success,image = vidcap.read()
        c += 1    
    
