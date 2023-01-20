import sys 
import os 

import cv2




if __name__ =='__main__':
    
    """python split_video.py <mp4 file> <folder>"""
    
    filename = sys.argv[1]
    savedir = sys.argv[2]
    if len(sys.argv)>3:
        start_from = int(sys.argv[3])
    else:
        start_from = None
    
    # create dir 
    savedir = savedir +"/"
    os.makedirs(savedir, exist_ok=True)
    
    # read 
    vidcap = cv2.VideoCapture(filename)
    success,image = vidcap.read()

    # save 
    c = 1
    while success:
        
        # skip frames 
        if start_from is not None and c<start_from:
            c += 1    
            success,image = vidcap.read()
            continue
                
        # get right name
        if c < 10: 
            i="000"+str(c)
        elif c < 100:
            i="00"+str(c)
        elif c < 1000:
            i="0"+str(c)
        else:
            i=str(c)
        
        # maxes out at 9,999 frames
        if c>9999: 
            break 

        # write
        _ = cv2.imwrite(savedir+"IM"+str(i)+".png", image)     # save frame as JPEG file      
        success,image = vidcap.read()
        c += 1    
    
