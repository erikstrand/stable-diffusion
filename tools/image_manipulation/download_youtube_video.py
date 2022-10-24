import os 
import sys 

from pytube import YouTube

if __name__ =='__main__':
    
    """Usage: python download_youtube.py <link> <savename> [resolution] [proxy]"""
    
    link = sys.argv[1]
    savename = sys.argv[2]
    
    # default settings for optional vars
    res='360' 
    proxy=None
    
    # grab optional params
    if len(sys.argv>3):
        res=sys.argv[3]
    if len(sys.argv>4):
        proxy = sys.argv[4]
        if proxy == 'harvard':
            proxy = {"http":"http://proxy.tch.harvard.edu:3128"}
    
    # get link 
    try:
        if proxy is not None:
            yt = YouTube(link, proxies=proxy)
        else:
            yt = YouTube(link)
    except: 
        print("Connection Error")
    

    # download 
    yt.streams.filter(progressive = True, file_extension = "mp4", res=res+'p').first().download(filename = savename)
    print("Success.")