import sys 
import os 
import moviepy.editor as mp


if __name__ == '__main__':
    
    mp4_file = sys.argv[1]
    start = float(sys.argv[2]) # in seconds - e.g. 56.24
    finish = float(sys.argv[3])
    savename = sys.argv[4]
    
    assert os.path.exists(mp4_file)
    assert finish > start
    
    my_clip = mp.VideoFileClip(mp4_file)
    my_clip_cut = my_clip.subclip(start, finish) 
    my_clip_cut.write_videofile(savename)