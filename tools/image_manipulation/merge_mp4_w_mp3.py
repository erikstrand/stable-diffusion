import sys 
import os 
import moviepy.editor as mp


if __name__ == '__main__':
    
    mp4_file = sys.argv[1]
    mp3_file = sys.argv[2]
    savename = sys.argv[3]
    
    assert os.path.exists(mp4_file)
    assert os.path.exists(mp3_file)

    my_clip_sd_video = mp.VideoFileClip(mp4_file)
    my_clip_sd_audio = mp.AudioFileClip(mp3_file)

    my_clip_sd_video.audio = mp.CompositeAudioClip([my_clip_sd_audio])
    my_clip_sd_video.write_videofile(savename)
    print("Done")
