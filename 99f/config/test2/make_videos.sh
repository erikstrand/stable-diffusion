ffmpeg -r 25 -f image2 -s 640x320 -i ../../../outputs/test2/clip_00/%06d.0.png -vcodec libx264 -crf 10 -pix_fmt yuv420p ../../../outputs/test2/videos/clip_00.mp4
