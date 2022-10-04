ffmpeg -r 12.5 -f image2 -s 640x320 -i ../../../outputs/test5/clip_00/%06d.0.png -vcodec libx264 -crf 10 -pix_fmt yuv420p ../../../outputs/test5/videos/clip_00.mp4
