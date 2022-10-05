ffmpeg -r 10.0 -f image2 -s 512x512 -i ../../outputs/test/clip_00/%06d.0.png -vcodec libx264 -crf 10 -pix_fmt yuv420p ../../outputs/test/zoom.mp4
