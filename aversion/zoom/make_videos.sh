ffmpeg -r 10.0 -f image2 -s 512x512 -i ../../outputs/zoom/00/frames/%06d.0.png -vcodec libx264 -crf 10 -pix_fmt yuv420p ../../outputs/zoom/zoom_00.mp4
ffmpeg -r 10.0 -f image2 -s 512x512 -i ../../outputs/lava/frames/%06d.0.png -vcodec libx264 -crf 10 -pix_fmt yuv420p ../../outputs/lava/lava_00.mp4
