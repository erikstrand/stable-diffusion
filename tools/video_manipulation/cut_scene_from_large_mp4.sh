# example of how to cut a scene from a giant mp4 file
giant_video=whole_video.mp4 
scene=rabbit_hole.mp4
scene_start="00:04:52" 
scene_duration="00:04:52" 
ffmpeg -i $giant_video -ss $scene_start -t $scene_duration -c:v copy -c:a copy $scene