####################################
# Copy generated images
####################################

# copy from fs
cd ~/w/code/sd/experiments/s20220929_eric/stable-diffusion/outputs/
rsync -av ~/fs/code/sd/experiments/s20220929_eric/stable-diffusion/outputs/* . 

# copy from mghpcc
cd ~/w/code/sd/experiments/s20220929_eric/stable-diffusion/outputs/
rsync -av ch215616@mghpcc:~/w/code/sd/experiments/s20220929_eric/stable-diffusion/outputs/* . 

# generate videos 
cd ~/w/code/sd/experiments/s20220929_eric/stable-diffusion/outputs
cd e1_1
for i in {1..41}
do 
    cd ../e1_${i}
    ffmpeg -y -r 10.0 -f image2 -s 1024x768 -i %06d.0.png -vcodec libx264 -crf 10 -pix_fmt yuv420p ../videos/clip_00_e1_${i}.mp4 
    #ffmpeg -y -r 10.0 -f image2 -s 1024x768 -i %06d.0.png -vcodec libx264 -crf 10 -pix_fmt yuv420p ../videos/clip_00_e1_${i}.mp4 > log_video_making
done


# check which videos exist 
for i in {1..41}
do
    echo $i
    cd ../e1_${i}
    ls ../videos/clip_00_e1_${i}.mp4
done 


# copy videos
cd ~/w/code/sd/experiments/s20220929_eric/stable-diffusion/outputs/
rclone delete didenco:/videos/aversion/serge/
rclone copy videos didenco:/videos/aversion/serge/ 



####################################
# SSH into servers
####################################

# karjakin
conda activate ldm_fs2
cd ~/fs/code/sd/experiments/s20220929_eric/stable-diffusion/

# e2 
e2
srun -A crl -p crl-gpu -t 01:00:00 --qos=crl --gres=gpu:NVIDIA_A40:1 --pty /bin/bash 
source activate /home/ch215616/.conda/envs/ldm
cd ~/w/code/sd/experiments/s20220929_eric/stable-diffusion/

# mghpcc
e2
mgh 
srun -t 08:00:00 -p mghpcc-gpu -n 8 --mem 80GB --gres=gpu:Quadro_RTX_8:1 --pty /bin/bash                                                         
conda activate ldm 
cd ~/w/code/sd/experiments/s20220929_eric/stable-diffusion/



####################################
# Copy latest config to servers (not git related)
####################################

# copy to fs 
cd ~/w/code/sd/experiments/s20220929_eric/stable-diffusion/aversion
rsync -av * ~/fs/code/sd/experiments/s20220929_eric/stable-diffusion/aversion/

# copy to mgh
cd ~/w/code/sd/experiments/s20220929_eric/stable-diffusion/
rsync -ravzhe aversion/* ~/w/code/aversion_temp/
e2 
cd ~/w/code/aversion_temp
rsync -ravzh * mghpcc:/scratch/ch215616/w/code/sd/experiments/s20220929_eric/stable-diffusion/aversion/

# [unused] copy to sd2 
cd ~/w/code/sd/experiments/s20220929_eric/stable-diffusion
t=~/w/code/sd2/experiments/s20220929_eric/stable-diffusion
rsync -av --exclude '*.png' --exclude '*.jpg' --exclude '*.gif' --exclude '*.mp4' ./ $t


# [unused]  copy to fs (all files)
cd ~/w/code/sd/experiments/s20220929_eric/stable-diffusion/
rsync -av --exclude '*.png' --exclude '*.jpg' --exclude '*.gif' --exclude '*.mp4' ./ ~/fs/code/sd/experiments/s20220929_eric/stable-diffusion/


####################################
# Run
####################################

# run with toml file 
i=22; python scripts/dream.py --from_dream_schedule aversion/e1_${i}/config.toml


# [unused] generate commands
i=4; python scripts/config_reader_to_text.py aversion/e1_${i}/config.toml aversion/e1_${i}/commands.txt

# [unused]  run with prompts file 
exp=5f; python scripts/dream.py --prompts_file 99f/config/test${exp}/prompts.txt --from_file 99f/config/test${exp}/commands.txt 
i=2; python scripts/dream.py --prompts aversion/e1_${i}/prompts.txt --from_file aversion/e1_${i}/commands_cut.txt


####################################
# Find last frame / Count runs 
####################################

# restart a running job - make sure to copy the data from fs and mgh first
cd ~/w/code/sd/experiments/s20220929_eric/stable-diffusion/outputs/e1_1
i=23; cd ../e1_${i}/; ls -latr *png | tail -2


# count runs 
for i in {1..41}
do 
    echo $i
    cd ../e1_${i}/; ls -latr *png | wc -l 
done 

# display last 
for i in {1..41}
do 
    echo ""
    echo $i
    cd ../e1_${i}/; ls -latr *png | tail -1 
done 


####################################
# Some copy problems 
####################################

mgh - e1_6 - does not copy for some reason - owner is ansible

####################################
# Prepare to restart a job - update .toml
####################################


# restart a running job - make sure to copy the data from fs and mgh first
cd ~/w/code/sd/experiments/s20220929_eric/stable-diffusion/outputs/e1_1
i=2; cd ../e1_${i}/; ls -latr *png | tail -2
nano ../../aversion/e1_${i}/config.toml


# what to restart 
2
4
5
6

# what to add motion to 
3  - and change seed
7
12 
15 - change speed to FASTER (and add more random numbers at the end)
20 - same as above 
22 


# also 
one more instance of liquid_chrome_dreams 

# just run mass test of various prompts 

# e3 (previously creepy)
"beautiful mannequin sculpted out of black glass by billelis + lit with geometric neon dripping gold + wearing a crown of beta fish, doorway opening with neon pink geometric fractal light + flowering bonsai trees!!!!, transcendent, clean linework, dramatic, finely detailed, 4 k, trending on artstation, award winning, photorealistic, volumetric lighting, octane render" 
-C9 -W1024 -H768 -n9 
>>
"beautiful mannequin sculpted out of black glass by billelis + lit with geometric neon dripping gold + wearing a crown of beta fish, doorway opening with neon pink geometric fractal light + flowering bonsai trees!!!!, transcendent, clean linework, dramatic, finely detailed, 4 k, trending on artstation, award winning, photorealistic, volumetric lighting, octane render"  -S3048261515 -C9 -W1024 -H768 
"beautiful mannequin sculpted out of black glass by billelis + lit with geometric neon dripping gold + wearing a crown of beta fish, doorway opening with neon pink geometric fractal light + flowering bonsai trees!!!!, transcendent, clean linework, dramatic, finely detailed, 4 k, trending on artstation, award winning, photorealistic, volumetric lighting, octane render"  -S3048261 -C9 -W1024 -H768 

# e7 (too similar to liquid dreams)
"abstract impossible chrome drippy sculpture fine art jewelry, concept art, 3D object, octane render, unreal engine, album art, by Alberto Seveso" -C10 -n9 -W1024 -H768 -S35309805
"abstract impossible chrome drippy sculpture fine art jewelry, concept art, 3D object, octane render, unreal engine, album art, by Alberto Seveso" -C10 -n9 -W1024 -H768 -S1419836632
>> 
"abstract impossible chrome drippy sculpture fine art jewelry, concept art, 3D object, octane render, unreal engine, album art, by Alberto Seveso" -C10 -S2408416340 -W1024 -H768 
"abstract impossible chrome drippy sculpture fine art jewelry, concept art, 3D object, octane render, unreal engine, album art, by Alberto Seveso" -C10 -S2641992039 -W1024 -H768 
"abstract impossible chrome drippy sculpture fine art jewelry, concept art, 3D object, octane render, unreal engine, album art, by Alberto Seveso" -C10 -S3646477555 -W1024 -H768 
"abstract impossible chrome drippy sculpture fine art jewelry, concept art, 3D object, octane render, unreal engine, album art, by Alberto Seveso" -C10 -S1172696231 -W1024 -H768 
"abstract impossible chrome drippy sculpture fine art jewelry, concept art, 3D object, octane render, unreal engine, album art, by Alberto Seveso" -C10 -S1075723117 -W1024 -H768 

# e12 (hourglass)
"cinema 4D render, A24!film cinematography, humanoid inside an hourglass, falling sand inside, futuristic 1990s contemporary art, sci-fi, trapped inside an hourglass, sand, time, deserted sand, glass, inside view, humanoid pov, intricate artwork by Tooth Wu and wlop and beeple. octane render, trending on artstation, greg rutkowski very coherent symmetrical artwork, depth field, unreal engine, cinematic, hyper realism, high detail, octane render, 8k" -C7 -n9  -W1024 -H768 
>> none 

# e28
"twisted, menagerie, quirky, A cybernetic symbiosis, ceramics, gaseous materials," -C7.5  -W1024 -H768 -n6 
>> none chosen 

# e29 
"fragrance by anish kapoor, highly detailed, intricate, breath taking,, octane,red" -C7.5  -W1024 -H768 -n9
>> none chosen 

# neon 
"neon hieroglyphics intertwined with shiny metal sculpture, onyx, pink, highly detailed, ultra realistic, octane render 8k, by guerrino boatto" -C10  -W1024 -H768 -n6 
>> best one is: 
    000012.2644993744, 000012.2937064479
"hieroglyphics egypt intertwined with shiny red sculpture, onyx, pink, saphire, highly detailed, ultra realistic, octane render 8k, by guerrino boatto" -C10  -W1024 -H768 -n6
>> best one is:
    000012.3670896136
"hieroglyphics intertwined with red gaseous symbiosis, onyx, pink, saphire, highly detailed, ultra realistic, octane render 8k, by guerrino boatto" -C10  -W1024 -H768 -n6
>> never tried this in the end

# lava
"a primordial landscape, lava ocean meets obsidian mountains, night sky with bright stars, steam, sparks, wide field of view, acrylic paint, brush strokes, godrays, ultra detailed, high resolution, art by inessa garmash, trending on artstation" -C10  -W1024 -H768 -n6 
>> chosen as the best 
    000002.3688524458 -> C10 
    000002.405867197.png -> C10 

# Tasks: 


# to run 
mgh - 33 x 2 -> running
karjakin / ankara / gamakichi - 28 x 3 (first three) -> running 
izmir - 28 x 2 (last two) -> running 

# make another 
- neon - 3 versions - 2xlonger - with more rotation and different seed -> ankara / karjakin / rayan
- lava - 2 versions - 2xlonger - with more rotation and different seed -> mgh 

# after wake up 
combine all videos -> note that some files would not transfer (e1_6...)
find out if any runs stopped prematurely - restart them
15, 20 - do FASTER runs...with motion.... 


# update 
