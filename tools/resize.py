from PIL import Image
import os

def resize_folder(folder, out_dims):
    to_resize = []
    for name in sorted(os.listdir(folder)):
        to_resize.append(os.path.join(folder, name))

    for path in to_resize:
        print(path)
        im = Image.open(path)
        im = im.resize(out_dims)
        out_path = path[:-3] + "png"
        im.save(out_path)

if __name__ == "__main__":
    resize_folder("/home/anna/repos/stable-diffusion/seance_full_size/frames/", (960, 540))