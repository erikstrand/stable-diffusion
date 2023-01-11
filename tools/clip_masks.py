#! python
# myByways simplified Stable Diffusion v0.3 - add clipseg

import os, sys, time
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from einops import rearrange
from pytorch_lightning import seed_everything
from contextlib import nullcontext
from ldm.util import instantiate_from_config
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.ddim import DDIMSampler
from transformers import logging
from torchvision import transforms

from torchvision.utils import save_image as tv_save_image
from clipseg import CLIPDensePredT
import cv2

PROMPTS = [         # --prompt, one or more in an array, one per output image
    'a photo of an old man in a tweed jacket and boots leaning on a fence, dramatic lighting',
]
NEGATIVES = [       # negative prompt, one or more, default None (or an empty array)
]

HEIGHT = 512        # --H, default 512, beyond causes M1 to crawl
WIDTH = 512         # --W, default 512, beyond causes M1 to crawl
FACTOR = 8          # --f downsampling factor, default 8

FIXED = 0           # --fixed_code, 1 for repeatable results, default 0
SEED = 42           # --seed, default 42
NOISE = 0           # --ddim_eta, 0 deterministic, no noise - 1.0 random noise, ignored for PLMS (must be 0)
PLMS = 0            # --plms, default 1 on M1 for txt2img but ignored for img2img (must be DDIM)
ITERATIONS = 1      # --n_iter, default 1
SCALE = 7.5         # --scale, 5 further -> 15 closer to prompt, default 7.5
STEPS = 50          # --ddim_steps, practically little improvement >50 but takes longer, default 50

IMAGE = '1.png'     # --init-img, img2img initial latent seed, default None
STRENGTH = 0.75     # --strength 0 more -> 1 less like image, default 0.75
MASK = [            # inpaint mask PNG where white=overwrite, black=retain, or an array of prompts for CLIPSeg, default=None
    'a woman'
]

FOLDER = 'outputs'  # --outdir for images and history file below
HISTORY = 'history.txt'
CONFIG = 'configs/stable-diffusion/v1-inference.yaml'
CHECKPOINT = 'models/ldm/stable-diffusion-v1/model.ckpt'

THRESHOLDS = 0.1    # CLIPSeg threshold to convert to black and white, or an array of 2 numbers
DILATE = 10         # number of pixels to dilate the mask by
CLIPSEG = 'models/clipseg_weights/rd64-uni-refined.pth'

def seed_pre():
    if not FIXED:
        seed_everything(SEED)

def seed_post(device):
    if FIXED:
        seed_everything(SEED)
        return torch.randn([1, 4, HEIGHT // FACTOR, WIDTH // FACTOR], device='cpu').to(torch.device(device.type))
    return None

def load_model(config, ckpt_file=CHECKPOINT):
    pl_sd = torch.load(ckpt_file, map_location='cpu')
    sd = pl_sd['state_dict']
    model = instantiate_from_config(config.model)
    model.load_state_dict(sd, strict=False)
    return model

def set_device(model):
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        precision = nullcontext
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        precision = torch.autocast
    else:
        device = torch.device('cpu')
        precision = torch.autocast
    model.to(device.type)
    model.eval()
    return device, precision

def load_image_mask(device, image_file=IMAGE, mask_file=MASK):
    image = Image.open(image_file).convert('RGB')
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))
    image = image.resize((w, h), resample=Image.Resampling.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    image = 2. * image - 1.
    image = image.to(device.type)
    mask = None
    if MASK:
        mask = Image.open(mask_file).convert('L')
        mask = mask.resize((w // FACTOR, h // FACTOR), resample=Image.Resampling.LANCZOS)
        mask = np.array(mask).astype(np.float32) / 255.0
        mask = np.tile(mask, (4, 1, 1))
        mask = mask[None].transpose(0, 1, 2, 3)
        mask = torch.from_numpy(mask).to(device.type)
    return image, mask

def load_clipseg(weights_file=CLIPSEG):
    clipseg = CLIPDensePredT(version='ViT-B/16', reduce_dim=64, complex_trans_conv=True)
    clipseg.eval()
    clipseg.load_state_dict(torch.load(weights_file, map_location=torch.device('cpu')), strict=False)
    return clipseg

def create_mask(clipseg, image_file=IMAGE, prompt=MASK):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Resize((WIDTH, HEIGHT)) ])
    image = transform(Image.open(image_file)).unsqueeze(0)
    size = len(MASK)
    with torch.no_grad():
        preds = clipseg(image.repeat(size,1,1,1), MASK)[0]
    image = torch.sigmoid(preds[0][0])
    for i in range(1, size):
        image += torch.sigmoid(preds[i][0])
    image = (image - image.min()) / image.max()
    if isinstance(THRESHOLDS, list):
        if len(THRESHOLDS) >= 2:
            image = torch.where(image >= THRESHOLDS[1], 1., image)
            image = torch.where(image <= THRESHOLDS[0], 0., image)
        else:
             image = torch.where(image >= THRESHOLDS[0], 1., 0.)
    else:
        image = torch.where(image >= THRESHOLDS, 1., 0.)
    mask_file = os.path.join(FOLDER, f'mask_{time.strftime("%Y%m%d_%H%M%S")}.png')
    tv_save_image(image, mask_file)
    return mask_file

def dilate_mask(mask_file):
    image = cv2.imread(mask_file, 0)
    kernel  = np.ones((DILATE, DILATE), np.uint8)
    dilated = cv2.dilate(image, kernel, iterations=1)
    cv2.imwrite(mask_file, dilated)
    return mask_file

def setup_sampler(model):
    global NOISE
    if IMAGE:
        sampler = DDIMSampler(model)
        sampler.make_schedule(ddim_num_steps=STEPS, ddim_eta=NOISE, verbose=False)
        image, mask = load_image_mask(model.device, IMAGE, MASK)
        sampler.t_enc = int(STRENGTH * STEPS)
        sampler.x0 = model.get_first_stage_encoding(model.encode_first_stage(image))
        sampler.z_enc = sampler.stochastic_encode(sampler.x0, torch.tensor([sampler.t_enc]).to(model.device.type))
        sampler.z_mask = mask
        if MASK:
            random = torch.randn(mask.shape, device=model.device)
            sampler.z_enc = (mask * random) + ((1-mask) * sampler.z_enc)
    elif PLMS:
        sampler = PLMSSampler(model)
        NOISE = 0
    else:
        sampler = DDIMSampler(model)
    return sampler

def get_prompts():
    global NEGATIVES
    if NEGATIVES is None:
        NEGATIVES = [''] * len(PROMPTS)
    else:
        NEGATIVES.extend([''] * (len(PROMPTS)-len(NEGATIVES)))
    return zip(PROMPTS, NEGATIVES)

def generate_samples(model, sampler, prompt, negative, start):
    uncond = model.get_learned_conditioning(negative) if SCALE != 1.0 else None
    cond = model.get_learned_conditioning(prompt)
    if IMAGE:
        samples = sampler.decode2(sampler.z_enc, cond, sampler.t_enc, 
            unconditional_guidance_scale=SCALE, unconditional_conditioning=uncond, 
            z_mask=sampler.z_mask, x0=sampler.x0)
    else:
        shape = [4, HEIGHT // FACTOR, WIDTH // FACTOR]
        samples, _ = sampler.sample(S=STEPS, conditioning=cond, batch_size=1,
            shape=shape, verbose=False, unconditional_guidance_scale=SCALE, 
            unconditional_conditioning=uncond, eta=NOISE, x_T=start)
    x_samples = model.decode_first_stage(samples)
    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
    return x_samples

def save_image(image):
    name = f'{time.strftime("%Y%m%d_%H%M%S")}.png'
    image = 255. * rearrange(image.cpu().numpy(), 'c h w -> h w c')
    img = Image.fromarray(image.astype(np.uint8))
    img.save(os.path.join(FOLDER, name))
    return name

def save_history(name, prompt, negative):
    with open(os.path.join(FOLDER, HISTORY), 'a') as history:
        history.write(f'{name} -> {"PLMS" if PLMS else "DDIM"}, Seed={SEED}{" fixed" if FIXED else ""}, Scale={SCALE}, Steps={STEPS}, Noise={NOISE}')
        if IMAGE:
            history.write(f', Image={IMAGE}, Strength={STRENGTH}')
        if MASK:
            history.write(f', Mask={MASK}')
        if len(negative):
            history.write(f'\n + {prompt}\n - {negative}\n')
        else:
            history.write(f'\n + {prompt}\n')

def main():
    global MASK
    print('*** Simple Stable Diffusion - myByways.com simple-sd version 0.3')
    logging.set_verbosity_error()
    os.makedirs(FOLDER, exist_ok=True)
    seed_pre()

    if isinstance(MASK, list):
        print(f'*** Loading CLIPSeg and creating mask')
        tic1 = time.time()
        clipseg = load_clipseg()
        MASK = create_mask(clipseg)
        if DILATE:
            MASK = dilate_mask(MASK)
        toc1 = time.time()
        print(f'*** CLIPSeg masking time: {(toc1 - tic1):.2f}s')
    if not PROMPTS:
        sys.exit(0)

    print(f'*** Loading Stable Diffusion')
    tic1 = time.time()
    config = OmegaConf.load(CONFIG)
    model = load_model(config)
    device, precision_scope = set_device(model)
    sampler = setup_sampler(model)
    start_code = seed_post(device)
    toc1 = time.time()
    print(f'*** Stable Diffusion setup time: {(toc1 - tic1):.2f}s')

    counter = 0
    with torch.no_grad():
        with precision_scope(device.type):
            with model.ema_scope():

                for iteration in range(ITERATIONS):
                    for prompt, negative in get_prompts():
                        print(f'*** Iteration {iteration + 1}: {prompt}')
                        tic2 = time.time()
                        images = generate_samples(model, sampler, prompt, negative, start_code)
                        for image in images:
                            name = save_image(image)
                            save_history(name, prompt, negative)
                            print(f'*** Saved image: {name}')

                        counter += len(images)
                        toc2 = time.time()
                        print(f'*** Synthesis time: {(toc2 - tic2):.2f}s')

    print(f'*** Total time: {(toc2 - tic1):.2f}s')
    print(f'*** Saved {counter} image(s) to {FOLDER} folder.')

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print('*** User abort, goodbye.')
    except FileNotFoundError as e:
        print(f'*** {e}')
