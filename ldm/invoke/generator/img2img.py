'''
ldm.invoke.generator.img2img descends from ldm.invoke.generator
'''

import torch
import numpy as  np
from ldm.invoke.devices             import choose_autocast
from ldm.invoke.generator.base      import Generator
from ldm.models.diffusion.ddim     import DDIMSampler

class Img2Img(Generator):
    def __init__(self, model, precision):
        super().__init__(model, precision)
        self.init_latent         = None    # by get_noise()

    def get_make_image(self,prompt,sampler,steps,cfg_scale,ddim_eta,
                       conditioning,init_image,strength,step_callback=None,threshold=0.0,perlin=0.0,**kwargs):
        """
        Returns a function returning an image derived from the prompt and the initial image
        Return value depends on the seed at the time you call it.
        """
        self.perlin = perlin

        sampler.make_schedule(
            ddim_num_steps=steps, ddim_eta=ddim_eta, verbose=False
        )

        print(f"init_image: {init_image.mean()}")
        scope = choose_autocast(self.precision)
        with scope(self.model.device.type):
            self.init_latent = self.model.get_first_stage_encoding(
                self.model.encode_first_stage(init_image)
            ) # move to latent space
        print(f"setting init_latent: {self.init_latent.mean()}")

        t_enc = int(strength * steps)
        uc, c   = conditioning

        def make_image(x_T):
            print(f"t_enc: {t_enc}")
            print(f"init_latent mean: {self.init_latent.mean()}")
            print(f"x_T mean: {x_T.mean()}")
            # encode (scaled latent)
            z_enc = sampler.stochastic_encode(
                self.init_latent,
                torch.tensor([t_enc]).to(self.model.device),
                noise=x_T
            )
            print(f"z enc mean: {z_enc.mean()}")
            # decode it
            samples = sampler.decode(
                z_enc,
                c,
                t_enc,
                img_callback = step_callback,
                unconditional_guidance_scale=cfg_scale,
                unconditional_conditioning=uc,
                init_latent = self.init_latent,  # changes how noising is performed in ksampler
            )
            print(f"output mean: {samples.mean()}")

            return self.sample_to_image(samples)

        return make_image

    def get_noise(self,width,height):
        device      = self.model.device
        init_latent = self.init_latent
        assert init_latent is not None,'call to get_noise() when init_latent not set'
        if device.type == 'mps':
            x = torch.randn_like(init_latent, device='cpu').to(device)
        else:
            x = torch.randn_like(init_latent, device=device)
        if self.perlin > 0.0:
            shape = init_latent.shape
            x = (1-self.perlin)*x + self.perlin*self.get_perlin_noise(shape[3], shape[2])
        return x
