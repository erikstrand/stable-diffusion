class Prompt:
    def __init__(
        self,
        prompt,
        latent_0,
        latent_1,
        latent_interpolation,
        seed,
        with_variations,
        cfg_scale,
        width,
        height,
        outdir
    ):
        self.prompt = prompt
        self.latent_0 = latent_0
        self.latent_1 = latent_1
        self.latent_interpolation = latent_interpolation
        self.seed = seed
        self.with_variations = with_variations
        self.cfg_scale = cfg_scale
        self.width = width
        self.height = height
        self.outdir = outdir

        self.strength = 0.0
        self.steps = 50
        self.sampler_name = "k_lms"
        self.grid = False
        self.individual = True
        self.save_original = False
        self.exclude_seed_from_filename = True
        self.variation_amount = 0.0
        self.init_img = None
        self.init_mask = None
        self.seamless = False
        self.fit = False
        self.gfpgan_strength=0.0
        self.upscale = None
        self.skip_normalize = False
        self.log_tokenization = False
        self.iterations = 1


class DreamState:
    def __init__(self, schedule):
        self.schedule = schedule
        self.prev_keyframe = schedule.keyframes[0]
        self.next_keyframe = schedule.keyframes[1]
        self.next_keyframe_idx = 1
        self.interp_duration = None
        self.frame_idx = 0
        self.interp_duration = float(self.next_keyframe.frame - self.prev_keyframe.frame)

    def advance_keyframe(self):
        self.next_keyframe_idx += 1
        self.prev_keyframe = self.next_keyframe
        self.next_keyframe = self.schedule.keyframes[self.next_keyframe_idx]
        self.interp_duration = float(self.next_keyframe.frame - self.prev_keyframe.frame)

    def advance_frame(self):
        self.frame_idx += 1
        if self.frame_idx == self.next_keyframe.frame:
            self.advance_keyframe()

    def done(self):
        # Note that we don't actually output the final keyframe. I should fix this eventually.
        # For now I'm just going to add a sentinel keyframe at the end.
        return (
            self.frame_idx + 1 == self.next_keyframe.frame and
            self.next_keyframe_idx + 1 == len(self.schedule.keyframes)
        )

    def get_opts(self):
        t = float(self.frame_idx - self.prev_keyframe.frame) / self.interp_duration
        scale = (1.0 - t) * self.prev_keyframe.scale + t * self.next_keyframe.scale
        strength = (1.0 - t) * self.prev_keyframe.strength + t * self.next_keyframe.strength

        n_prev_variations = len(self.prev_keyframe.seed_variations)
        n_next_variations = len(self.next_keyframe.seed_variations)
        variations = [[var.seed, var.amount] for var in self.prev_keyframe.seed_variations]
        if n_next_variations == 0:
            # If the next keyframe is a new seed at full strength, interpolate to it.
            variations.append([self.next_keyframe.seed, t])
        elif n_next_variations > n_prev_variations:
            # If the next keyframe has more variations, interpolate the strength of the new one.
            assert(n_next_variations == n_prev_variations + 1)
            variations.append([
                self.next_keyframe.seed_variations[-1].seed,
                self.next_keyframe.seed_variations[-1].amount * t,
            ])
        # Otherwise, we're keeping the seed/variations constant.

        if len(variations) == 0:
            variations = None

        return {
            "prompt": "",
            "latent_0": self.prev_keyframe.prompt,
            "latent_1": self.next_keyframe.prompt,
            "latent_interpolation": t,
            "seed": self.prev_keyframe.seed,
            "with_variations": variations,
            "cfg_scale": scale,
            #"strength": strength, # only need this for img2img
            "width": self.schedule.width,
            "height": self.schedule.height,
            "outdir": str(self.schedule.out_dir),
        }

    def get_prompt(self):
        return Prompt(**self.get_opts())

    def get_command(self):
        opts = self.get_opts()
        pairs = [(k, v) for k, v in opts.items() if k != "prompt" and v is not None]
        return opts["prompt"] + ' ' + ' '.join(f"--{k} {v}" for k, v in pairs)
