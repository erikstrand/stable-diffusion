import random


class DreamState:
    def __init__(self, schedule):
        self.schedule = schedule
        self.prev_keyframe = schedule.keyframes[0]
        self.next_keyframe = schedule.keyframes[1]
        self.next_keyframe_idx = 1
        self.interp_duration = None
        self.frame_idx = 0
        self.interp_duration = float(self.next_keyframe.frame - self.prev_keyframe.frame)
        self.random = random.Random(self.prev_keyframe.seed)

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
        # Determine our position between prev_keyframe and next_keyframe.
        t = float(self.frame_idx - self.prev_keyframe.frame) / self.interp_duration

        # Create the final list of prompt variations. The weight of the last variation may be interpolated.
        n_prev_variations = len(self.prev_keyframe.prompt_variations)
        n_next_variations = len(self.next_keyframe.prompt_variations)
        prompt_variations = [[var.prompt, var.amount] for var in self.prev_keyframe.prompt_variations]
        if n_next_variations == 0:
            # If the next keyframe is a new prompt at full strength, and we're strictly between
            # keyframes, interpolate to it.
            if self.frame_idx > self.prev_keyframe.frame:
                prompt_variations.append([self.next_keyframe.prompt, t])
        elif n_next_variations > n_prev_variations:
            # If the next keyframe has more variations, interpolate the strength of the new one.
            assert(n_next_variations == n_prev_variations + 1)
            prompt_variations.append([
                self.next_keyframe.prompt_variations[-1].prompt,
                self.next_keyframe.prompt_variations[-1].amount * t,
            ])
        elif n_next_variations == n_prev_variations:
            # Otherwise, interpolate the weight of the last variation.
            prev_prompt = self.prev_keyframe.prompt_variations[-1].prompt
            prev_weight = self.prev_keyframe.prompt_variations[-1].amount
            next_prompt = self.next_keyframe.prompt_variations[-1].prompt
            next_weight = self.next_keyframe.prompt_variations[-1].amount
            assert(prev_prompt == next_prompt)
            prompt_variations[-1][1] = (1.0 - t) * prev_weight + t * next_weight
        else:
            # We shouldn't ever reach here, since there's no way to remove some but not all variations.
            assert(False)
        if len(prompt_variations) == 0:
            prompt_variations = None


        # Scale and strength are interpolated linearly.
        scale = (1.0 - t) * self.prev_keyframe.scale + t * self.next_keyframe.scale
        strength = (1.0 - t) * self.prev_keyframe.strength + t * self.next_keyframe.strength

        # We use the same number of steps as the previous keyframe.
        steps = self.prev_keyframe.steps

        # If we're using random seeds, generate one for this frame.
        # Otherwise use the base seed from the previous keyframe.
        if self.prev_keyframe.seed == None:
            seed = self.random.getrandbits(32)
        else:
            seed = self.prev_keyframe.seed

        # Create the final list of seed variations. The weight of the last variation may be interpolated.
        n_prev_variations = len(self.prev_keyframe.seed_variations)
        n_next_variations = len(self.next_keyframe.seed_variations)
        seed_variations = [[var.seed, var.amount] for var in self.prev_keyframe.seed_variations]
        if n_next_variations == 0:
            # If the next keyframe is a new seed at full strength, and we're strictly between
            # keyframes, interpolate to it.
            if self.frame_idx > self.prev_keyframe.frame:
                seed_variations.append([self.next_keyframe.seed, t])
        elif n_next_variations > n_prev_variations:
            # If the next keyframe has more variations, interpolate the strength of the new one.
            assert(n_next_variations == n_prev_variations + 1)
            seed_variations.append([
                self.next_keyframe.seed_variations[-1].seed,
                self.next_keyframe.seed_variations[-1].amount * t,
            ])
        elif n_next_variations == n_prev_variations:
            # Otherwise, interpolate the weight of the last variation.
            prev_seed = self.prev_keyframe.seed_variations[-1].seed
            prev_weight = self.prev_keyframe.seed_variations[-1].amount
            next_seed = self.next_keyframe.seed_variations[-1].seed
            next_weight = self.next_keyframe.seed_variations[-1].amount
            assert(prev_seed == next_seed)
            seed_variations[-1][1] = (1.0 - t) * prev_weight + t * next_weight
        else:
            # We shouldn't ever reach here, since there's no way to remove some but not all variations.
            assert(False)
        if len(seed_variations) == 0:
            seed_variations = None

        color_coherence = self.prev_keyframe.color_coherence
        is_color_reference = (self.frame_idx == self.prev_keyframe.frame and self.prev_keyframe.is_color_reference)

        return {
            "prompt_index": self.prev_keyframe.prompt,
            "prompt_variations": prompt_variations,
            "seed": seed,
            "with_variations": seed_variations,
            "cfg_scale": scale,
            "strength": strength,
            "steps": steps,
            "width": self.schedule.width,
            "height": self.schedule.height,
            "outdir": str(self.schedule.out_dir),
            "animation": self.prev_keyframe.animation,
            "color_coherence": color_coherence,
            "is_color_reference": is_color_reference,
        }

    def get_command(self):
        opts = self.get_opts()
        pairs = [(k, v) for k, v in opts.items() if v is not None]
        return ' '.join(f"--{k} {v}" for k, v in pairs)
