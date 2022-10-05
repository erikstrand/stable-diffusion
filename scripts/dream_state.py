class DreamState:
    def __init__(self, schedule):
        self.schedule = schedule
        self.prev_keyframe = None
        self.next_keyframe = schedule.keyframes[0]
        self.next_keyframe_idx = 1
        self.interp_duration = None
        self.frame_idx = 0

    def advance_keyframe(self):
        self.prev_keyframe = self.next_keyframe
        self.next_keyframe = self.schedule.keyframes[self.next_keyframe_idx]
        self.next_keyframe_idx += 1
        self.interp_duration = float(self.next_keyframe.frame - self.prev_keyframe.frame)

    def advance_frame(self):
        self.frame_idx += 1
        if self.frame_idx == self.next_keyframe.frame:
            self.advance_keyframe()

    def done(self):
        # Note that we don't actually output the final keyframe. I should fix this eventually.
        # For now I'm just going to add a sentinel keyframe at the end.
        return self.next_keyframe_idx >= len(self.schedule.keyframes)

    def get_opts(self):
        t = float(self.frame_idx - self.prev_keyframe.frame) / self.interp_duration
        scale = (1.0 - t) * self.prev_keyframe.scale + t * self.next_keyframe.scale
        strength = (1.0 - t) * self.prev_keyframe.strength + t * self.next_keyframe.strength

        n_prev_variations = len(self.prev_keyframe.seed_variations)
        n_next_variations = len(self.next_keyframe.seed_variations)
        variations = [[var.seed, var.amount] for var in self.prev_keyframe.seed_variations]
        if n_next_variations == n_prev_variations or self.frame_idx == self.prev_keyframe.frame:
            # If the next keyframe has the same number of variations, we aren't changing anything now.
            pass
        elif n_next_variations < n_prev_variations:
            # If the next keyframe has fewer variations, we interpolate to its seed.
            assert(n_next_variations == 0)
            variations.append([self.next_keyframe.seed, t])
        else:
            # If the next keyframe has more variations, interpolate the strength of the new one.
            assert(n_next_variations == n_prev_variations + 1)
            variations.append({
                self.next_keyframe.variations[-1].seed,
                self.next_keyframe.variations[-1].amount * t,
            })
        if len(variations) == 0:
            variations = None

        return {
            "prompt": "",
            "latent_0": self.prev_keyframe.prompt,
            "latent_1": self.next_keyframe.prompt,
            "latent_interpolation": t,
            "seed": self.prev_keyframe.seed,
            "cfg_scale": scale,
            "strength": strength,
            "width": self.schedule.width,
            "height": self.schedule.height,
            "with_variations": variations,
        }
