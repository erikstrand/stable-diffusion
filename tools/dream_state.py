import copy
import random
from masks import Mask


class DreamState:
    def __init__(self, schedule):
        # Initialize constant data.
        self.schedule = schedule
        self.keyframes = [kf for kf in schedule.keyframes]
        # We duplicate the last keyframe so that it doesn't require a special case.
        # (Specifically, there is always a next keyframe and we are always interpolating.)
        self.keyframes.append(copy.copy(self.keyframes[-1]))
        self.keyframes[-1].frame = schedule.keyframes[-1].frame + 1

        # Initialize mutable iteration state.
        self.__iter__()

    def __iter__(self):
        """Sets the iteration state to one before the first frame.

        So note that you can't call get_command() or anything until you've called __next__.
        """

        self.frame_idx = self.keyframes[0].frame - 1
        self.prev_keyframe = None
        self.next_keyframe = self.keyframes[0]
        self.next_keyframe_idx = 0
        self.interp_duration = None
        if self.next_keyframe.seed is not None:
            self.random = random.Random(self.next_keyframe.seed)
        else:
            self.random = random.Random(42)
        self.color_reference = None
        self.seed = None
        return self

    def __next__(self):
        self.frame_idx += 1

        # If we've hit the next frame, update the keyframe state.
        if self.frame_idx == self.next_keyframe.frame:
            self.next_keyframe_idx += 1
            if self.next_keyframe_idx == len(self.keyframes):
                raise StopIteration
            self.prev_keyframe = self.next_keyframe
            self.next_keyframe = self.keyframes[self.next_keyframe_idx]
            self.interp_duration = float(self.next_keyframe.frame - self.prev_keyframe.frame)

        # If we're using random seeds, generate one for this frame.
        # Otherwise use the base seed from the previous keyframe.
        if self.prev_keyframe.seed == None:
            self.seed = self.random.getrandbits(32)
        else:
            self.seed = self.prev_keyframe.seed

        return self

    def input_image_path(self):
        if self.prev_keyframe.input_image is None:
            return None
        else:
            return self.prev_keyframe.input_image.get_path(self.schedule.in_dir, self.frame_idx)

    def output_file(self):
        return f"frame_{self.frame_idx:06d}.png"

    def mask_path(self):
        if self.prev_keyframe.input_image is None:
            return None
        else:
            return str(self.schedule.out_dir / "masks" / self.output_file())

    def output_path(self):
        return self.schedule.out_dir / "frames" / self.output_file()

    @staticmethod
    def interpolate_variations(prev_variations, next_base, next_variations, t):
        n_prev_variations = len(prev_variations)
        n_next_variations = len(next_variations)

        # If neither keyframe has variations, this frame has no variations.
        if n_prev_variations == 0 and n_next_variations == 0:
            return []

        variations = [[var.base, var.amount] for var in prev_variations]
        if n_next_variations == 0:
            # If the next keyframe is a new prompt/seed at full strength, interpolate to it.
            if t > 0.0:
                if variations[-1][0] == next_base:
                    current_weight = variations
                    variations[-1][1] = (1.0 - t) * prev_variations[-1].amount + t
                else:
                    variations.append([next_base, t])
        elif n_next_variations > n_prev_variations:
            # If the next keyframe has a new variation, interpolate its strength.
            assert(n_next_variations == n_prev_variations + 1)
            assert(len(variations) == 0 or variations[-1][0] != next_base)
            if t > 0.0:
                variations.append([
                    next_variations[-1].base,
                    next_variations[-1].amount * t,
                ])
        elif n_next_variations == n_prev_variations:
            # Otherwise, we may interpolate the weight of the last variation.
            prev_weight = prev_variations[-1].amount
            next_weight = next_variations[-1].amount
            if prev_weight != next_weight:
                prev_v_base = prev_variations[-1].base
                next_v_base = next_variations[-1].base
                assert(prev_v_base == next_v_base)
                current_weight = (1.0 - t) * prev_weight + t * next_weight
                variations[-1][1] = current_weight
        else:
            # We shouldn't ever reach here, since there's no way to remove some but not all variations.
            assert(False)
        return variations

    def get_opts(self):
        """Generates the arguments needed to build an InvokeAI command.

        This method is pure -- no state is mutated.
        """

        # Determine our position between prev_keyframe and next_keyframe.
        t = float(self.frame_idx - self.prev_keyframe.frame) / self.interp_duration

        # Determine the output filename.
        outpath = self.output_path()

        # Determine the input image (if any).
        init_img = self.input_image_path()

        # Determine if we need to transform the input image.
        if init_img is None or self.prev_keyframe.transform is None:
            init_img_transform = None
        else:
            init_img_transform = self.prev_keyframe.transform.arg_string()

        # Set the mask path (if any).
        if self.has_mask():
            mask = self.mask_path()
        else:
            mask = None

        # Create the final list of prompt variations. The weight of the last variation may be interpolated.
        prompt_variations = self.interpolate_variations(
            self.prev_keyframe.prompt_variations,
            self.next_keyframe.prompt_idx,
            self.next_keyframe.prompt_variations,
            t
        )
        if len(prompt_variations) == 0:
            prompt_variations = None
        else:
            prompt_variations = [f"{prompt_idx}:{amount:.3f}" for prompt_idx, amount in prompt_variations]
            prompt_variations = ','.join(prompt_variations)

        # If we're using random seeds, generate one for this frame.
        # Otherwise use the base seed from the previous keyframe.
        if self.prev_keyframe.seed == None:
            seed = self.random.getrandbits(32)
        else:
            seed = self.prev_keyframe.seed

        # Create the final list of seed variations. The weight of the last variation may be interpolated.
        seed_variations = self.interpolate_variations(
            self.prev_keyframe.seed_variations,
            self.next_keyframe.seed,
            self.next_keyframe.seed_variations,
            t
        )
        if len(seed_variations) == 0:
            seed_variations = None
        else:
            seed_variations = [f"{seed}:{amount:.3f}" for seed, amount in seed_variations]
            seed_variations = ','.join(seed_variations)

        # Scale and strength are interpolated linearly.
        scale = (1.0 - t) * self.prev_keyframe.scale + t * self.next_keyframe.scale
        strength = (1.0 - t) * self.prev_keyframe.strength + t * self.next_keyframe.strength
        # strength == 0 doesn't seem to work, so we cap it
        strength = max(strength, 0.001)

        # We use the same number of steps and color correction as the previous keyframe.
        steps = self.prev_keyframe.steps
        correct_colors = self.prev_keyframe.correct_colors
        init_color = self.color_reference if correct_colors else None

        # Record the current output file as a color reference, if requested.
        set_color_reference = (self.frame_idx == self.prev_keyframe.frame and self.prev_keyframe.set_color_reference)
        if set_color_reference:
            self.color_reference = str(self.output_path())

        return {
            "-I": init_img,
            "-M": mask,
            "--prompt_idx": self.prev_keyframe.prompt_idx,
            "-P": prompt_variations,
            "-S": seed,
            "-V": seed_variations,
            "-C": f"{scale:.2f}",
            "-f": f"{strength:.3f}",
            "-s": steps,
            "-tf": init_img_transform,
            "--init_color": init_color,
            "-W": self.schedule.width,
            "-H": self.schedule.height,
            "-o": str(outpath.parent),
            "-N": str(outpath.name),
        }

    def get_command(self):
        opts = self.get_opts()
        pairs = [(k, v) for k, v in opts.items() if v is not None]
        return ' '.join(f"{k} {v}" for k, v in pairs)

    def has_mask(self):
        return len(self.prev_keyframe.masks) > 0

    def get_masks(self):
        # Determine our position between prev_keyframe and next_keyframe.
        t = float(self.frame_idx - self.prev_keyframe.frame) / self.interp_duration
        prev_masks = self.prev_keyframe.masks
        next_masks = self.next_keyframe.masks
        n_prev_masks = len(prev_masks)
        n_next_masks = len(next_masks)

        # If the masks are added in the next keyframe, there are no masks now.
        if n_prev_masks == 0:
            return []

        # If the masks are removed in the next keyframe, use the current masks.
        if n_next_masks == 0:
            return prev_masks

        # If there are the same number of masks, interpolate.
        if n_prev_masks == n_next_masks:
            masks = []
            for prev_mask, next_mask in zip(prev_masks, next_masks):
                center = (1.0 - t) * prev_mask.center + t * next_mask.center
                radius = (1.0 - t) * prev_mask.radius + t * next_mask.radius
                masks.append(Mask(center, radius))
            return masks

        # Currently the case where n_prev_masks != n_next_masks and both are greater than zero is
        # not supported.
        assert(False), "Currently you have to create or remove all masks at once."

    def get_seg_masks(self):
        seg_masks = self.next_keyframe.seg_masks
        return seg_masks
