import toml
import re
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from dream_state import DreamState
from masks import Mask

# TODO
# - implement "interpolate" for fields other than strength


@dataclass
class PromptVariation:
    __slots__ = ["prompt", "prompt_idx", "amount"]
    prompt: str
    prompt_idx: int
    amount: float

    @property
    def base(self):
        return self.prompt_idx


@dataclass
class SeedVariation:
    __slots__ = ["seed", "amount"]
    seed: int
    amount: float

    @property
    def base(self):
        return self.seed


@dataclass
class InputImage:
    """A smart object that constructs the -I argument for a command.

    This class handles three different cases. When all fields are None, it will generate "-1" (which
    means use the previous frame). When path_start is set and path_end is None, it generates
    f"{path_start}". Otherwise, it generates f"{path_start}{str(frame +
    frame_delta).zfill(n_digits)}{path_end}". The latter is used to sequentially process multiple
    frames from an input video.
    """

    __slots__ = ["path_start", "path_end", "keyframe", "frame_delta", "n_digits"]
    path_start: str
    path_end: str
    keyframe: int
    frame_delta: int
    n_digits: int

    def in_prev_mode(self):
        return self.path_start is None

    def in_video_mode(self):
        return self.path_start is not None

    @classmethod
    def from_prev(cls):
        return cls(path_start=None, path_end=None, keyframe=None, frame_delta=None, n_digits=None)

    @classmethod
    def from_path(cls, path, keyframe_idx):
        re_res = re.search(r"{(\d+)}", path)
        if re_res is None:
            return cls(
                path_start  = path,
                path_end    = None,
                keyframe    = keyframe_idx,
                frame_delta = None,
                n_digits    = None,
            )
        else:
            start_frame = int(re_res.group(1))
            span = re_res.span()
            return cls(
                path_start  = path[:span[0]],
                path_end    = path[span[1]:],
                keyframe    = keyframe_idx,
                frame_delta = start_frame - keyframe_idx,
                n_digits    = span[1] - span[0] - 2,
            )

    @classmethod
    def from_string(cls, string, frame):
        if string == "previous":
            return cls.from_prev()
        else:
            return cls.from_path(string, frame)

    def get_path(self, in_dir, frame):
        if self.path_start is None:
            return "-1"
        elif self.path_end is None:
            return in_dir / self.path_start
        else:
            assert(frame >= self.keyframe)
            input_frame = frame + self.frame_delta
            input_frame = str(input_frame).zfill(self.n_digits)
            return in_dir / f"{self.path_start}{input_frame}{self.path_end}"


@dataclass
class MaskFillImage:
    __slots__ = ["use_prev", "frame"]

    def __init__(self, use_prev, frame):
        self.use_prev = use_prev
        self.frame = frame

    @classmethod
    def from_string(cls, string, frame):
        if string == "previous":
            return cls(use_prev=True, frame=None)
        else:
            x = int(string)
            if x >= 0:
                return cls(use_prev=False, frame=x)
            else:
                return cls(use_prev=False, frame=frame + x)

    def get_frame(self, frame):
        if self.use_prev:
            return frame - 1
        else:
            return self.frame


class Transform2D:
    __slots__ = ["zoom", "translation", "rotation"]

    def __init__(self, zoom, translation, rotation):
        self.rotation = float(rotation) # in degrees
        self.zoom = float(zoom) # in unitless scale factor
        self.translation = (float(translation[0]), float(translation[1])) # in pixels

    def arg_string(self):
        return f"\"{self.rotation:.3f}:{self.zoom:.3f}:{self.translation[0]:.3f}:{self.translation[1]:.3f}\""


class MaskFillTransform:
    __slots__ = ["zoom", "translation", "center"]

    def __init__(self, zoom, translation, center):
        self.zoom = float(zoom) # in unitless scale factor
        self.translation = (float(translation[0]), float(translation[1])) # in pixels
        self.center = (float(center[0]), float(center[1])) # in pixels

    def arg_string(self):
        return f"\"{self.zoom:.3f}:{self.translation[0]:.3f}:{self.translation[1]:.3f}:{self.center[0]:.3f}:{self.center[1]:.3f}\""


class KeyFrame:
    __slots__ = [
        "frame",
        "input_image",
        "prompt",
        "prompt_idx",
        "prompt_variations",
        "seed",
        "seed_variations",
        "scale",
        "strength",
        "steps",
        "input_mask",
        "masks",
        "fill_mask",
        "transform",
        "correct_colors",
        "set_color_reference"
    ]

    def __init__(
        self,
        frame,
        input_image,
        prompt,
        prompt_variations,
        seed,
        seed_variations,
        scale,
        strength,
        steps,
        input_mask,
        masks,
        fill_mask,
        transform,
        correct_colors,
        set_color_reference
    ):
        self.frame = frame

        assert(input_image is None or isinstance(input_image, InputImage))
        self.input_image = input_image

        assert(isinstance(prompt, str))
        self.prompt = prompt

        # This is set by DreamSchedule once it has accumulated the prompts from all keyframes.
        self.prompt_idx = None

        assert(isinstance(prompt_variations, list))
        for pv in prompt_variations:
            assert(isinstance(pv, PromptVariation))
        self.prompt_variations = prompt_variations

        self.seed = seed

        assert(isinstance(seed_variations, list))
        for sv in seed_variations:
            assert(isinstance(sv, SeedVariation))
        self.seed_variations = seed_variations

        self.scale = scale
        self.strength = strength
        self.steps = steps

        assert(input_mask is None or isinstance(input_mask, InputImage))
        self.input_mask = input_mask

        assert(isinstance(masks, list))
        for mask in masks:
            assert(isinstance(mask, Mask))
        # For now you can only mask pre-existing frames.
        if (len(masks) > 0):
            assert(input_image.in_video_mode())
            # And we don't combine with a pre-existing mask.
            assert(self.input_mask is None)
        self.masks = masks

        assert(fill_mask is None or isinstance(fill_mask, MaskFillImage))
        self.fill_mask = fill_mask

        assert(transform is None or isinstance(transform, Transform2D))
        self.transform = transform

        assert(correct_colors in [True, False])
        self.correct_colors = correct_colors

        assert(set_color_reference in [True, False])
        self.set_color_reference = set_color_reference

    @classmethod
    def from_dict(cls, dict):
        """This method is only used to process the first keyframe."""

        # By default we start at frame 1.
        if "frame" in dict:
            frame = int(dict["frame"])
        else:
            frame = None

        if "input_image" not in dict or dict["input_image"] == "none":
            input_image = None
        else:
            input_image = InputImage.from_string(dict["input_image"], dict["frame"])

        assert("prompt" in dict)
        prompt = dict["prompt"]

        if "prompt_weight" in dict:
            assert(float(dict["prompt_weight"]) == 1.0)

        seed = dict["seed"]
        # This means we want a new seed every frame.
        if seed == "random":
            seed = None
        else:
            seed = int(seed)

        # Note: I considered allowing seed variations, but the problem is that interpolating between
        # keyframes then requires interpolating between interpolations. This interface ensures you
        # can only ask for things that are representable by one layer of interpolation. (The same
        # applies for prompts.)
        if "seed_weight" in dict:
            assert(float(dict["seed_weight"]) == 1.0)

        scale = float(dict["scale"]) if "scale" in dict else 7.5
        strength = float(dict["strength"]) if "strength" in dict else 0.0
        steps = int(dict["steps"]) if "steps" in dict else 50

        if "input_mask" not in dict or dict["input_mask"] == "none":
            input_mask = None
        else:
            input_mask = InputImage.from_string(dict["input_mask"], dict["frame"])

        if "masks" not in dict or len(dict["masks"]) == 0:
            masks = []
        else:
            masks = [Mask(**mask) for mask in dict["masks"]]

        # You can't mask paste on the first frame.
        if "fill_mask" in dict:
            assert(dict["fill_mask"] == "none")
        fill_mask = None

        if "transform" not in dict or dict["transform"] == "none":
            transform = None
        else:
            transform = Transform2D(
                dict["transform"]["zoom"],
                dict["transform"]["translate"],
                dict["transform"]["rotate"],
            )

        # The first frame has no color reference (except perhaps itself, which would do nothing).
        if "correct_colors" in dict:
            assert(dict["correct_colors"] is False)
        correct_colors = False

        if "set_color_reference" not in dict:
            set_color_reference = False
        else:
            set_color_reference = True

        return KeyFrame(
            frame=frame,
            input_image=input_image,
            prompt=prompt,
            prompt_variations=[],
            seed=seed,
            seed_variations=[],
            scale=scale,
            strength=strength,
            steps=steps,
            input_mask=input_mask,
            masks=masks,
            fill_mask=fill_mask,
            transform=transform,
            correct_colors=correct_colors,
            set_color_reference=set_color_reference
        )

    @classmethod
    def from_dict_and_previous_keyframe(cls, dict, prev_keyframe):
        # This method is used to process all but the first keyframe.

        # Every keyframe must have an absolute frame number or a duration.
        if "frame" in dict:
            frame = int(dict["frame"])
        else:
            if "duration" in dict:
                frame = prev_keyframe.frame + int(dict["duration"])
            else:
                raise ValueError("Keyframe must have either a frame or duration")

        # The input image defaults to that of the previous keyframe. (But if you put the frame
        # number in brackets e.g. "IM{0001}.png" then it will be incremented each frame.)
        if "input_image" not in dict:
            input_image = prev_keyframe.input_image
        elif dict["input_image"] == "none":
            input_image = None
        else:
            input_image = InputImage.from_string(dict["input_image"], frame)

        # The prompt defaults to that of the previous keyframe.
        if "prompt" not in dict or dict["prompt"] == "same":
            prompt = prev_keyframe.prompt
        else:
            prompt = dict["prompt"]

        # The prompt weight defaults to 1.0.
        if "prompt_weight" in dict:
            if dict["prompt_weight"] == "same":
                prompt_weight = prev_keyframe.prompt_weight
            else:
                prompt_weight = float(dict["prompt_weight"])
                assert(0.0 <= prompt_weight <= 1.0)
        else:
            prompt_weight = 1.0

        # Prompt variations accumulate until we use a prompt with weight 1.0.
        if prompt_weight == 1.0:
            prompt_variations = []
        else:
            prompt_variations = [variation for variation in prev_keyframe.prompt_variations]
            # If we've haven't added a new prompt, we may need to update the weight of the previous one.
            if len(prompt_variations) > 0 and prompt == prompt_variations[-1].prompt:
                last_variation = prompt_variations.pop()
            prompt_variations.append(PromptVariation(prompt, None, prompt_weight))

        # The base prompt only updates if the prompt weight is 1.0.
        if prompt_weight == 1.0:
            base_prompt = prompt
        else:
            base_prompt = prev_keyframe.prompt

        # The seed defaults to that of the previous keyframe.
        # The special value "random" means to generate a new (deterministic) seed for each frame.
        if "seed" not in dict or dict["seed"] == "same":
            seed = prev_keyframe.seed
        else:
            if dict["seed"] == "random":
                seed = None
            else:
                seed = int(dict["seed"])

        # The seed weight defaults to 1.0.
        if "seed_weight" in dict:
            if dict["seed_weight"] == "same":
                seed_weight = prev_keyframe.seed_weight
            else:
                assert(seed is not None), "Random seeds cannot have a seed_weight"
                seed_weight = float(dict["seed_weight"])
                assert(0.0 <= seed_weight <= 1.0)
        else:
            seed_weight = 1.0

        # Seed variations accumulate until we use a seed with weight 1.0.
        if seed_weight == 1.0:
            seed_variations = []
        else:
            seed_variations = [variation for variation in prev_keyframe.seed_variations]
            # If we've haven't added a new seed, we may need to update the weight of the previous one.
            if len(seed_variations) > 0 and seed == seed_variations[-1].seed:
                last_variation = seed_variations.pop()
            seed_variations.append(SeedVariation(seed, seed_weight))

        # The base seed only updates if the seed weight is 1.0.
        if seed_weight == 1.0:
            base_seed = seed
        else:
            base_seed = prev_keyframe.seed

        # Scale, strength, and steps default to the values from the previous keyframe.
        if "scale" not in dict or dict["scale"] == "same":
            dict["scale"] = prev_keyframe.scale
        if "strength" not in dict or dict["strength"] == "same":
            dict["strength"] = prev_keyframe.strength
        elif dict["strength"] == "interpolate":
            dict["strength"] = None
        if "steps" not in dict or dict["steps"] == "same":
            dict["steps"] = prev_keyframe.steps

        # The input mask defaults to that of the previous keyframe. (But if you put the frame
        # number in brackets e.g. "mask_{0001}.png" then it will be incremented each frame.)
        if "input_mask" not in dict:
            input_mask = prev_keyframe.input_mask
        elif dict["input_mask"] == "none":
            input_mask = None
        else:
            input_mask = InputImage.from_string(dict["input_mask"], frame)

        # Masks default to those used in the last keyframe.
        if "masks" not in dict or dict["masks"] == "same":
            dict["masks"] = prev_keyframe.masks
        else:
            dict["masks"] = [Mask(**mask) for mask in dict["masks"]]

        # Mask fill defaults to the value from the previous keyframe.
        if "fill_mask" not in dict or dict["fill_mask"] == "same":
            fill_mask = prev_keyframe.fill_mask
        elif dict["fill_mask"] == "none":
            fill_mask = None
        else:
            fill_mask = MaskFillImage.from_string(dict["fill_mask"], frame)

        # The active transform (applied to each frame) defaults to that of the previous keyframe.
        if "transform" not in dict or dict["transform"] == "same":
            transform = prev_keyframe.transform
        elif dict["transform"] == "none":
            transform = None
        else:
            transform = Transform2D(
                dict["transform"]["zoom"],
                dict["transform"]["translate"],
                dict["transform"]["rotate"],
            )

        # Color correction defaults to that of the previous keyframe.
        if "correct_colors" not in dict or dict["correct_colors"] == "same":
            correct_colors = prev_keyframe.correct_colors
        else:
            correct_colors = dict["correct_colors"]

        # This indicates the current frame should be used as a color reference.
        # It defaults to False.
        if "set_color_reference" not in dict:
            set_color_reference = False
        else:
            set_color_reference = True

        return KeyFrame(
            frame=frame,
            input_image=input_image,
            prompt=base_prompt,
            prompt_variations=prompt_variations,
            seed=base_seed,
            seed_variations=seed_variations,
            scale=dict["scale"],
            strength=dict["strength"],
            steps=dict["steps"],
            input_mask=input_mask,
            masks=dict["masks"],
            fill_mask=fill_mask,
            transform=transform,
            correct_colors=correct_colors,
            set_color_reference=set_color_reference,
        )

    def __str__(self):
        result = f"KeyFrame {self.frame}: ({self.input_image}) \"{self.prompt}\""
        if self.prompt_idx is not None:
            result += f" (idx {self.prompt_idx})"
        result +=f" with {len(self.prompt_variations)} variations"
        result +=f", seed {self.seed}, {len(self.seed_variations)} variations, scale {self.scale}, strength {self.strength}, {len(self.masks)} masks"
        return result


class DreamSchedule:
    __slots__ = ["in_dir", "out_dir", "mask_in_dir", "width", "height", "prompts", "keyframes", "mask_fill_frames"]

    def __init__(self, in_dir, out_dir, mask_in_dir, width, height, named_prompts, keyframes):
        self.in_dir = Path(in_dir)
        self.out_dir = Path(out_dir)
        self.mask_in_dir = Path(mask_in_dir)
        self.keyframes = keyframes
        self.width = width
        self.height = height

        # Check that the keyframes are in order.
        assert(len(self.keyframes) >= 1)
        keyframe_frames = [keyframe.frame for keyframe in self.keyframes]
        assert(keyframe_frames == sorted(keyframe_frames))

        # Interpolate strengths where relevant.
        self._interpolate_strengths()

        # Collect all anonymous prompts.
        anonymous_prompts = {keyframe.prompt for keyframe in self.keyframes}
        anonymous_prompts = [*anonymous_prompts]

        # Collect all prompts in a single list.
        prompt_to_idx = {}
        all_prompts = []
        for name, prompt in named_prompts.items():
            prompt_to_idx[name] = len(all_prompts)
            prompt_to_idx[prompt] = len(all_prompts)
            all_prompts.append(prompt)
        for prompt in anonymous_prompts:
            # If this prompt is a name, it's not actually a prompt.
            if prompt in named_prompts:
                continue
            prompt_to_idx[prompt] = len(all_prompts)
            all_prompts.append(prompt)
        self.prompts = all_prompts

        # Convert prompt strings to indices.
        for keyframe in self.keyframes:
            assert keyframe.prompt in prompt_to_idx
            keyframe.prompt_idx = prompt_to_idx[keyframe.prompt]
            for variation in keyframe.prompt_variations:
                assert variation.prompt in prompt_to_idx
                variation.prompt_idx = prompt_to_idx[variation.prompt]

        # Collect all outputs used as mask fills (other than by referencing the previous frame).
        self.mask_fill_frames = set()
        for keyframe in self.keyframes:
            # Kinda hacky, I'm referencing what could be private state.
            if keyframe.fill_mask is not None and not keyframe.fill_mask.use_prev:
                self.mask_fill_frames.add(keyframe.fill_mask.frame)

    def _interpolate_strengths(self):
        assert(self.keyframes[0].strength is not None)
        assert(self.keyframes[-1].strength is not None)
        keyframe_idx = 0

        # We could add the condition "while keyframe_idx < len(self.keyframes)",
        # but the first statement in the loop checks this anyway.
        while True:
            # Look for a keyframe that doesn't have an assigned strength.
            while (keyframe_idx < len(self.keyframes) and self.keyframes[keyframe_idx].strength is not None):
                keyframe_idx += 1

            # If there are no more keyframes that don't have assigned strenghts, we're done.
            if keyframe_idx >= len(self.keyframes):
                return

            # Remember the last keyframe that did have a strength.
            prev_keyframe_idx = keyframe_idx - 1

            # Find the next keyframe that does have an assigned strength.
            # We know there is one because we asserted that the last keyframe has an assigned strength.
            keyframe_idx += 1
            while (self.keyframes[keyframe_idx].strength is None):
                keyframe_idx += 1

            # Interpolate strength between the two keyframes.
            prev_keyframe = self.keyframes[prev_keyframe_idx]
            next_keyframe = self.keyframes[keyframe_idx]
            n_frames = float(next_keyframe.frame - prev_keyframe.frame)
            for i in range(prev_keyframe_idx + 1, keyframe_idx):
                t = (self.keyframes[i].frame - prev_keyframe.frame) / n_frames
                self.keyframes[i].strength = (1.0 - t) * prev_keyframe.strength + t * next_keyframe.strength

            keyframe_idx += 1

    @classmethod
    def from_dict(cls, data):
        in_dir = data["in_dir"]
        out_dir = data["out_dir"]
        mask_in_dir = data["mask_in_dir"] if "mask_in_dir" in data else None
        width = data["width"]
        height = data["height"]
        named_prompts = data["prompts"]

        schedule = []

        # Parse the first keyframe. This keyframe only depends on its TOML data.
        schedule.append(KeyFrame.from_dict(data["keyframes"][0]))

        # Parse all other keyframes. Each depends on its TOML data and the previous keyframe.
        for keyframe_dict in data["keyframes"][1:]:
            schedule.append(KeyFrame.from_dict_and_previous_keyframe(keyframe_dict, schedule[-1]))

        return DreamSchedule(in_dir, out_dir, mask_in_dir, width, height, named_prompts, schedule)

    @classmethod
    def from_file(cls, config_path):
        with open(config_path, "r") as f:
            data = toml.load(f)
        return cls.from_dict(data)

    def prompt_command(self):
        quoted_prompts = '"' + '" "'.join(self.prompts) + '"'
        return f"!set_prompts {quoted_prompts}"

    def frames(self):
        return DreamState(self)

    def print(self):
        print(f"in_dir: {self.in_dir}")
        print(f"out_dir: {self.out_dir}")
        print(f"width: {self.width}")
        print(f"height: {self.height}")
        for keyframe in self.keyframes:
            print(keyframe)
            for sv in keyframe.seed_variations:
                print(f"  {sv}")
            for mask in keyframe.masks:
                print(f"  {mask}")


if __name__ == "__main__":
    schedule = DreamSchedule.from_file("example_animation.toml")
    schedule.print()
