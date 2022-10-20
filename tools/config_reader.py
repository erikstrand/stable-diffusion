import toml
import numpy as np
from pathlib import Path
from dream_state import DreamState

# TODO
# - implement "pass" (meaning interpolation is defined by previous/later non "pass" frames)
# - implement a way to shift between image inputs and pure txt2img


class SeedVariation:
    __slots__ = ["seed", "amount"]

    def __init__(self, seed, amount):
        self.seed = int(seed)
        self.amount = float(amount)

    def __str__(self):
        return f"Variation: seed {self.seed}, amount {self.amount:.5f}"


class Mask:
    __slots__ = ["center", "radius"]

    def __init__(self, center, radius):
        self.center = np.array(center)
        assert(self.center.shape == (2,))
        self.radius = float(radius)

    def __str__(self):
        return f"Mask: center {self.center[0]}, {self.center[1]}, radius {self.radius}"


class Animation2D:
    __slots__ = ["zoom", "translation", "rotation"]

    def __init__(self, zoom, translation, rotation):
        self.zoom = float(zoom)
        self.translation = (float(translation[0]), float(translation[1]))
        self.rotation = float(rotation)


class KeyFrame:
    __slots__ = [
        "frame",
        "prompt",
        "seed",
        "seed_variations",
        "scale",
        "strength",
        "steps",
        "masks",
        "animation",
        "correct_colors",
        "set_color_reference"
    ]

    def __init__(
        self,
        frame,
        prompt,
        seed,
        seed_variations,
        scale,
        strength,
        steps,
        masks,
        animation,
        correct_colors,
        set_color_reference
    ):
        self.frame = frame
        self.prompt = prompt
        self.seed = seed

        assert(isinstance(seed_variations, list))
        for sv in seed_variations:
            assert(isinstance(sv, SeedVariation))
        self.seed_variations = seed_variations

        self.scale = scale
        self.strength = strength
        self.steps = steps

        assert(isinstance(masks, list))
        for mask in masks:
            assert(isinstance(mask, Mask))
        self.masks = masks

        assert(animation is None or isinstance(animation, Animation2D))
        self.animation = animation

        assert(correct_colors in [True, False])
        self.correct_colors = correct_colors

        assert(set_color_reference in [True, False])
        self.set_color_reference = set_color_reference

    @classmethod
    def from_dict(cls, dict):
        assert("frame" in dict)
        assert("prompt" in dict)

        seed = dict["seed"]
        # This means we want a new seed every frame.
        if seed == "random":
            seed = None
        else:
            seed = int(seed)

        if "seed_weight" in dict:
            assert(dict["seed_weight"] == 1.0)

        scale = float(dict["scale"]) if "scale" in dict else 7.5
        strength = float(dict["strength"]) if "strength" in dict else 0.0
        steps = int(dict["steps"]) if "steps" in dict else 50

        if "masks" not in dict or len(dict["masks"]) == 0:
            masks = []
        else:
            masks = [Mask(**mask) for mask in dict["masks"]]

        if "animation" not in dict or dict["animation"] == "none":
            animation = None
        else:
            animation = Animation2D(
                dict["animation"]["zoom"],
                dict["animation"]["translate"],
                dict["animation"]["rotate"],
            )

        if "correct_colors" not in dict:
            correct_colors = False
        else:
            correct_colors = True

        if "set_color_reference" not in dict:
            set_color_reference = False
        else:
            set_color_reference = True

        return KeyFrame(
            int(dict["frame"]),
            dict["prompt"],
            seed,
            [],
            scale,
            strength,
            steps,
            masks,
            animation,
            correct_colors,
            set_color_reference
        )

    @classmethod
    def from_dict_and_previous_keyframe(cls, dict, prev_keyframe):
        if "frame" not in dict:
            if "duration" in dict:
                dict["frame"] = prev_keyframe.frame + dict["duration"]
            else:
                raise ValueError("Keyframe must have either a frame or duration")

        if "prompt" not in dict or dict["prompt"] == "same":
            dict["prompt"] = prev_keyframe.prompt

        if "seed" not in dict or dict["seed"] == "same":
            seed = prev_keyframe.seed
        else:
            if dict["seed"] == "random":
                seed = None
            else:
                seed = int(dict["seed"])

        if "seed_weight" in dict:
            assert seed is not None, "Random seeds cannot have a seed_weight"
            seed_weight = float(dict["seed_weight"])
            assert(0.0 <= seed_weight <= 1.0)
        else:
            seed_weight = 1.0

        if seed_weight == 1.0:
            seed_variations = []
        else:
            seed_variations = [variation for variation in prev_keyframe.seed_variations]
            if seed != prev_keyframe.seed:
                seed_variations.append(SeedVariation(seed, seed_weight))

        if "scale" not in dict or dict["scale"] == "same":
            dict["scale"] = prev_keyframe.scale

        if "strength" not in dict or dict["strength"] == "same":
            dict["strength"] = prev_keyframe.strength

        if "steps" not in dict or dict["steps"] == "same":
            dict["steps"] = prev_keyframe.steps

        if "masks" not in dict or dict["masks"] == "same":
            dict["masks"] = prev_keyframe.masks
        else:
            dict["masks"] = [Mask(**mask) for mask in dict["masks"]]

        if "animation" not in dict or dict["animation"] == "same":
            animation = prev_keyframe.animation
        elif dict["animation"] == "none":
            animation = None
        else:
            animation = Animation2D(
                dict["animation"]["zoom"],
                dict["animation"]["translate"],
                dict["animation"]["rotate"],
            )

        # correct_colors defaults to the same value as the last keyframe
        if "correct_colors" not in dict or dict["correct_colors"] == "same":
            correct_colors = prev_keyframe.correct_colors
        else:
            correct_colors = True

        # set_color_reference is never interpolated
        if "set_color_reference" not in dict:
            set_color_reference = False
        else:
            set_color_reference = True

        return KeyFrame(
            dict["frame"],
            dict["prompt"],
            seed,
            seed_variations,
            dict["scale"],
            dict["strength"],
            dict["steps"],
            dict["masks"],
            animation,
            correct_colors,
            set_color_reference,
        )

    def __str__(self):
        return f"KeyFrame {self.frame}: prompt {self.prompt}, seed {self.seed}, {len(self.seed_variations)} variations, scale {self.scale}, strength {self.strength}, {len(self.masks)} masks"


class DreamSchedule:

    __slots__ = ["in_dir", "mask_dir", "out_dir", "keyframes", "width", "height", "stride","restart_from", "prompts"]

    def __init__(self, in_dir, mask_dir, out_dir, keyframes, width, height, stride,restart_from):

        self.in_dir = Path(in_dir)
        self.mask_dir = Path(mask_dir)
        self.out_dir = Path(out_dir)
        self.keyframes = keyframes
        self.width = width
        self.height = height
        self.stride = int(stride)
        self.restart_from = restart_from

        # Check that the keyframes are in order.
        assert(len(self.keyframes) >= 1)
        keyframe_frames = [keyframe.frame for keyframe in self.keyframes]
        assert(keyframe_frames == sorted(keyframe_frames))

        # Collect all the prompts.
        prompts_set = {keyframe.prompt for keyframe in self.keyframes}
        self.prompts = [*prompts_set]

        # Convert prompt strings to indices.
        for keyframe in self.keyframes:
            keyframe.prompt = self.prompts.index(keyframe.prompt)
            assert(0 <= keyframe.prompt < len(self.prompts))

    def print(self):
        print(f"in_dir: {self.in_dir}")
        print(f"out_dir: {self.out_dir}")
        print(f"mask_dir: {self.mask_dir}")
        print(f"width: {self.width}")
        print(f"height: {self.height}")
        print(f"stride: {self.stride}")
        print(f"restart_from: {self.restart_from}")
        for keyframe in self.keyframes:
            print(keyframe)
            for sv in keyframe.seed_variations:
                print(f"  {sv}")
            for mask in keyframe.masks:
                print(f"  {mask}")


def load_config(config_path):
    with open(config_path, "r") as f:
        data = toml.load(f)

    in_dir = data["in_dir"]
    mask_dir = data["mask_dir"]
    out_dir = data["out_dir"]
    width = data["width"]
    height = data["height"]
    stride = data["stride"]
    if 'restart_from' in data:
        restart_from = data["restart_from"]
    else:
        restart_from = None

    schedule = []
    schedule.append(KeyFrame.from_dict(data["keyframes"][0]))

    for keyframe_dict in data["keyframes"][1:]:
        schedule.append(KeyFrame.from_dict_and_previous_keyframe(keyframe_dict, schedule[-1]))


    return DreamSchedule(in_dir, mask_dir, out_dir, schedule, width, height, stride, restart_from)


if __name__ == "__main__":
    schedule = load_config("aversion/liquid_chrome_dreams/config.toml")
    schedule.print()

    dream_state = DreamState(schedule)
    while not dream_state.done():
        command = dream_state.get_command()
        print(command)
        dream_state.advance_frame()
