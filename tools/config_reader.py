import toml
import numpy as np
from pathlib import Path

# TODO
# - implement "pass" (meaning interpolation is defined by previous/later non "pass" frames)
# - implement a way to shift between image inputs and pure txt2img
# - implement 2D movement (zoom, translation, rotation)


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
    __slots__ = ["frame", "prompt", "seed", "seed_variations", "scale", "strength", "masks", "animation"]

    def __init__(self, frame, prompt, seed, variations, scale, strength, masks, animation):
        self.frame = frame
        self.prompt = prompt
        self.seed = seed

        assert(isinstance(variations, list))
        for sv in variations:
            assert(isinstance(sv, SeedVariation))
        self.seed_variations = variations

        self.scale = scale
        self.strength = strength

        assert(isinstance(masks, list))
        for mask in masks:
            assert(isinstance(mask, Mask))
        self.masks = masks

        assert(animation is None or isinstance(animation, Animation2D))
        self.animation = animation

    @classmethod
    def from_dict(cls, dict):
        assert("frame" in dict)
        assert("prompt" in dict)
        assert("seed" in dict)

        if not "seed_variations" in dict or len(dict["seed_variations"]) == 0:
            variations = []
        else:
            variations = [SeedVariation(**sv) for sv in dict["seed_variations"]]

        scale = float(dict["scale"]) if "scale" in dict else 7.5
        strength = float(dict["strength"]) if "strength" in dict else 0.0

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

        return KeyFrame(
            int(dict["frame"]),
            dict["prompt"],
            int(dict["seed"]),
            variations,
            scale,
            strength,
            masks,
            animation,
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
            reset_variations = False
        else:
            seed = int(dict["seed"])
            reset_variations = True

        if "seed_variations" in dict:
            if dict["seed_variations"] == "same":
                variations = prev_keyframe.seed_variations
            else:
                variations = [SeedVariation(**sv) for sv in dict["seed_variations"]]
        else:
            if reset_variations:
                variations = []
            else:
                variations = prev_keyframe.seed_variations

        if "scale" not in dict or dict["scale"] == "same":
            dict["scale"] = prev_keyframe.scale

        if "strength" not in dict or dict["strength"] == "same":
            dict["strength"] = prev_keyframe.strength

        if "masks" not in dict or dict["masks"] == "same":
            dict["masks"] = prev_keyframe.masks
        else:
            dict["masks"] = [Mask(**mask) for mask in dict["masks"]]

        if "animation" not in dict or dict["animation"] == "same":
            dict["animation"] = prev_keyframe.animation
        elif dict["animation"] == "none":
            print(dict["animation"])
            dict["animation"] = None
        else:
            print(dict["animation"])
            dict["animation"] = Animation2D(
                dict["animation"]["zoom"],
                dict["animation"]["translate"],
                dict["animation"]["rotate"],
            )

        return KeyFrame(
            dict["frame"],
            dict["prompt"],
            seed,
            variations,
            dict["scale"],
            dict["strength"],
            dict["masks"],
            dict["animation"],
        )

    def __str__(self):
        return f"KeyFrame {self.frame}: prompt {self.prompt}, seed {self.seed}, scale {self.scale}, strength {self.strength}, {len(self.masks)} masks"


class DreamSchedule:
    __slots__ = ["indir", "maskdir", "outdir", "keyframes", "width", "height", "stride", "prompts"]

    def __init__(self, indir, maskdir, outdir, keyframes, width, height, stride):
        self.indir = Path(indir)
        self.maskdir = Path(maskdir)
        self.outdir = Path(outdir)
        self.keyframes = keyframes
        self.width = width
        self.height = height
        self.stride = int(stride)

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
        print(f"indir: {self.indir}")
        print(f"outdir: {self.outdir}")
        print(f"maskdir: {self.maskdir}")
        print(f"width: {self.width}")
        print(f"height: {self.height}")
        print(f"stride: {self.stride}")
        for keyframe in self.keyframes:
            print(keyframe)
            for mask in keyframe.masks:
                print(f"  {mask}")


def load_config(config_path):
    with open(config_path, "r") as f:
        data = toml.load(f)

    indir = data["indir"]
    maskdir = data["maskdir"]
    outdir = data["outdir"]
    width = data["width"]
    height = data["height"]
    stride = data["stride"]

    schedule = []
    schedule.append(KeyFrame.from_dict(data["keyframes"][0]))

    for keyframe_dict in data["keyframes"][1:]:
        schedule.append(KeyFrame.from_dict_and_previous_keyframe(keyframe_dict, schedule[-1]))

    return DreamSchedule(indir, maskdir, outdir, schedule, width, height, stride)


if __name__ == "__main__":
    schedule = load_config("config_example.toml")
    schedule.print()
