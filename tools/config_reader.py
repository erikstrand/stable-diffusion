import toml
import numpy as np
from pathlib import Path

# TODO
# - implement "pass" (meaning interpolation is defined by previous/later non "pass" frames)
# - implement a way to shift between image inputs and pure txt2img
# - implement 2D movement (zoom, translation, rotation)


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
    __slots__ = ["frame", "prompt", "seed", "scale", "strength", "masks", "animation"]

    def __init__(self, frame, prompt, seed, scale, strength, masks, animation):
        self.frame = frame
        self.prompt = prompt
        self.seed = seed
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
        if not "scale" in dict:
            dict["scale"] = 7.5
        if not "strenth" in dict:
            dict["strength"] = 0.0
        if "masks" not in dict or len(dict["masks"]) == 0:
            dict["masks"] = []
        else:
            dict["masks"] = [Mask(**mask) for mask in dict["masks"]]
        if "animation" not in dict or dict["animation"] == "none":
            dict["animation"] = None
        else:
            dict["animation"] = Animation2D(
                dict["animation"]["zoom"],
                dict["animation"]["translate"],
                dict["animation"]["rotate"],
            )

        return KeyFrame(
            dict["frame"],
            dict["prompt"],
            dict["seed"],
            dict["scale"],
            dict["strength"],
            dict["masks"],
            dict["animation"],
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
            dict["seed"] = prev_keyframe.seed

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
            dict["seed"],
            dict["scale"],
            dict["strength"],
            dict["masks"],
            dict["animation"],
        )

    def __str__(self):
        return f"KeyFrame {self.frame}: prompt {self.prompt}, seed {self.seed}, scale {self.scale}, strength {self.strength}, {len(self.masks)} masks"


class DreamSchedule:
    __slots__ = ["indir", "maskdir", "outdir", "schedule", "width", "height", "stride"]

    def __init__(self, indir, maskdir, outdir, schedule, width, height, stride):
        self.indir = Path(indir)
        self.maskdir = Path(maskdir)
        self.outdir = Path(outdir)
        self.schedule = schedule
        self.width = width
        self.height = height
        self.stride = int(stride)

        assert(len(self.schedule) >= 1)
        keyframe_frames = [keyframe.frame for keyframe in self.schedule]
        assert(keyframe_frames == sorted(keyframe_frames))

    def print(self):
        print(f"indir: {self.indir}")
        print(f"outdir: {self.outdir}")
        print(f"maskdir: {self.maskdir}")
        print(f"width: {self.width}")
        print(f"height: {self.height}")
        print(f"stride: {self.stride}")
        for keyframe in self.schedule:
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
