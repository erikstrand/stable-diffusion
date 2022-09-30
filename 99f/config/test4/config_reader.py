import json
import numpy as np
from pathlib import Path


class Mask:
    __slots__ = ["center", "radius"]

    def __init__(self, center, radius):
        self.center = np.array(center)
        assert(self.center.shape == (2,))
        self.radius = float(radius)

    def __str__(self):
        return f"Mask: center {self.center[0]}, {self.center[1]}, radius {self.radius}"


class KeyFrame:
    __slots__ = ["frame", "prompt", "seed", "strength", "masks"]

    def __init__(self, frame, prompt, seed, strength, masks):
        self.frame = frame
        self.prompt = prompt
        self.seed = seed
        self.strength = strength
        self.masks = [Mask(**mask) for mask in masks]

    def __str__(self):
        return f"KeyFrame {self.frame}: prompt {self.prompt}, seed {self.seed}, strength {self.strength}, {len(self.masks)} masks"


class DreamSchedule:
    __slots__ = ["indir", "maskdir", "outdir", "schedule", "stride"]

    def __init__(self, indir, maskdir, outdir, schedule, stride):
        self.indir = Path(indir)
        self.maskdir = Path(maskdir)
        self.outdir = Path(outdir)
        self.schedule = schedule
        self.stride = int(stride)

        assert(len(self.schedule) >= 1)
        keyframe_frames = [keyframe.frame for keyframe in self.schedule]
        assert(keyframe_frames == sorted(keyframe_frames))

    def print(self):
        print(f"indir: {self.indir}")
        print(f"outdir: {self.outdir}")
        print(f"maskdir: {self.maskdir}")
        print(f"stride: {self.stride}")
        for keyframe in self.schedule:
            print(keyframe)
            for mask in keyframe.masks:
                print(f"  {mask}")


def load_config(config_path):
    with open(config_path, "r") as f:
        data = json.load(f)

    indir = data["indir"]
    maskdir = data["maskdir"]
    outdir = data["outdir"]
    stride = data["stride"]
    schedule = [KeyFrame(**frame) for frame in data["schedule"]]
    return DreamSchedule(indir, maskdir, outdir, schedule, stride)


if __name__ == "__main__":
    schedule = load_config("config_01.json")
    schedule.print()
