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
        if masks is None:
            self.masks = []
        else:
            assert(isinstance(masks, list))
            if len(masks) == 0 or isinstance(masks[0], Mask):
                self.masks = masks
            else:
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

    schedule = []

    kwargs = data["schedule"][0]
    assert("frame" in kwargs)
    assert("prompt" in kwargs)
    assert("seed" in kwargs)
    if not "strenth" in kwargs:
        kwargs["strength"] = 0.0
    if "masks" not in kwargs:
        kwargs["masks"] = []
    schedule.append(KeyFrame(**kwargs))

    for kwargs in data["schedule"][1:]:
        __slots__ = ["frame", "prompt", "seed", "strength", "masks"]
        assert("frame" in kwargs)
        if "prompt" not in kwargs:
            kwargs["prompt"] = schedule[-1].prompt
        if "seed" not in kwargs:
            kwargs["seed"] = schedule[-1].seed
        if "strength" not in kwargs:
            kwargs["strength"] = schedule[-1].strength
        if "masks" not in kwargs:
            kwargs["masks"] = schedule[-1].masks
        schedule.append(KeyFrame(**kwargs))

    return DreamSchedule(indir, maskdir, outdir, schedule, stride)
