import json
import numpy as np
from config_reader import load_config
from pathlib import Path
import argparse


class CommandData:
    __slots__ = ["prompt0", "prompt1", "seed0", "seed1", "t", "strength", "image", "mask"]

    def __init__(self, prompt0, prompt1, seed0, seed1, t, strength, image, mask=None):
        self.prompt0 = prompt0
        self.prompt1 = prompt1
        self.seed0 = seed0
        self.seed1 = seed1
        self.t = t
        self.strength = strength
        self.image = image
        self.mask = mask

    def generate_command_string(self, indir, maskdir, outdir):
        command = f"-I {indir/self.image}"
        if self.mask is not None:
            command += f" -M {maskdir/self.mask}"
        command += f" --latent_0 {self.prompt0} --latent_1 {self.prompt1} -u {self.t:.5f}"
        command += f" -S {self.seed0} -V {self.seed1}:{self.t:.5f}"
        # when f is literally 0.0, it gets thrown out and I think effectively uses 1.0
        command += f" -f {max(0.001, self.strength)}"
        command += " -W640 -H320 -e"
        command += f" --outdir {outdir}"
        return command

    def __str__(self):
        return f"CommandData: prompt0 {self.prompt0}, prompt1 {self.prompt1}, seed0 {self.seed0}, seed1 {self.seed1}, t {self.t}, strength {self.strength}, image {self.image}, mask {self.mask}"


def generate_command_data(dream_schedule, first_frame, last_frame):
    prev_keyframe = None
    next_keyframe = dream_schedule.schedule[0]
    next_keyframe_idx = 1

    command_data = []
    while next_keyframe_idx < len(dream_schedule.schedule):
        prev_keyframe = next_keyframe
        next_keyframe = dream_schedule.schedule[next_keyframe_idx]
        next_keyframe_idx += 1
        interp_len = float(next_keyframe.frame - prev_keyframe.frame)
        n_masks = min(len(prev_keyframe.masks), len(next_keyframe.masks))

        # Keep going if the first frame doesn't appear between these keyframes.
        if next_keyframe.frame <= first_frame:
            continue

        for frame_idx in range(prev_keyframe.frame, next_keyframe.frame, dream_schedule.stride):
            # Stop if we've past the last frame
            if last_frame is not None and frame_idx > last_frame:
                break

            t = float(frame_idx - prev_keyframe.frame) / interp_len
            strength = (1.0 - t) * prev_keyframe.strength + t * next_keyframe.strength
            command_data.append(CommandData(
                prompt0=prev_keyframe.prompt,
                prompt1=next_keyframe.prompt,
                seed0=prev_keyframe.seed,
                seed1=next_keyframe.seed,
                t=t,
                strength=strength,
                image=f"IM{frame_idx:05d}.jpg",
                mask=(f"IM{frame_idx:05d}.png" if n_masks > 0 else None)
            ))

    # Add the last frame if needed.
    if last_frame is None or next_keyframe.frame <= last_frame:
        command_data.append(CommandData(
            prompt0=next_keyframe.prompt,
            prompt1=next_keyframe.prompt,
            seed0=next_keyframe.seed,
            seed1=next_keyframe.seed,
            t=0.0,
            strength=next_keyframe.strength,
            image=f"IM{next_keyframe.frame:05d}.jpg",
            mask=(f"IM{next_keyframe.frame:05d}.png" if n_masks > 0 else None)
        ))

    return command_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--first_frame',
        '-f',
        default=None,
        type=int,
        help='create masks starting at this frame'
    )
    parser.add_argument(
        '--last_frame',
        '-g',
        default=None,
        type=int,
        help='create masks ending after this frame'
    )
    args = parser.parse_args()

    dream_config = load_config("config.json")
    first_frame = args.first_frame if args.first_frame is not None else dream_config.schedule[0].frame
    last_frame = args.last_frame

    id = dream_config.indir
    md = dream_config.maskdir
    od = dream_config.outdir
    command_data = generate_command_data(dream_config, first_frame, last_frame)
    commands = [command.generate_command_string(id, md, od) for command in command_data]

    with open("commands.txt", "w") as f:
        f.write("\n".join(commands))
