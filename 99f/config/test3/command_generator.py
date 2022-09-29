import json
from pathlib import Path


class KeyFrame:
    __slots__ = ["frame", "prompt", "seed", "strength"]

    def __init__(self, frame, prompt, seed, strength):
        self.frame = frame
        self.prompt = prompt
        self.seed = seed
        self.strength = strength

    def __str__(self):
        return f"KeyFrame {self.frame}: prompt {self.prompt}, seed {self.seed}, strength {self.strength}"


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
        command = f"-M {maskdir/self.mask}"
        command += f" --latent_0 {self.prompt0} --latent_1 {self.prompt1} -u {self.t:.5f}"
        command += f" -S {self.seed0} -V {self.seed1}:{self.t:.5f}"
        # when f is literally 0.0, it gets thrown out and I think effectively uses 1.0
        command += f" -f {max(0.001, self.strength)}"
        command += " -W640 -H320 -e"
        command += f" --outdir {outdir}"
        return command

    def __str__(self):
        return f"CommandData: prompt0 {self.prompt0}, prompt1 {self.prompt1}, seed0 {self.seed0}, seed1 {self.seed1}, t {self.t}, strength {self.strength}, image {self.image}"


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

    def generate_command_data(self):
        prev_keyframe = None
        next_keyframe = self.schedule[0]
        next_keyframe_idx = 1

        command_data = []
        while next_keyframe_idx < len(self.schedule):
            prev_keyframe = next_keyframe
            next_keyframe = self.schedule[next_keyframe_idx]
            next_keyframe_idx += 1
            interp_len = float(next_keyframe.frame - prev_keyframe.frame)

            for frame_idx in range(prev_keyframe.frame, next_keyframe.frame, self.stride):
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
                    mask=f"IM{frame_idx:05d}.jpg"
                ))

        # Add the last frame
        command_data.append(CommandData(
            prompt0=next_keyframe.prompt,
            prompt1=next_keyframe.prompt,
            seed0=next_keyframe.seed,
            seed1=next_keyframe.seed,
            t=0.0,
            strength=next_keyframe.strength,
            image=f"IM{next_keyframe.frame:05d}.jpg"
        ))

        return command_data


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
    dream_configs = []
    #dream_configs.append(load_config("config_00.json"))
    dream_configs.append(load_config("config_01.json"))

    commands = []
    for dream_config in dream_configs:
        id = dream_config.indir
        od = dream_config.outdir
        command_data = dream_config.generate_command_data()
        commands += [command.generate_command_string(id, od) for command in command_data]

    with open("commands.txt", "w") as f:
        f.write("\n".join(commands))
