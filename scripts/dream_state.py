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

    def done(self):
        return self.next_keyframe_idx >= len(self.schedule.keyframes)

    def get_opts(self):
        t = float(self.frame_idx - self.prev_keyframe.frame) / self.interp_duration
        strength = (1.0 - t) * self.prev_keyframe.strength + t * self.next_keyframe.strength

        command_data.append(CommandData(
            prompt0=prev_keyframe.prompt,
            prompt1=next_keyframe.prompt,
            seed0=prev_keyframe.seed,
            seed1=next_keyframe.seed,
            t=t,
            strength=strength,
            image=f"{frame_idx:06d}.0.png",
            mask=(f"{frame_idx:06d}.0.png" if n_masks > 0 else None)
        ))
