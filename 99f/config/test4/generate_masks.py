from config_reader import load_config
from pathlib import Path
from masks import save_mask_image

width = 640
height = 360

if __name__ == "__main__":
    # load the dream schedule
    ds = load_config("config_05.json")

    indir = Path(ds.indir)
    maskdir = Path(ds.maskdir)
    maskdir.mkdir(exist_ok=True)

    prev_keyframe = None
    next_keyframe = ds.schedule[0]
    next_keyframe_idx = 1

    masks = []
    while next_keyframe_idx < len(ds.schedule):
        prev_keyframe = next_keyframe
        next_keyframe = ds.schedule[next_keyframe_idx]
        next_keyframe_idx += 1
        interp_len = float(next_keyframe.frame - prev_keyframe.frame)

        for frame_idx in range(prev_keyframe.frame, next_keyframe.frame, ds.stride):
            t = float(frame_idx - prev_keyframe.frame) / interp_len
            interp_masks = []
            for prev_mask, next_mask in zip(prev_keyframe.masks, next_keyframe.masks):
                center = (1.0 - t) * prev_mask.center + t * next_mask.center
                radius = (1.0 - t) * prev_mask.radius + t * next_mask.radius
                interp_masks.append((center, radius))

            if len(interp_masks) > 0:
                image_file = indir / f"IM{frame_idx:05d}.jpg"
                mask_file = maskdir / f"IM{frame_idx:05d}.png"
                print(f"generating mask for frame {frame_idx} ({mask_file})")
                save_mask_image(width, height, interp_masks, image_file, mask_file)
