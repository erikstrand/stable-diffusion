from config_reader import load_config
from pathlib import Path
from masks import save_mask_image
import argparse

width = 640
height = 320

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

    # load the dream schedule
    ds = load_config("config_merge.json")

    indir = Path(ds.indir)
    maskdir = Path(ds.maskdir)
    maskdir.mkdir(exist_ok=True)

    prev_keyframe = None
    next_keyframe = ds.schedule[0]
    next_keyframe_idx = 1

    first_frame = args.first_frame if args.first_frame is not None else next_keyframe.frame
    last_frame = args.last_frame

    masks = []
    while next_keyframe_idx < len(ds.schedule):
        prev_keyframe = next_keyframe
        next_keyframe = ds.schedule[next_keyframe_idx]
        next_keyframe_idx += 1
        interp_len = float(next_keyframe.frame - prev_keyframe.frame)

        # Keep going if the first frame doesn't appear between these keyframes.
        if next_keyframe.frame <= first_frame:
            continue

        for frame_idx in range(prev_keyframe.frame, next_keyframe.frame, ds.stride):
            # Stop if we've past the last frame
            if last_frame is not None and frame_idx > last_frame:
                break

            t = float(frame_idx - prev_keyframe.frame) / interp_len
            interp_masks = []

            # If the next keyframe removes all masks, just use the current masks.
            # Note: This code isn't designed to handle changing the number of masks except from or
            # to zero.
            if len(prev_keyframe.masks) > 0 and len(next_keyframe.masks) == 0:
                for mask in prev_keyframe.masks:
                    interp_masks.append((mask.center, mask.radius))
            else:
                # Otherwise, interpolate.
                for prev_mask, next_mask in zip(prev_keyframe.masks, next_keyframe.masks):
                    center = (1.0 - t) * prev_mask.center + t * next_mask.center
                    radius = (1.0 - t) * prev_mask.radius + t * next_mask.radius
                    interp_masks.append((center, radius))

            if len(interp_masks) > 0:
                image_file = indir / f"{frame_idx:06d}.0.png"
                mask_file = maskdir / f"{frame_idx:06d}.0.png"
                print(f"generating mask for frame {frame_idx} ({mask_file})")
                save_mask_image(width, height, interp_masks, image_file, mask_file)