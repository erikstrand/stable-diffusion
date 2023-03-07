import argparse
import numpy as np
import re
from PIL import Image
from pathlib import Path


def image_to_array(image):
    # convert a PIL image to a numpy array
    image_array = np.array(image)
    return np.swapaxes(np.flip(image_array, 0), 0, 1)


def array_to_image(array):
    # convert a numpy array to a PIL image
    array = np.flip(np.swapaxes(array, 0, 1), 0)
    format = 'RGB' if array.shape[2] == 3 else 'RGBA'
    return Image.fromarray(array, format)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--in_dir",
        type=str,
        help="The path to a directory containing frames.",
        required=True
    )
    parser.add_argument(
        "-o",
        "--out_dir",
        type=str,
        help="Where to store the manipulated frames",
        required=True
    )
    parser.add_argument(
        "-m",
        "--mask_dir",
        type=str,
        help="The path to a directory containing masks.",
        required=True
    )
    parser.add_argument(
        "-b",
        "--background_image",
        type=str,
        help="The path to the file that should be pasted in the masked regions of each frame.",
        required=True
    )
    parser.add_argument(
        "--frame_pattern",
        type=str,
        help="The names of the input frame files, with the number of digits after 'percent'",
        # e.g. frame_%6.png (default) or IM%4.jpg
        # (I can't put % in the help string or argparse gets confused)
        default="frame_%6.png"
    )
    parser.add_argument(
        "--mask_pattern",
        type=str,
        help="The names of the mask files, with the number of digits after 'percent'",
        # e.g. frame_%6.png (default) or IM%4.jpg
        # (I can't put % in the help string or argparse gets confused)
        default="frame_%6.png"
    )
    parser.add_argument(
        "-s",
        "--start_at",
        type=int,
        help="The first frame to render (1-indexed).",
        required=True
    )
    parser.add_argument(
        "-e",
        "--end_at",
        type=int,
        help="The last frame to render (1-indexed).",
        required=True
    )
    parser.add_argument(
        "--stride",
        type=int,
        help="How much to increment the frame counter each step. If stride is 2, every other frame is included.",
        default=1
    )

    args = parser.parse_args()

    # Parse paths.
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    mask_dir = Path(args.mask_dir)
    background_image = Path(args.background_image)
    assert(in_dir.exists())
    assert(out_dir.exists())
    assert(mask_dir.exists())
    assert(background_image.exists())

    # Parse the input filename pattern.
    re_res = re.search(r"%(\d+)", args.frame_pattern)
    if re_res is None:
        print("Invalid input pattern, you must include '%' followed by a number to indicate the format of the frame number")
        exit(0)
    else:
        frame_n_digits = int(re_res.group(1))
        span = re_res.span()
        frame_file_start = args.frame_pattern[:span[0]]
        frame_file_end = args.frame_pattern[span[1]:]

    # Parse the mask filename pattern.
    re_res = re.search(r"%(\d+)", args.mask_pattern)
    if re_res is None:
        static_mask = True
        mask_file = args.mask_pattern
    else:
        static_mask = False
        mask_n_digits = int(re_res.group(1))
        span = re_res.span()
        mask_file_start = args.mask_pattern[:span[0]]
        mask_file_end = args.mask_pattern[span[1]:]

    # Generate file lists.
    frames = [
        str(in_dir / f"{frame_file_start}{str(i).zfill(frame_n_digits)}{frame_file_end}")
        for i in range(args.start_at, args.end_at + 1, args.stride)
    ]
    if static_mask:
        masks = [ str(mask_dir / mask_file) for i in range(args.start_at, args.end_at + 1, args.stride) ]
    else:
        masks = [
            str(mask_dir / f"{mask_file_start}{str(i).zfill(mask_n_digits)}{mask_file_end}")
            for i in range(args.start_at, args.end_at + 1, args.stride)
        ]
    outputs = [
        str(out_dir / f"{frame_file_start}{str(i).zfill(frame_n_digits)}{frame_file_end}")
        for i in range(args.start_at, args.end_at + 1, args.stride)
    ]

    # Load the background image.
    background_pil = Image.open(background_image)
    background_np = image_to_array(background_pil)

    # Paste the background image into the masked region of each frame.
    for i in range(len(frames)):
        # Print progress.
        print(f"Processing {frames[i]}")

        # Load the frame and mask.
        frame_pil = Image.open(frames[i])
        mask_pil = Image.open(masks[i])
        frame_np = image_to_array(frame_pil)
        mask_np = image_to_array(mask_pil)

        # The frame, mask, and backgruond image must be the same size.
        assert(frame_np.shape[0:2] == mask_np.shape[0:2])
        assert(frame_np.shape[0:2] == background_np.shape[0:2])
        # The mask must have an alpha channel.
        assert(len(mask_np.shape) == 3)
        assert(mask_np.shape[2] == 4)

        # Paste the background image into the masked region of the frame.
        alpha = mask_np[:, :, 3].reshape(mask_np.shape[0], mask_np.shape[1], 1) / 255
        frame_np[:,:,0:3] = alpha * frame_np[:,:,0:3] + (1 - alpha) * background_np[:,:,0:3]

        # Save the result.
        frame_pil = array_to_image(frame_np)
        frame_pil.save(outputs[i])
