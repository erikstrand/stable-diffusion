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


class FrameName:
    def __init__(self, name_beginning, name_ending, n_digits):
        self.name_beginning = name_beginning
        self.name_ending = name_ending
        self.n_digits = n_digits

    def filename(self, frame_number):
        if self.n_digits is not None:
            return self.name_beginning + str(frame_number).zfill(self.n_digits) + self.name_ending
        else:
            return self.name_beginning

    @classmethod
    def from_string(cls, string):
        re_res = re.search(r"%(\d+)", string)
        if re_res is None:
            return cls(string, None, None)
        else:
            n_digits = int(re_res.group(1))
            span = re_res.span()
            name_beginning = string[:span[0]]
            name_ending = string[span[1]:]
            return cls(name_beginning, name_ending, n_digits)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-c",
        "--copy_dir",
        type=str,
        help="The path to a directory containing frames from which mask regions will be copied.",
        required=True
    )
    parser.add_argument(
        "-p",
        "--paste_dir",
        type=str,
        help="The path to a directory containing frames into which mask regions will be pasted.",
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
        "-o",
        "--out_dir",
        type=str,
        help="Where to store the manipulated frames",
        required=True
    )
    parser.add_argument(
        "--copy_pattern",
        type=str,
        help="The names of the input frame files, with the number of digits after 'percent'",
        # e.g. frame_%6.png (default) or IM%4.jpg
        # (I can't put % in the help string or argparse gets confused)
        default="frame_%6.png"
    )
    parser.add_argument(
        "--paste_pattern",
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
        "--invert_masks",
        help="Whether to invert the mask before using it.",
        action="store_true",
        default=False
    )
    parser.add_argument(
        "-s",
        "--start_at",
        type=int,
        help="The first paste frame to render (1-indexed).",
        required=True
    )
    parser.add_argument(
        "--copy_start",
        type=int,
        help="The first copy frame to render (1-indexed). Defaults to start_at if not specified.",
        default=None
    )
    parser.add_argument(
        "--mask_start",
        type=int,
        help="The first mask frame to render (1-indexed). Defaults to start_at if not specified.",
        default=None
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
    copy_dir = Path(args.copy_dir)
    paste_dir = Path(args.paste_dir)
    mask_dir = Path(args.mask_dir)
    out_dir = Path(args.out_dir)
    assert(copy_dir.exists())
    assert(paste_dir.exists())
    assert(mask_dir.exists())
    if not out_dir.exists():
        print(f"Creating output directory {out_dir}")
        out_dir.mkdir(parents=True)
    assert(out_dir.exists())

    # Parse the filename patterns.
    copy_pattern = FrameName.from_string(args.copy_pattern)
    paste_pattern = FrameName.from_string(args.paste_pattern)
    mask_pattern = FrameName.from_string(args.mask_pattern)
    out_pattern = copy_pattern

    # Generate file lists.
    n_frames = args.end_at - args.start_at + 1
    copy_start = args.copy_start if args.copy_start is not None else args.start_at
    copy_end = copy_start + n_frames
    mask_start = args.mask_start if args.mask_start is not None else args.start_at
    mask_end = mask_start + n_frames
    copy_frames = [
        str(copy_dir / copy_pattern.filename(i))
        for i in range(copy_start, copy_end, args.stride)
    ]
    paste_frames = [
        str(paste_dir / paste_pattern.filename(i))
        for i in range(args.start_at, args.end_at + 1, args.stride)
    ]
    mask_frames = [
        str(mask_dir / mask_pattern.filename(i))
        for i in range(mask_start, mask_end, args.stride)
    ]
    outputs = [
        str(out_dir / out_pattern.filename(i))
        for i in range(args.start_at, args.end_at + 1, args.stride)
    ]

    # Paste the background image into the masked region of each frame.
    for i in range(len(copy_frames)):
        # Print progress.
        print(f"Copying from {copy_frames[i]} into {paste_frames[i]} with mask {mask_frames[i]} and saving to {outputs[i]}")

        # Load the images.
        copy_pil = Image.open(copy_frames[i])
        paste_pil = Image.open(paste_frames[i])
        mask_pil = Image.open(mask_frames[i])

        # Scale the mask if necessary.
        if copy_pil.width != mask_pil.width or copy_pil.height != mask_pil.height:
            x_scale = copy_pil.width / mask_pil.width
            y_scale = copy_pil.height / mask_pil.height
            if x_scale == y_scale and float(int(x_scale)) == x_scale:
                print(f"Resizing mask from {mask_pil.width}x{mask_pil.height} to {copy_pil.width}x{copy_pil.height}")
                mask_pil = mask_pil.resize((copy_pil.width, copy_pil.height), resample=Image.BICUBIC)
            else:
                print(f"Cannot resize mask from {mask_pil.width}x{mask_pil.height} to {copy_pil.width}x{copy_pil.height} (x_scale={x_scale}, y_scale={y_scale})")
                assert(False)

        # Convert to numpy arrays.
        copy_np = image_to_array(copy_pil)
        paste_np = image_to_array(paste_pil)
        mask_np = image_to_array(mask_pil)

        # We expect the images to be RGB (with optional ignored alpha).
        assert(len(copy_np.shape) >= 3)
        assert(len(paste_np.shape) >= 3)
        # The copy and paste images must be the same size.
        assert(copy_np.shape[0:2] == paste_np.shape[0:2])
        alpha_channel = len(mask_np.shape) - 1

        if args.invert_masks:
            print("inverting mask")
            mask_np[:, :, alpha_channel] = 255 - mask_np[:, :, alpha_channel]

        # copy/paste
        alpha = mask_np[:, :, alpha_channel].reshape(mask_np.shape[0], mask_np.shape[1], 1) / 255
        paste_np[:,:,0:3] = alpha * paste_np[:,:,0:3] + (1 - alpha) * copy_np[:,:,0:3]

        # Save the result.
        paste_pil = array_to_image(paste_np)
        paste_pil.save(outputs[i])
