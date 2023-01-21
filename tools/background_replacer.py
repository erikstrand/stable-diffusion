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
    copy_dir = Path(args.copy_dir)
    paste_dir = Path(args.paste_dir)
    mask_dir = Path(args.mask_dir)
    out_dir = Path(args.out_dir)
    assert(copy_dir.exists())
    assert(paste_dir.exists())
    assert(mask_dir.exists())
    assert(out_dir.exists())

    # Parse the filename patterns.
    copy_pattern = FrameName.from_string(args.copy_pattern)
    paste_pattern = FrameName.from_string(args.paste_pattern)
    mask_pattern = FrameName.from_string(args.mask_pattern)
    out_pattern = copy_pattern

    # Generate file lists.
    copy_frames = [
        str(copy_dir / copy_pattern.filename(i))
        for i in range(args.start_at, args.end_at + 1, args.stride)
    ]
    paste_frames = [
        str(paste_dir / paste_pattern.filename(i))
        for i in range(args.start_at, args.end_at + 1, args.stride)
    ]
    mask_frames = [
        str(mask_dir / mask_pattern.filename(i))
        for i in range(args.start_at, args.end_at + 1, args.stride)
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

        # Convert to numpy arrays.
        copy_np = image_to_array(copy_pil)
        paste_np = image_to_array(paste_pil)
        mask_np = image_to_array(mask_pil)

        if args.invert_masks:
            print("inverting mask")
            mask_np[:, :, 3] = 255 - mask_np[:, :, 3]

        # The copy, paste, and mask images must be the same size.
        assert(copy_np.shape[0:2] == paste_np.shape[0:2])
        assert(copy_np.shape[0:2] == mask_np.shape[0:2])

        # The mask must have an alpha channel.
        assert(len(mask_np.shape) == 3)
        assert(mask_np.shape[2] == 4)

        # copy/paste
        alpha = mask_np[:, :, 3].reshape(mask_np.shape[0], mask_np.shape[1], 1) / 255
        paste_np[:,:,0:3] = alpha * paste_np[:,:,0:3] + (1 - alpha) * copy_np[:,:,0:3]

        # Save the result.
        paste_pil = array_to_image(paste_np)
        paste_pil.save(outputs[i])
