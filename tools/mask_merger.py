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
        "-i",
        "--input_dir_1",
        type=str,
        help="The path to a directory containing masks.",
        required=True
    )
    parser.add_argument(
        "-j",
        "--input_dir_2",
        type=str,
        help="The path to a directory containing masks.",
        required=True
    )
    parser.add_argument(
        "-o",
        "--out_dir",
        type=str,
        help="Where to store the merged masks",
        required=True
    )
    parser.add_argument(
        "--pattern_1",
        type=str,
        help="The names of the masks in input_dir_1, with the number of digits after 'percent'",
        # e.g. frame_%6.png (default) or IM%4.jpg
        # (I can't put % in the help string or argparse gets confused)
        default="frame_%6.png"
    )
    parser.add_argument(
        "--pattern_2",
        type=str,
        help="The names of the masks in input_dir_2, with the number of digits after 'percent'",
        # e.g. frame_%6.png (default) or IM%4.jpg
        # (I can't put % in the help string or argparse gets confused)
        default="frame_%6.png"
    )
    parser.add_argument(
        "--output_pattern",
        type=str,
        help="The names of the generated files, with the number of digits after 'percent'. Defaults to pattern_1.",
        # e.g. frame_%6.png (default) or IM%4.jpg
        # (I can't put % in the help string or argparse gets confused)
        default=None
    )
    parser.add_argument(
        "--invert_1",
        help="Whether to invert the masks in input_dir_1.",
        action="store_true",
        default=False
    )
    parser.add_argument(
        "--invert_2",
        help="Whether to invert the masks in input_dir_2.",
        action="store_true",
        default=False
    )
    parser.add_argument(
        "-s",
        "--start_1",
        type=int,
        help="The first frame to render from input_dir_1.",
        required=True
    )
    parser.add_argument(
        "--start_2",
        type=int,
        help="The first frame to render from input_dir_2. Defaults to start_at if not specified.",
        default=None
    )
    parser.add_argument(
        "-e",
        "--end_at",
        type=int,
        help="The last frame to render.",
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
    in_dir_1 = Path(args.input_dir_1)
    in_dir_2 = Path(args.input_dir_2)
    out_dir = Path(args.out_dir)
    assert(in_dir_1.exists())
    assert(in_dir_2.exists())
    if not out_dir.exists():
        print(f"Creating output directory {out_dir}")
        out_dir.mkdir(parents=True)
    assert(out_dir.exists())

    # Parse the filename patterns.
    in_pattern_1 = FrameName.from_string(args.pattern_1)
    in_pattern_2 = FrameName.from_string(args.pattern_2)
    if args.output_pattern is None:
        out_pattern = in_pattern_1
    else:
        out_pattern = FrameName.from_string(args.output_pattern)

    # Generate file lists.
    start_1 = args.start_1
    start_2 = args.start_2 if args.start_2 is not None else args.start_1
    end_1 = args.end_at + 1
    n_frames = args.end_at - args.start_1 + 1
    end_2 = start_2 + n_frames
    in_files_1 = [
        str(in_dir_1 / in_pattern_1.filename(i))
        for i in range(start_1, end_1, args.stride)
    ]
    in_files_2 = [
        str(in_dir_2 / in_pattern_2.filename(i))
        for i in range(start_2, end_2, args.stride)
    ]
    outputs = [
        str(out_dir / out_pattern.filename(i))
        for i in range(start_1, end_1, args.stride)
    ]

    # Merge the masks.
    for i in range(len(in_files_1)):
        # Print progress.
        print(f"Merging {in_files_1[i]} and {in_files_2[i]} and saving to {outputs[i]}")

        # Load the images.
        mask_1_pil = Image.open(in_files_1[i])
        mask_2_pil = Image.open(in_files_2[i])

        # Convert to numpy arrays.
        mask_1_np = image_to_array(mask_1_pil)
        mask_2_np = image_to_array(mask_2_pil)

        # Ensure the shapes make sense.
        assert(len(mask_1_np.shape) == 3)
        assert(len(mask_2_np.shape) == 3)
        assert(mask_1_np.shape[0:2] == mask_1_np.shape[0:2])

        # Find the alpha channels.
        alpha_1 = mask_1_np.shape[2] - 1
        alpha_2 = mask_2_np.shape[2] - 1
        #print(f"mask 1 alpha channel is {alpha_1} (shape={mask_1_np.shape})")
        #print(f"mask 1 min={np.min(mask_1_np)}, max={np.max(mask_1_np)}")

        if args.invert_1:
            print("inverting mask 1")
            mask_1_np[:, :, alpha_1] = 255 - mask_1_np[:, :, alpha_1]
        if args.invert_2:
            print("inverting mask 2")
            mask_2_np[:, :, alpha_2] = 255 - mask_2_np[:, :, alpha_2]

        # Merge
        mask_1_np[:,:,alpha_1] = np.maximum(mask_1_np[:,:,alpha_1], mask_2_np[:,:,alpha_2])

        # Save the result.
        result_pil = array_to_image(mask_1_np)
        result_pil.save(outputs[i])
