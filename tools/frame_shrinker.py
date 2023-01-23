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
        "--input_pattern",
        type=str,
        help="The names of the input frame files, with the number of digits after 'percent'",
        # e.g. frame_%6.png (default) or IM%4.jpg
        # (I can't put % in the help string or argparse gets confused)
        default="frame_%6.png",
        required=True
    )
    parser.add_argument(
        "--output_pattern",
        type=str,
        help="The names of the output frame files, with the number of digits after 'percent'",
        # e.g. frame_%6.png (default) or IM%4.jpg
        # (I can't put % in the help string or argparse gets confused)
        default=None,
        required=False
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
    parser.add_argument(
        "-w",
        "--width",
        type=int,
        help="The desired width.",
        default=None
    )
    parser.add_argument(
        "--height",
        type=int,
        help="The desired height. (Can't use -h cause that's for --help.)",
        default=None
    )

    args = parser.parse_args()

    # Ensure at least one of --width or --height is specified.
    assert(args.width is not None or args.height is not None)

    # Parse paths.
    in_dir = Path(args.in_dir)
    assert(in_dir.exists())
    out_dir = Path(args.out_dir)
    if in_dir == out_dir:
        assert(args.input_pattern != args.output_pattern)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Parse the filename patterns.
    input_pattern = FrameName.from_string(args.input_pattern)
    if args.output_pattern is None:
        output_pattern = input_pattern
    else:
        output_pattern = FrameName.from_string(args.output_pattern)

    # Generate file lists.
    input_files = [
        str(in_dir / input_pattern.filename(i))
        for i in range(args.start_at, args.end_at + 1, args.stride)
    ]
    output_files = [
        str(out_dir / output_pattern.filename(i))
        for i in range(args.start_at, args.end_at + 1, args.stride)
    ]

    # Resize frames.
    for i in range(len(input_files)):
        # Load the frame.
        frame_pil = Image.open(input_files[i])
        frame_np = image_to_array(frame_pil)

        # Make sure the input frame is at least as big as the desired size.
        in_width = frame_np.shape[0]
        in_height = frame_np.shape[1]
        if args.width is not None:
            width = args.width
        else:
            width = in_width
        if args.height is not None:
            height = args.height
        else:
            height = in_height
        assert(in_width >= width)
        assert(in_height >= height)
        print(f"{input_files[i]}: {in_width}x{in_height} --> {output_files[i]}: {width}x{height}")

        # Generate the canvas.
        n_channels = frame_np.shape[2]
        result_np = np.zeros((width, height, n_channels), dtype=frame_np.dtype)

        # Paste the frame in.
        x_offset = (in_width - width) // 2
        y_offset = (in_height - height) // 2
        result_np = frame_np[x_offset:x_offset+width, y_offset:y_offset+height, :]

        # Save the result.
        result_pil = array_to_image(result_np)
        result_pil.save(output_files[i])
