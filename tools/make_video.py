import argparse
import re
import subprocess
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "dir",
        type=str,
        help="The path to a directory containing frames."
    )
    parser.add_argument(
        "-o",
        "--outfile",
        type=str,
        help="Where to store the generated video",
        required=True
    )

    # Deprecated (see use below)
    """
    parser.add_argument(
        "-rn",
        "--rename",
        help="Rename IMxxxx to frame_xxxxxx",
        action="store_true"
    )
    """

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
        "--input_pattern",
        type=str,
        help="The names of the input frames, with the number of digits after '%', e.g. frame_%6.png (default) or IM%4.jpg",
        default="frame_%6.png"
    )
    parser.add_argument(
        "-i",
        "--irregular",
        action="store_true",
        help="When set, only frames that exist are included.",
    )

    # Video Options
    parser.add_argument(
        "-r",
        "--framerate",
        type=float,
        help="The framerate of the output video (only used with -v).",
        default=10.0
    )
    parser.add_argument(
        "-crf",
        "--constant_rate_factor",
        type=int,
        help="The constant rate factor of the output video (0-51, lower is higher quality but larger file) (only used with -v).",
        default=23
    )

    args = parser.parse_args()

    # Parse the input filename pattern.
    re_res = re.search(r"%(\d+)", args.input_pattern)
    if re_res is None:
        print("Invalid input pattern, you must include '%' followed by a number to indicate the format of the frame number")
        exit(0)
    else:
        n_digits = int(re_res.group(1))
        span = re_res.span()
        path_start = args.input_pattern[:span[0]]
        path_end = args.input_pattern[span[1]:]

    # Deprecated for now (Jan 2023), might add back later.
    """
    if args.rename:
        outdir = Path(args.dir)
        files = [(i, outdir / f"IM{i:04d}.png") for i in range(args.start_at, args.end_at + 1, args.stride)]
        files = [(i, f) for i, f in files if f.exists()]
        for i, f in files:
            f.rename(outdir / f"frame_{i:06d}.png")
        exit(0)
    """

    # Generate the list of frames, write it to a file.
    outdir = Path(args.dir)
    files = [
        str(outdir / f"{path_start}{str(i).zfill(n_digits)}{path_end}")
        for i in range(args.start_at, args.end_at + 1, args.stride)
    ]
    if args.irregular:
        files = [f for f in files if Path(f).exists()]
    with open('frames.txt', 'w') as outfile:
        for file in files:
            outfile.write(f"file '{file}'\n")

    # Generate the video. We run an ffmpeg command like the following.
    # ffmpeg -r 12.5 -f concat -i frame_file.txt -vcodec libx264 -crf 10 -pix_fmt yuv420p video.mp4
    args = [
        "ffmpeg",
        "-r",
        str(args.framerate),
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        "frames.txt",
        "-vcodec",
        "libx264",
        "-crf",
        str(args.constant_rate_factor),
        "-pix_fmt",
        "yuv420p",
        args.outfile
    ]
    subprocess.run(args)
