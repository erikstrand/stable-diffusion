import argparse
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

    # Generate the list of frames, write it to a file.
    outdir = Path(args.dir)
    files = [str(outdir / f"frame_{i:06d}.png") for i in range(args.start_at, args.end_at + 1)]
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
