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
        default=None,
    )
    parser.add_argument(
        "-e",
        "--end_at",
        type=int,
        help="The last frame to render (1-indexed).",
        default=None,
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
        help="The names of the input frames, with a '#' in place of the frame number, e.g. frame_#.png (default) or IM#.jpg",
        default="frame_#.png"
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

    # Parse the input filename pattern.
    path_components = args.input_pattern.split("#")
    if len(path_components) != 2:
        print("Invalid input pattern, you must include exactly one '#' to indicate where the frame number appears")
        exit(0)
    path_start = path_components[0]
    path_end = path_components[1]
    path_start_len = len(path_start)
    path_end_len = len(path_end)

    # Get a list of filenames in the output directory.
    outdir = Path(args.dir)
    files = [
        f for f in outdir.iterdir()
        if f.is_file() and f.name.startswith(path_start) and f.name.endswith(path_end)
    ]
    if len(files) == 0:
        print("Found no files matching the specified pattern.")
        exit(0)

    # Extract the frame numbers from the filenames.
    frames = []
    for idx, f in enumerate(files):
        frame = int(f.name[path_start_len:-path_end_len])
        frames.append((idx, frame))

    # Check for missing frames.
    frames = sorted(frames, key=lambda x: x[1])
    first_frame = frames[0][1]
    last_frame = frames[-1][1]
    missing_frames = last_frame - first_frame + 1 - len(frames)
    print(f"Found {len(frames)} frames ({first_frame}-{last_frame}).")
    if missing_frames > 0:
        print(f"Warning: Missing {missing_frames} frames.")

    # Sort filenames.
    files = [files[file_tuple[0]] for file_tuple in frames]

    # Write the filenames to a file.
    filenames = [str(f) for f in files]
    frame_duration = 1.0 / args.framerate
    with open('frames.txt', 'w') as outfile:
        for filename in filenames:
            outfile.write(f"file '{filename}'\n")
            outfile.write(f"duration {frame_duration}\n")

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
