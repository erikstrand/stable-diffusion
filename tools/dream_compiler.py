import argparse
import subprocess
from pathlib import Path
from dream_schedule import DreamSchedule
from masks import save_mask_image

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # You always have to specify a .toml config file.
    parser.add_argument(
        "config_file",
        type=str,
        help="The path of a .toml file containing the dream schedule."
    )

    # Commands
    parser.add_argument(
        "-c",
        "--commands",
        action="store_true",
        help="Store InvokeAI commands in a text file"
    )
    parser.add_argument(
        "-m",
        "--masks",
        action="store_true",
        help="Generate masks"
    )
    parser.add_argument(
        "-v",
        "--video",
        action="store_true",
        help="Combine generated frames into an mp4 video"
    )

    # General Options
    parser.add_argument(
        "-o",
        "--outfile",
        type=str,
        help="The path of the output file. Defaults to the input file with a different extension (.txt for commands, .mp4 for videos)."
    )
    parser.add_argument(
        "-s",
        "--start_at",
        type=int,
        help="The first frame to render (1-indexed).",
        default=None
    )
    parser.add_argument(
        "-e",
        "--end_at",
        type=int,
        help="The last frame to render (1-indexed).",
        default=None
    )
    parser.add_argument(
        "--stride",
        type=int,
        help="Allows skipping frames to more quickly render rough versions (render frames 1 + n * stride).",
        default=1
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

    # Parse args.
    args = parser.parse_args()
    if args.commands is False and args.masks is False and args.video is False:
        print("No commands specified! Use -c to generate commands, -m to generate masks, or -v to generate a video.")
        exit(0)
    if args.video and (args.commands or args.masks):
        print("The video option must be used on its own.")
        exit(0)
    if args.outfile is None:
        assert(args.config_file.endswith(".toml"))
        if args.commands:
            args.outfile = args.config_file[:-4] + "txt"
        else:
            args.outfile = args.config_file[:-4] + "mp4"

    # Build the dream schedule.
    schedule = DreamSchedule.from_file(args.config_file)

    # Determine the range of frames to render.
    if args.start_at is None:
        args.start_at = schedule.keyframes[0].frame
    if args.end_at is None:
        args.end_at = schedule.keyframes[-1].frame

    # Initialize state.
    frame_names = [] # records the filenames of all generated frames
    outfile = open(args.outfile, "w") if args.commands else None

    # We write the !set_prompts command regardless of start_at and end_at.
    if args.commands:
        prompt_command = schedule.prompt_command()
        outfile.write(prompt_command + '\n')

    # Iterate over all frames.
    for frame in schedule.frames():
        if frame.frame_idx < args.start_at or (frame.frame_idx - 1) % args.stride != 0:
            continue

        if frame.frame_idx > args.end_at:
            break

        # Record this frame's filename (used for video generation).
        frame_names.append(frame.output_path())

        # Write the command if requested.
        if args.commands:
            outfile.write(frame.get_command() + '\n')

        # Generate the mask if requested.
        if args.masks:
            masks = frame.get_masks()
            if len(masks) > 0:
                image_file = frame.input_image_path()
                mask_file = frame.mask_path()
                print(f"generating mask for frame {frame.frame_idx} ({mask_file})")
                save_mask_image(
                    schedule.width,
                    schedule.height,
                    masks,
                    image_file,
                    mask_file
                )

    # Print a summary if we wrote commands.
    if args.commands:
        print(f"Wrote commands for {len(frame_names)} frames ({args.start_at} through {args.end_at}) to {args.outfile}.")

    if args.video:
        # Write the list of frames to a file.
        assert(args.config_file.endswith(".toml"))
        frame_file = args.config_file[:-5] + "_frames.txt"
        config_dir = Path(args.config_file).resolve().parent
        config_delta = config_dir.relative_to(Path.cwd())
        n_dirs = len(config_delta.parts)
        prefix = Path("../" * n_dirs)
        with open(frame_file, 'w') as outfile:
            for frame in frame_names:
                outfile.write(f"file '{prefix / frame}'\n")

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
            frame_file,
            "-vcodec",
            "libx264",
            "-crf",
            str(args.constant_rate_factor),
            "-pix_fmt",
            "yuv420p",
            args.outfile
        ]
        subprocess.run(args)
