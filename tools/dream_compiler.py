import argparse
import subprocess
from pathlib import Path
from dream_schedule import DreamSchedule
from dream_state import DreamState
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
        default=1
    )
    parser.add_argument(
        "-e",
        "--end_at",
        type=int,
        help="The last frame to render (1-indexed).",
        default=None
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

    # Initialize the dream state.
    if args.end_at is None:
        # Minus two since we don't output the last keyframe, and we're 1-indexed.
        args.end_at = schedule.keyframes[-1].frame - 2
    dream_state = DreamState(schedule)
    frames = []
    outfile = open(args.outfile, "w") if args.commands else None

    # If we're writing commands, we always write the prompts.
    if args.commands:
        prompt_command = schedule.prompt_command()
        outfile.write(prompt_command + '\n')

    # Iterate over all frames.
    n_commands = 0
    while not dream_state.done() and dream_state.frame_idx <= args.end_at:
        # We generate the command even if we're not writing it anywhere so that random seeds are the
        # same no matter where we start.
        command = dream_state.get_command()
        frames.append(dream_state.output_path())

        if args.start_at <= dream_state.frame_idx:
            # Write the command if requested.
            if args.commands:
                outfile.write(command + '\n')
                n_commands += 1

            # Generate the mask if requested.
            if args.masks:
                masks = dream_state.get_masks()
                if len(masks) > 0:
                    image_file = dream_state.input_image_path()
                    mask_file = dream_state.mask_path()
                    print(f"generating mask for frame {dream_state.frame_idx} ({mask_file})")
                    save_mask_image(
                        schedule.width,
                        schedule.height,
                        masks,
                        image_file,
                        mask_file
                    )

        dream_state.advance_frame()

    # Print a summary if we wrote commands.
    if args.commands:
        print(f"Wrote {n_commands} commands to {args.outfile} (frames {args.start_at} to {args.end_at}).")

    if args.video:
        # Write the list of frames to a file.
        assert(args.config_file.endswith(".toml"))
        frame_file = args.config_file[:-5] + "_frames.txt"
        config_dir = Path(args.config_file).resolve().parent
        config_delta = config_dir.relative_to(Path.cwd())
        n_dirs = len(config_delta.parts)
        prefix = Path("../" * n_dirs)
        with open(frame_file, 'w') as outfile:
            for frame in frames:
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
