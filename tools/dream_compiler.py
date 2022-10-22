import argparse
from dream_schedule import DreamSchedule
from dream_state import DreamState
from masks import save_mask_image

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config_file",
        type=str,
        help="The path of a .toml file containing the dream schedule."
    )
    parser.add_argument(
        "-o",
        "--outfile",
        type=str,
        help="The path of the output file. Defaults to the input file but .txt instead of .toml."
    )
    parser.add_argument(
        "-s",
        "--start_at",
        type=int,
        help="The first frame to render (1-indexed).",
        default=0
    )
    parser.add_argument(
        "-e",
        "--end_at",
        type=int,
        help="The last frame to render (1-indexed).",
        default=None
    )
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

    # Parse args.
    args = parser.parse_args()
    if args.outfile is None:
        assert(args.config_file.endswith(".toml"))
        args.outfile = args.config_file[:-4] + "txt"
    if args.commands is False and args.masks is False:
        print("No commands specified! Use -c to generate commands, and/or -m to generate masks.")
        exit(0)

    # Build the dream schedule.
    schedule = DreamSchedule.from_file(args.config_file)

    # Generate commands.
    if args.end_at is None:
        args.end_at = schedule.keyframes[-1].frame
    dream_state = DreamState(schedule)
    with open(args.outfile, "w") as outfile:
        if args.commands:
            # We have to set our prompts regardless of start and end frames.
            prompt_command = schedule.prompt_command()
            outfile.write(prompt_command + '\n')

        # Render the frames.
        while not dream_state.done() and dream_state.frame_idx <= args.end_at:
            # We always generate the command so that random seeds are the same no matter where we start.
            command = dream_state.get_command()

            if args.start_at <= dream_state.frame_idx:
                if args.commands:
                    outfile.write(command + '\n')
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
