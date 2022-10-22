import argparse
from dream_schedule import DreamSchedule
from dream_state import DreamState

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

    # Parse args.
    args = parser.parse_args()
    if args.outfile is None:
        assert(args.config_file.endswith(".toml"))
        args.outfile = args.config_file[:-4] + "txt"

    # Build the dream schedule.
    schedule = DreamSchedule.from_file(args.config_file)

    # Generate commands.
    if args.end_at is None:
        args.end_at = schedule.keyframes[-1].frame
    dream_state = DreamState(schedule)
    with open(args.outfile, "w") as outfile:
        # We always have to set our prompts.
        prompt_command = schedule.prompt_command()
        outfile.write(prompt_command + '\n')

        # Render the frames.
        while not dream_state.done() and dream_state.frame_idx <= args.end_at:
            # We generate the command so that random seeds are the same no matter where we start.
            command = dream_state.get_command()
            masks = dream_state.get_masks()
            #print(masks)
            if args.start_at <= dream_state.frame_idx:
                outfile.write(command + '\n')
            dream_state.advance_frame()


        """
        if len(interp_masks) > 0:
            image_file = indir / f"{frame_idx:06d}.0.png"
            mask_file = maskdir / f"{frame_idx:06d}.0.png"
            print(f"generating mask for frame {frame_idx} ({mask_file})")
            save_mask_image(width, height, interp_masks, image_file, mask_file)
        """
