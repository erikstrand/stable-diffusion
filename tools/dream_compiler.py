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
            if args.start_at <= dream_state.frame_idx:
                command = dream_state.get_command()
                outfile.write(command + '\n')
            dream_state.advance_frame()
