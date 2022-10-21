import argparse
from dream_schedule import DreamSchedule
from dream_state import DreamState

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file")
    parser.add_argument("command_file")
    args = parser.parse_args()

    schedule = DreamSchedule.from_file(args.config_file)
    dream_state = DreamState(schedule)

    # TODO fix variations
    # TODO allow starting from a specific frame

    with open(args.command_file, "w") as outfile:
        prompt_command = schedule.prompt_command()
        outfile.write(prompt_command + '\n')
        while not dream_state.done():
            command = dream_state.get_command()
            outfile.write(command + '\n')
            dream_state.advance_frame()
