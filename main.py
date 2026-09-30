"""
Entry point for the Humble-Nishiyama game simulation.

Run from the project root. With no flags it asks what to do:

    uv run main.py
        Prompt for one of the two options below.

Or pick an option directly with a flag:

    uv run main.py --show
        Display the most up-to-date heatmaps.

    uv run main.py --add
        Add 1,000,000 new decks (the default), score them, and update the heatmaps.

    uv run main.py --add N
        Add N new decks instead, where N is from 1 to 10,000,000.
        Commas and underscores are allowed, e.g. --add 2,500,000 or --add 2_500_000.

    uv run main.py --help
        Print this usage information.

Only the new decks are generated and scored; decks from earlier runs are never redone.
"""

import argparse

from src.datagen import generate_batch
from src.dataprocessing import process_new_batches
from src.datavis import generate_heatmaps

DEFAULT_N_DECKS = 1_000_000  # used when --add is given without a number
MAX_N_DECKS = 10_000_000  # upper limit on decks added in one run


def parse_n_decks(text: str) -> int:
    """
    Convert the number given after --add into an integer, allowing commas
    and underscores. Raises an error if it is not between 1 and MAX_N_DECKS.
    """
    cleaned = text.replace(",", "").replace("_", "")
    if not cleaned.isdigit() or not 1 <= int(cleaned) <= MAX_N_DECKS:
        raise argparse.ArgumentTypeError(
            f"must be a whole number from 1 to {MAX_N_DECKS:,} (got {text!r})"
        )
    return int(cleaned)


def parse_args() -> argparse.Namespace:
    """
    Read the command-line flags. --show and --add cannot be combined; if
    neither is given, main() asks the user which one they want instead.
    """
    parser = argparse.ArgumentParser(
        description="Simulate the Humble-Nishiyama game and display the results."
    )
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--show",
        action="store_true",
        help="redraw the heatmaps from the results already processed",
    )
    action.add_argument(
        "--add",
        nargs="?",
        const=DEFAULT_N_DECKS,
        type=parse_n_decks,
        metavar="N",
        help=f"generate N new decks, score them, and update the heatmaps "
        f"(default {DEFAULT_N_DECKS:,}, max {MAX_N_DECKS:,})",
    )
    return parser.parse_args()


def prompt_menu_choice() -> str:
    """
    Ask which of the two options the user wants and return "1" or "2".
    Used when main.py is run without any flags. Keeps asking until the
    answer is valid.
    """
    print("1. Display the most up-to-date heatmaps")
    print("2. Add more decks to the simulation")
    while True:
        choice = input("Choose an option (1 or 2): ").strip()
        if choice in ("1", "2"):
            return choice
        print("Please enter 1 or 2.")


def prompt_number_of_decks() -> int:
    """
    Ask how many decks to add and return it as a positive whole number.
    Pressing Enter without typing anything uses DEFAULT_N_DECKS.
    """
    while True:
        question = f"How many decks to add? (Enter for {DEFAULT_N_DECKS:,}): "
        answer = input(question).strip()
        if answer == "":
            return DEFAULT_N_DECKS
        try:
            return parse_n_decks(answer)
        except argparse.ArgumentTypeError as error:
            print(error)


def add_decks(n_decks: int) -> None:
    """Generate new decks, score only those decks, and redraw the heatmaps."""
    generate_batch(n_decks)  # create the new decks
    process_new_batches()  # score only the decks not scored yet
    generate_heatmaps()  # redraw the heatmaps for every deck so far


def show_heatmaps() -> None:
    """
    Redraw the heatmaps from results already processed, explaining what to
    do first if no decks have been simulated yet.
    """
    try:
        generate_heatmaps()
    except FileNotFoundError as error:
        print(error)


def main() -> None:
    args = parse_args()

    if args.show:
        show_heatmaps()
    elif args.add is not None:
        add_decks(args.add)
    else:
        # No flag was given, so ask which option the user wants
        if prompt_menu_choice() == "1":
            show_heatmaps()
        else:
            add_decks(prompt_number_of_decks())


if __name__ == "__main__":
    main()
