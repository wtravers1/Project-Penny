"""
# TODO ask about using flags vs prompting user
Entry point for the Humble-Nishiyama game simulation.

Run from the project root with exactly one of these flags:

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
MAX_N_DECKS = 10_000_000     # upper limit on decks added in one run


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
    Read the command-line flags. Exactly one of --show or --add is required.
    """
    parser = argparse.ArgumentParser(
        description="Simulate the Humble-Nishiyama game and display the results."
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--show",
        action="store_true",
        help="redraw the heatmaps from the results already processed",
    )
    action.add_argument(
        "--add",
        nargs="?",                # the number after --add is optional...
        const=DEFAULT_N_DECKS,    # ...and defaults to 1,000,000 if left out
        type=parse_n_decks,
        metavar="N",
        help=f"generate N new decks, score them, and update the heatmaps "
             f"(default {DEFAULT_N_DECKS:,}, max {MAX_N_DECKS:,})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.show:
        generate_heatmaps()

    else:
        generate_batch(args.add)       # create the new decks
        process_new_batches()          # score only the decks not scored yet
        generate_heatmaps()            # redraw the heatmaps for every deck so far


if __name__ == "__main__":
    main()
