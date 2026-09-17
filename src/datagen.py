"""
Generate reproducible batches of shuffled card decks for the Humble-Nishiyama game.

Each deck is represented only by card color, since suit and rank do not matter:
    0 = black, 1 = red
Every deck contains exactly 26 black and 26 red cards in a random order.

Decks are created and saved in batches. Each batch:
    - uses its own unique random seed (the previous batch's seed + 1)
    - is saved to its own file in data/raw/decks/ and never edited afterward
    - is recorded in data/raw/deck_log.json (file name, seed, size, timestamp)

Run this file directly to add one new batch of decks.
"""

import json
from datetime import datetime as dt
from pathlib import Path

import numpy as np

# Paths are built from this file's location so the script works from any working directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PATH_DECKS = PROJECT_ROOT / "data" / "raw" / "decks"
PATH_DECK_LOG = PROJECT_ROOT / "data" / "raw" / "deck_log.json"

SEED_BASE = 1
CARDS_PER_COLOR = 26
N_CARDS = 52
BLACK = 0
RED = 1


def make_decks(seed: int, n_decks: int) -> np.ndarray:
    """
    Create n_decks shuffled decks using the given random seed.

    Each row is one deck: an ordered deck of 26 zeros and 26 ones, shuffled.
    Shuffling (instead of drawing random 0s and 1s) guarantees that every
    deck has exactly 26 black and 26 red cards.

    Returns an array of shape (n_decks, 52) with dtype uint8.
    """
    rng = np.random.default_rng(seed)
    ordered_deck = np.array(
        [BLACK] * CARDS_PER_COLOR + [RED] * CARDS_PER_COLOR, dtype=np.uint8
    )
    unshuffled_decks = np.tile(
        ordered_deck, (n_decks, 1)
    )  # one copy of the deck per row
    return rng.permuted(unshuffled_decks, axis=1)  # shuffle each row independently


def load_deck_log() -> list:
    """
    Return the list of batch records saved so far, or an empty list if
    no batches have been generated yet.
    """
    if not PATH_DECK_LOG.exists():
        return []
    with PATH_DECK_LOG.open("r") as f:
        return json.load(f)


def get_next_seed() -> int:
    """
    Return the seed for the next batch: one more than the last seed used,
    or SEED_BASE if no batches exist yet.

    This only reads the log. The log is updated by record_batch() after the
    batch file has been saved, so a crash partway through never makes it look
    like a seed was used when it was not.
    """
    deck_log = load_deck_log()
    if not deck_log:
        print(f"No deck log found, starting with seed {SEED_BASE}")
        return SEED_BASE
    return deck_log[-1]["seed"] + 1


def save_decks(decks: np.ndarray, seed: int) -> Path:
    """
    Save a batch of decks to data/raw/decks/ and return the file path.

    Decks are bit-packed before saving (1 bit per card instead of 1 byte),
    which shrinks 1 million decks from about 52 MB to about 7 MB.
    Use load_decks() to read them back as plain 0s and 1s.
    """
    PATH_DECKS.mkdir(parents=True, exist_ok=True)

    n_decks = decks.shape[0]
    n_cards = decks.shape[1]
    filename = PATH_DECKS / f"decks_{n_decks}x{n_cards}_seed_{seed}.npy"

    # Never overwrite an existing batch: batches are not edited after creation
    if filename.exists():
        raise FileExistsError(
            f"{filename} already exists; seed {seed} was already used"
        )

    np.save(filename, np.packbits(decks, axis=1))
    return filename


def record_batch(filename: Path, seed: int, n_decks: int) -> None:
    """
    Append a record of a saved batch to the deck log.
    """
    deck_log = load_deck_log()
    deck_log.append(
        {
            "file": filename.name,
            "seed": seed,
            "n_decks": n_decks,
            "n_cards": N_CARDS,
            "created": str(dt.now()),
        }
    )
    PATH_DECK_LOG.parent.mkdir(parents=True, exist_ok=True)
    with PATH_DECK_LOG.open("w") as f:
        json.dump(deck_log, f, indent=2)


def load_decks(filename: Path) -> np.ndarray:
    """
    Load a saved batch and unpack it back into an (n_decks, 52) array of 0s and 1s.
    """
    packed_decks = np.load(filename)
    # packbits pads 52 bits up to 56 (7 whole bytes), so drop the 4 padding bits
    return np.unpackbits(packed_decks, axis=1)[:, :N_CARDS]
    # TODO Ask why we need to upload data to GitHub if anyone running this script can generate the same numbers anyways


def generate_batch(n_decks: int) -> Path:
    """
    Full pipeline for one batch: pick the next seed, make the decks,
    save them, then record the batch in the log. Returns the saved file path.
    """
    seed = get_next_seed()
    decks = make_decks(seed, n_decks)
    filename = save_decks(decks, seed)
    record_batch(filename, seed, n_decks)
    print(f"Saved {n_decks:,} decks with seed {seed} to {filename.name}")
    return filename


if __name__ == "__main__":
    generate_batch(n_decks=1_000_000)
