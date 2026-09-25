"""
Score saved decks of cards and turn the results into heatmap-ready probabilities.

For every deck, all 28 pairs of 3-card sequences are played against each other
(28 unordered pairs cover all 56 ordered matchups in the heatmap). Each pair is
scored two ways:
    "tricks" -- the original Humble-Nishiyama game: +1 for each pile won
    "cards"  -- Ron's variation: the number of cards in the piles won

The rules of a single game:
    Cards are dealt one at a time. As soon as the last three cards dealt match
    one player's sequence, that player wins the pile (every card dealt since the
    last pile was won). Dealing then continues with a fresh window, so the three
    cards that won a pile cannot help win the next one. Cards left over at the
    end of the deck belong to nobody.

Two files are written to data/processed/:
    counts.json         running totals of wins/losses/ties, plus the list of
                        deck files already scored. Counts can be added together,
                        so new batches are scored once and never re-scored.
    probabilities.json  the presentation-ready percentages read by datavis.py.

This module is not meant to be run on its own. main.py calls process_new_batches().
All paths are relative to the project root, so run main.py from there.
"""

import json
from itertools import combinations
from pathlib import Path

import numpy as np

from src.datagen import BLACK, N_CARDS, PATH_DECKS, RED, load_deck_log, load_decks

PATH_PROCESSED = Path("data/processed")
PATH_COUNTS = PATH_PROCESSED / "counts.json"
PATH_PROBABILITIES = PATH_PROCESSED / "probabilities.json"

# Row/column order of the heatmap, matching the sample figure
SEQUENCES = ["BBB", "BBR", "BRB", "BRR", "RBB", "RBR", "RRB", "RRR"]
SCORING_VERSIONS = ("tricks", "cards")

CARDS_IN_SEQUENCE = 3
COLOR_CODES = {"B": BLACK, "R": RED}


# ---------------------------------------------------------------------------
# Turning sequences and decks into numbers
# ---------------------------------------------------------------------------


def sequence_to_code(sequence: str) -> int:
    """
    Turn a 3-card sequence such as "BRR" into a single number from 0 to 7.

    The three cards are read as binary digits (B = 0, R = 1), so
    "BBB" -> 0, "BBR" -> 1, "BRB" -> 2, ... "RRR" -> 7. Comparing one
    number is much faster than comparing three cards.
    """
    code = 0
    for letter in sequence:
        code = code * 2 + COLOR_CODES[letter]
    return code


def deck_window_codes(decks: np.ndarray) -> np.ndarray:
    """
    Turn an (n_decks, 52) array of cards into an (n_decks, 50) array of codes,
    where column k is the code of the 3-card window starting at card k.

    Window k covers cards k, k+1, k+2, so there are 52 - 2 = 50 windows.
    """
    first_cards = decks[:, :-2].astype(np.uint8)
    second_cards = decks[:, 1:-1].astype(np.uint8)
    third_cards = decks[:, 2:].astype(np.uint8)
    return first_cards * 4 + second_cards * 2 + third_cards


# ---------------------------------------------------------------------------
# Playing the game
# ---------------------------------------------------------------------------


def play_pair(window_codes: np.ndarray, code_i: int, code_j: int) -> tuple:
    """
    Play one pair of sequences against each other across every deck at once.

    Every deck is at the same position in the deal at the same time, so the
    state of each game (pile size, tricks won, cards won) is kept in an array
    with one entry per deck, and one pass over the 52 card positions plays
    every game in step.

    Returns four arrays, one entry per deck:
        (tricks_i, tricks_j, cards_i, cards_j)
    """
    n_decks = window_codes.shape[0]

    pile = np.zeros(n_decks, dtype=np.int16)  # cards dealt since the last pile was won
    fresh = np.zeros(n_decks, dtype=np.int16)  # cards dealt since the window reset
    tricks_i = np.zeros(n_decks, dtype=np.int16)
    tricks_j = np.zeros(n_decks, dtype=np.int16)
    cards_i = np.zeros(n_decks, dtype=np.int16)
    cards_j = np.zeros(n_decks, dtype=np.int16)

    for card_index in range(N_CARDS):
        pile += 1
        fresh += 1

        # The window ending on this card starts two cards earlier
        window_index = card_index - (CARDS_IN_SEQUENCE - 1)
        if window_index < 0:
            continue  # fewer than three cards dealt so far

        window = window_codes[:, window_index]
        # A pile can only be won once three cards have been dealt since the reset
        complete = fresh >= CARDS_IN_SEQUENCE
        won_i = complete & (window == code_i)
        won_j = complete & (window == code_j)

        tricks_i += won_i
        tricks_j += won_j
        cards_i += pile * won_i
        cards_j += pile * won_j

        # Start a new pile and a fresh window wherever a pile was just won
        won = won_i | won_j
        pile[won] = 0
        fresh[won] = 0

    return tricks_i, tricks_j, cards_i, cards_j


def play_pair_one_deck(deck: np.ndarray, code_i: int, code_j: int) -> tuple:
    """
    Play one pair of sequences on a single deck, one card at a time.

    This is the plain-English version of the rules, kept as a readable
    reference and used by the tests to check that play_pair() (which plays
    every deck at once) produces the same results.

    Returns (tricks_i, tricks_j, cards_i, cards_j) for this deck.
    """
    pile = 0
    fresh = 0
    tricks_i = tricks_j = cards_i = cards_j = 0

    for card_index in range(len(deck)):
        pile += 1
        fresh += 1
        if fresh < CARDS_IN_SEQUENCE:
            continue

        window = (
            deck[card_index - 2] * 4 + deck[card_index - 1] * 2 + deck[card_index]
        )
        if window == code_i:
            tricks_i += 1
            cards_i += pile
        elif window == code_j:
            tricks_j += 1
            cards_j += pile
        else:
            continue

        pile = 0
        fresh = 0

    return tricks_i, tricks_j, cards_i, cards_j


# ---------------------------------------------------------------------------
# Counting outcomes
# ---------------------------------------------------------------------------


def pair_key(sequence_i: str, sequence_j: str) -> str:
    """Return the key used to store one pair, e.g. "BBB_vs_BBR"."""
    return f"{sequence_i}_vs_{sequence_j}"


def tally_outcomes(scores_i: np.ndarray, scores_j: np.ndarray) -> dict:
    """
    Count how many decks each player won and how many ended in a tie.

    Counts (rather than percentages) are stored because counts from separate
    batches can simply be added together.
    """
    return {
        "wins_i": int((scores_i > scores_j).sum()),
        "wins_j": int((scores_j > scores_i).sum()),
        "ties": int((scores_i == scores_j).sum()),
    }


def score_decks(decks: np.ndarray) -> dict:
    """
    Score one batch of decks for all 28 pairs and both scoring versions.

    Returns {"tricks": {pair_key: outcome_counts}, "cards": {...}}.
    """
    window_codes = deck_window_codes(decks)
    counts = {version: {} for version in SCORING_VERSIONS}

    for sequence_i, sequence_j in combinations(SEQUENCES, 2):
        tricks_i, tricks_j, cards_i, cards_j = play_pair(
            window_codes, sequence_to_code(sequence_i), sequence_to_code(sequence_j)
        )
        key = pair_key(sequence_i, sequence_j)
        counts["tricks"][key] = tally_outcomes(tricks_i, tricks_j)
        counts["cards"][key] = tally_outcomes(cards_i, cards_j)

    return counts


# ---------------------------------------------------------------------------
# Running totals
# ---------------------------------------------------------------------------


def empty_counts() -> dict:
    """Return the starting totals used before any decks have been scored."""
    return {
        "scored_files": [],
        "n_decks": 0,
        "tricks": {},
        "cards": {},
    }


def load_counts() -> dict:
    """Return the running totals saved so far, or empty totals if none exist."""
    if not PATH_COUNTS.exists():
        return empty_counts()
    with PATH_COUNTS.open("r") as f:
        return json.load(f)


def add_counts(totals: dict, batch_counts: dict, filename: str, n_decks: int) -> dict:
    """
    Add one batch's counts to the running totals and return the updated totals.

    Adding counts is what makes new decks cheap: old batches are never re-scored.
    """
    for version in SCORING_VERSIONS:
        for key, outcome in batch_counts[version].items():
            running = totals[version].setdefault(
                key, {"wins_i": 0, "wins_j": 0, "ties": 0}
            )
            for outcome_name, count in outcome.items():
                running[outcome_name] += count

    totals["scored_files"].append(filename)
    totals["n_decks"] += n_decks
    return totals


def save_counts(totals: dict) -> None:
    """Write the running totals to data/processed/counts.json."""
    PATH_PROCESSED.mkdir(parents=True, exist_ok=True)
    with PATH_COUNTS.open("w") as f:
        json.dump(totals, f, indent=2)


# ---------------------------------------------------------------------------
# Presentation-ready probabilities
# ---------------------------------------------------------------------------


def write_probabilities(totals: dict) -> Path:
    """
    Turn the running counts into the percentages the heatmaps display and
    write them to data/processed/probabilities.json.

    Both the exact fractions (for the cell colors) and the rounded whole
    percentages (for the cell labels) are stored, so no arithmetic is left
    for the figure code to do.
    """
    n_decks = totals["n_decks"]
    probabilities = {"n_decks": n_decks, "sequence_order": SEQUENCES}

    for version in SCORING_VERSIONS:
        probabilities[version] = {}
        for key, outcome in totals[version].items():
            p_i = outcome["wins_i"] / n_decks
            p_j = outcome["wins_j"] / n_decks
            p_tie = outcome["ties"] / n_decks
            probabilities[version][key] = {
                "p_i_beats_j": p_i,
                "p_j_beats_i": p_j,
                "p_tie": p_tie,
                "win_pct_i": round(p_i * 100),
                "win_pct_j": round(p_j * 100),
                "tie_pct": round(p_tie * 100),
            }

    PATH_PROCESSED.mkdir(parents=True, exist_ok=True)
    with PATH_PROBABILITIES.open("w") as f:
        json.dump(probabilities, f, indent=2)
    return PATH_PROBABILITIES


# ---------------------------------------------------------------------------
# Top-level entry point used by main.py
# ---------------------------------------------------------------------------


def process_new_batches() -> None:
    """
    Score every deck file that has not been scored yet, add its results to the
    running totals, and rewrite the probabilities used by the heatmaps.
    """
    totals = load_counts()
    already_scored = set(totals["scored_files"])
    new_batches = [
        entry for entry in load_deck_log() if entry["file"] not in already_scored
    ]

    if not new_batches:
        print("No new decks to score.")
        return

    for entry in new_batches:
        decks = load_decks(PATH_DECKS / entry["file"])
        print(f"Scoring {entry['n_decks']:,} decks from {entry['file']}...")
        totals = add_counts(
            totals, score_decks(decks), entry["file"], entry["n_decks"]
        )

    save_counts(totals)
    write_probabilities(totals)
    print(f"Scored {len(new_batches)} new batch(es); totals now cover {totals['n_decks']:,} decks.")
