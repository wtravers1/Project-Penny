import json
import shutil
from datetime import datetime as dt
from pathlib import Path
 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
 
# Paths are relative to the project root (run main.py from there)
PATH_PROBABILITIES = Path("data/processed/probabilities.json") #update with path once dataprocessing is done
PATH_FIGURES = Path("figures")
PATH_FIGURES_ARCHIVE = Path("figures/archive")
 
SCORING_VERSIONS = ("tricks", "cards")
VERSION_DISPLAY_NAMES = {
    "tricks": "Humble-Nishiyama Game (Scored by Tricks)",
    "cards": "Ron's Variation (Scored by Cards)",
}
 
 
# ---------------------------------------------------------------------------
# Loading processed data
# ---------------------------------------------------------------------------
 
def load_probabilities() -> dict:
    """
    Load the probabilities.json file produced by dataprocessing.py.
 
    Raises a clear error if the file doesn't exist yet.
    """
    if not PATH_PROBABILITIES.exists():
        raise FileNotFoundError(
            f"No processed probabilities found at {PATH_PROBABILITIES}. "
            "Generate and process some decks first (see main.py)."
        )
    with PATH_PROBABILITIES.open("r") as f:
        return json.load(f)
 
 
# ---------------------------------------------------------------------------
# Building the 8x8 matrices for one scoring version
# ---------------------------------------------------------------------------
 
def get_pair_probability(
    pair_dict: dict,
    row_seq: str,
    col_seq: str,
    sequence_order: list[str],
) -> tuple[float, int]:
    """
    Look up the win probability and tie count for one cell of the
    heatmap: P(row_seq beats col_seq), plus their tie count.
 
    Pairs are stored unordered in pair_dict, under a key built from
    whichever of the two sequences comes first in sequence_order (that
    one is "i", the other is "j"). This function figures out which
    direction was stored, fetches the right entry, and returns the
    probability from row_seq's perspective regardless of which side of
    the pair row_seq happened to be.
 
    Args:
        pair_dict: probabilities[version] -- the dict of all 28 pairs
            for one scoring version.
        row_seq: the sequence for this cell's row ("my choice").
        col_seq: the sequence for this cell's column ("opponent choice").
        sequence_order: the full ordered list of 8 sequences, used to
            determine which sequence is "i" vs "j" for the stored key.
 
    Returns:
        (p_row_beats_col, ties)
    """
    row_index = sequence_order.index(row_seq)
    col_index = sequence_order.index(col_seq)
 
    if row_index < col_index:
        # row_seq is "i", col_seq is "j" -- stored directly
        key = f"{row_seq}_vs_{col_seq}"
        entry = pair_dict[key]
        return entry["p_i_beats_j"], entry["ties"]
    else:
        # col_seq is "i", row_seq is "j" -- stored in the other direction
        key = f"{col_seq}_vs_{row_seq}"
        entry = pair_dict[key]
        return entry["p_j_beats_i"], entry["ties"]
 
 
def build_matrix(
    probabilities: dict, version: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Build the 8x8 probability matrix, tie-count matrix, and diagonal
    mask for one scoring version, in sequence_order's row/column order.
 
    Returns:
        - win_probs: (8, 8) float array. win_probs[row][col] =
          P(sequence_order[row] beats sequence_order[col]). Diagonal
          values are meaningless (they're masked).
        - tie_counts: (8, 8) int array, same indexing.
        - mask: (8, 8) boolean array, True on the diagonal.
        - sequence_labels: the row/column tick labels, in order.
    """
    sequence_order = probabilities["sequence_order"]
    n = len(sequence_order)
    pair_dict = probabilities[version]
 
    win_probs = np.zeros((n, n))
    tie_counts = np.zeros((n, n), dtype=int)
    mask = np.eye(n, dtype=bool)  # True on the diagonal, False elsewhere
 
    for row_i, row_seq in enumerate(sequence_order):
        for col_j, col_seq in enumerate(sequence_order):
            if row_i == col_j:
                continue  # diagonal is masked, leave as 0/placeholder
            p, ties = get_pair_probability(pair_dict, row_seq, col_seq, sequence_order)
            win_probs[row_i, col_j] = p
            tie_counts[row_i, col_j] = ties
 
    return win_probs, tie_counts, mask, sequence_order
 
 
def build_annotations(win_probs: np.ndarray, tie_counts: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Build the per-cell text labels in the required "XX (YY)" format --
    win percentage as a plain rounded number, followed by the tie count
    in brackets. Diagonal cells get an empty string (they're masked/gray).
    """
    n = win_probs.shape[0]
    annotations = np.empty((n, n), dtype=object)
    for i in range(n):
        for j in range(n):
            if mask[i, j]:
                annotations[i, j] = ""
            else:
                pct = round(win_probs[i, j] * 100)
                annotations[i, j] = f"{pct} ({tie_counts[i, j]})"
    return annotations
 
 
# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
 
def plot_heatmap(
    win_probs: np.ndarray,
    annotations: np.ndarray,
    mask: np.ndarray,
    sequence_labels: list[str],
    version: str,
    n_decks: int,
) -> plt.Figure:
    """
    Build one heatmap figure for one scoring version, matching the
    professor's sample formatting requirements: gray diagonal, axes
    labeled "My Choice"/"Opponent Choice", custom "XX (YY)" cell text,
    and a title stating the scoring version and sample size.
    """
    fig, ax = plt.subplots(figsize=(9, 7))
 
    sns.heatmap(
        win_probs,
        mask=mask,
        annot=annotations,
        fmt="",  # use our own annotation strings, not seaborn's number formatting
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        square=True,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Win Probability"},
        xticklabels=sequence_labels,
        yticklabels=sequence_labels,
        ax=ax,
    )
 
    # Color the masked (diagonal) cells gray instead of leaving them blank
    ax.set_facecolor("lightgray")
 
    ax.set_xlabel("Opponent Choice")
    ax.set_ylabel("My Choice")
    ax.set_title(
        f"{VERSION_DISPLAY_NAMES[version]}\n"
        f"Cell values: win percentage (tie count) — N = {n_decks:,} decks"
    )
 
    fig.tight_layout()
    return fig
 
 
# ---------------------------------------------------------------------------
# Saving figures (with archiving of old versions, per assignment spec)
# ---------------------------------------------------------------------------
 
def archive_existing_figures() -> None:
    """
    Move any figures currently in the top level of figures/ into
    figures/archive/ before saving new ones, so the top level always
    contains only the most recent heatmaps.
    """
    if not PATH_FIGURES.exists():
        return
 
    PATH_FIGURES_ARCHIVE.mkdir(parents=True, exist_ok=True)
    timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
 
    for item in PATH_FIGURES.iterdir():
        if item.is_file():  # skip the archive/ subfolder itself
            archived_name = f"{timestamp}_{item.name}"
            shutil.move(str(item), str(PATH_FIGURES_ARCHIVE / archived_name))
 
 
def save_figure(fig: plt.Figure, filename: str) -> Path:
    """
    Save a figure to the top level of figures/.
    """
    PATH_FIGURES.mkdir(parents=True, exist_ok=True)
    path = PATH_FIGURES / filename
    fig.savefig(path, dpi=200, bbox_inches="tight")
    return path
 
 
# ---------------------------------------------------------------------------
# Top-level entry point used by main.py
# ---------------------------------------------------------------------------
 
def generate_heatmaps() -> None:
    """
    The main entry point, called from main.py when the user chooses to
    "display the most up-to-date heatmaps." Builds and saves both
    scoring versions' heatmaps from the current processed probabilities.
    """
    probabilities = load_probabilities()
    n_decks = probabilities["n_decks"]
 
    archive_existing_figures()
 
    for version in SCORING_VERSIONS:
        win_probs, tie_counts, mask, sequence_labels = build_matrix(probabilities, version)
        annotations = build_annotations(win_probs, tie_counts, mask)
        fig = plot_heatmap(win_probs, annotations, mask, sequence_labels, version, n_decks)
        saved_path = save_figure(fig, f"heatmap_{version}.png")
        plt.close(fig)
        print(f"Saved {version} heatmap to {saved_path}")
 
    print(f"Done. Heatmaps reflect {n_decks:,} decks.")
 