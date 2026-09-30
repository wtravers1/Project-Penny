import json
import shutil
from datetime import datetime as dt
from pathlib import Path
 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
 
PATH_PROBABILITIES = Path("data/processed/probabilities.json") 
PATH_FIGURES = Path("figures")
PATH_FIGURES_ARCHIVE = Path("figures/archive")
 
SCORING_VERSIONS = ("tricks", "cards")
VERSION_DISPLAY_NAMES = {
    "tricks": "Tricks",  # original
    "cards": "Cards",    # Ron's version
}
 
 
# Loading processed data
 
def load_probabilities() -> dict:
    """
    Load the probabilities.json file produced by dataprocessing.py
 
    Raises an error if the file doesn't exist
    """
    if not PATH_PROBABILITIES.exists():
        raise FileNotFoundError(
            f"No processed probabilities found at {PATH_PROBABILITIES}. "
            "Generate and process some decks first (see main.py)."
        )
    with PATH_PROBABILITIES.open("r") as f:
        return json.load(f)
 
 
# Building the 8x8 matrices for one scoring version
 
def get_pair_probability(
    pair_dict: dict,
    row_seq: str,
    col_seq: str,
    sequence_order: list[str],
) -> tuple[float, int, int]:
    """
    Look up one cell of the heatmap: the probability that row_seq beats
    col_seq (used for the cell color), plus the already-rounded win and tie
    percentages stored by dataprocessing.py (used for the cell label)
 
    Pairs are stored unordered in pair_dict, under a key built from
    whichever of the two sequences comes first in sequence_order (that
    one is "i", the other is "j"). This function figures out which
    direction was stored, fetches the right entry, and returns the
    probability from row_seq's perspective regardless of which side of
    the pair row_seq happened to be
 
    Args:
        pair_dict: probabilities[version] -- the dict of all 28 pairs
            for one scoring version
        row_seq: the sequence for this cell's row ("opponent choice")
        col_seq: the sequence for this cell's column ("my choice").
        sequence_order: the full ordered list of 8 sequences, used to
            determine which sequence is "i" vs "j" for the stored key
 
    Returns:
        (p_row_beats_col, win_pct, tie_pct)
    """
    row_index = sequence_order.index(row_seq)
    col_index = sequence_order.index(col_seq)
 
    if row_index < col_index:
        # row_seq is "i", col_seq is "j" -- stored directly
        key = f"{row_seq}_vs_{col_seq}"
        entry = pair_dict[key]
        return entry["p_i_beats_j"], entry["win_pct_i"], entry["tie_pct"]
    else:
        # col_seq is "i", row_seq is "j" -- stored in the other direction
        key = f"{col_seq}_vs_{row_seq}"
        entry = pair_dict[key]
        return entry["p_j_beats_i"], entry["win_pct_j"], entry["tie_pct"]
 
 
def build_matrix(
    probabilities: dict, version: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Build the 8x8 win-probability matrix, the win and tie percentage
    matrices, and a diagonal mask for one scoring version, in
    sequence_order's row/column order
 
    Rows are the opponent's choice and columns are my choice
 
    Returns:
        - win_probs: (8, 8) float array. 
        - win_pcts, tie_pcts: (8, 8) int arrays of whole percentages
        - mask: (8, 8) boolean array, True on the diagonal
        - sequence_labels: the row/column tick labels, in order.
    """
    sequence_order = probabilities["sequence_order"]
    n = len(sequence_order)
    pair_dict = probabilities[version]
 
    win_probs = np.zeros((n, n))
    win_pcts = np.zeros((n, n), dtype=int)
    tie_pcts = np.zeros((n, n), dtype=int)
    mask = np.eye(n, dtype=bool) 
 
    for row_i, opp_seq in enumerate(sequence_order):
        for col_j, my_seq in enumerate(sequence_order):
            if row_i == col_j:
                continue 
            p, win_pct, tie_pct = get_pair_probability(
                pair_dict, my_seq, opp_seq, sequence_order
            )
            win_probs[row_i, col_j] = p
            win_pcts[row_i, col_j] = win_pct
            tie_pcts[row_i, col_j] = tie_pct

    return win_probs, win_pcts, tie_pcts, mask, sequence_order
 
 
def build_annotations(win_pcts: np.ndarray, tie_pcts: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Build the per-cell text labels
    Diagonal cells get an empty string (they're masked/gray)
    """
    n = win_pcts.shape[0]
    annotations = np.empty((n, n), dtype=object)
    for i in range(n):
        for j in range(n):
            if mask[i, j]:
                annotations[i, j] = ""
            else:
                annotations[i, j] = f"{win_pcts[i, j]}({tie_pcts[i, j]})"
    return annotations
 
# Plotting
 
def plot_heatmap(
    win_probs: np.ndarray,
    annotations: np.ndarray,
    mask: np.ndarray,
    sequence_labels: list[str],
    version: str,
    n_decks: int,
) -> plt.Figure:
    """
    Build one heatmap figure for one scoring version: gray diagonal, axes
    labeled, and a title stating the scoring version and sample size
    """
    fig, ax = plt.subplots(figsize=(9, 7))
 
    sns.heatmap(
        win_probs,
        mask=mask,
        annot=annotations,
        fmt="",  
        cmap="Blues",
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
 

    ax.set_facecolor("lightgray")
 
    ax.set_xlabel("My Choice")
    ax.set_ylabel("Opponent Choice")
    ax.set_title(
        f"My Probability of Win(Tie)\n"
        f"Scoring By {VERSION_DISPLAY_NAMES[version]}\n"
        f"N={n_decks:,}"
    )
 
    fig.tight_layout()
    return fig
 
 

# Saving figures and archiving of old versions

 
def archive_existing_figures() -> None:
    """
    Move any figures currently in the top level of figures/ into
    figures/archive/ before saving new ones
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
 
 
#entry point used by main.py
 
def generate_heatmaps() -> None:
    """
    The main entry point, called from main.py. Builds and saves both
    scoring versions heatmaps from the current processed probabilities
    """
    probabilities = load_probabilities()
    n_decks = probabilities["n_decks"]
 
    archive_existing_figures()
 
    for version in SCORING_VERSIONS:
        win_probs, win_pcts, tie_pcts, mask, sequence_labels = build_matrix(
            probabilities, version
        )
        annotations = build_annotations(win_pcts, tie_pcts, mask)
        fig = plot_heatmap(win_probs, annotations, mask, sequence_labels, version, n_decks)
        saved_path = save_figure(fig, f"heatmap_{version}.png")
        plt.close(fig)
        print(f"Saved {version} heatmap to {saved_path}")
 
    print(f"Done. Heatmaps reflect {n_decks:,} decks.")
 