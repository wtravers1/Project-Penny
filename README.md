# Project Penny

Simulating the Humble-Nishiyama Randomness Game, a two-player card game based on
Penney's Game, to find which 3-card sequences give a player the best odds.

## The game

**Penney's Game** is played with coin flips. Player 1 picks a sequence of three
outcomes (for example Heads-Heads-Tails), then Player 2 picks a different one.
The coin is flipped repeatedly and whoever's sequence appears first wins. The
surprise is that the game is *non-transitive*: whatever Player 1 picks, Player 2
can always choose a sequence that beats it, often by a wide margin.

**The Humble-Nishiyama (H-N) game** plays the same idea with a standard 52-card
deck, using only card color (26 red, 26 black). Each player picks a sequence of
three colors, such as Black-Black-Red. Cards are dealt one at a time, and as soon
as the last three cards match a player's sequence, that player wins the pile:
every card dealt since the last pile was won. Dealing continues with a fresh
window until the deck runs out, and leftover cards go to nobody.

Because cards are dealt without replacement, the H-N game is not the same as
flipping coins: once many red cards are gone, black becomes more likely.

Two ways of scoring are simulated:

1. **Tricks** (the original H-N game): a player scores +1 for each pile won.
2. **Cards** (Ron's variation): a player scores the number of cards in the
   piles they won.

## Purpose

For every one of the 56 possible matchups, we estimate how often each sequence
beats each other sequence, and how often they tie, by simulating millions of
shuffled decks. The results are shown as two heatmaps, one per scoring version,
which reveal the best choice for each player and whether the two versions agree.

## How to run

The project uses [uv](https://docs.astral.sh/uv/) to manage dependencies.

```bash
uv sync                      # install dependencies (first time only)
uv run main.py --add         # add 1,000,000 decks, score them, update heatmaps
uv run main.py --add 500000  # add a specific number of decks instead
uv run main.py --show        # redraw the heatmaps from results already processed
uv run main.py --help        # show usage
```

Run these from the project root. Adding decks takes roughly 90 seconds per
million (about 40s to generate, 40s to score).

Adding decks never regenerates or re-scores decks from earlier runs: only the
new decks are simulated, and their results are added to the running totals.

## Project structure

```
main.py                       entry point; the only file meant to be run
src/
  datagen.py                  create and save shuffled decks (raw data)
  dataprocessing.py           score decks into win/tie counts and probabilities
  datavis.py                  draw the heatmaps from the processed data
data/
  raw/
    deck_log.json             every batch of decks: file name, seed, size, time
    decks_<n>x52_seed_<s>.npy one batch of decks, bit-packed (1 bit per card)
  processed/
    counts.json               running win/loss/tie totals + files already scored
    probabilities.json        presentation-ready percentages for the heatmaps
figures/
  heatmap_tricks.png          most recent heatmaps
  heatmap_cards.png
  archive/                    previous versions of the figures
```

Decks are stored as 0 for black and 1 for red, with each batch generated from a
recorded random seed, so any batch can be reproduced exactly.

## Findings

*(To be written once the results have stabilized: the best sequence for each
player, and whether the tricks and cards versions agree.)*
