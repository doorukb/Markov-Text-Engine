from __future__ import annotations
import numpy as np
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from markov.matrix import Markov_Matrix
from markov.tokenizer import Tokenizer

def _token_labels(tokenizer: Tokenizer, indices: list[int] | range) -> list[str]:
    return [tokenizer.index_to_word[int(i)] for i in indices]

# transition matrix heatmap for the first vocabulary that hit max tokens
def plot_matrix(matrix: Markov_Matrix, tokenizer: Tokenizer, *, max_tokens: int = 40) -> Figure:
    if matrix.order != 1 or matrix._matrix is None:
        raise ValueError("plot_matrix requires an order-1 matrix")

    n = min(max_tokens, matrix.vocab_size)
    subset = matrix._matrix[:n, :n]
    labels = _token_labels(tokenizer, range(n))

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(subset, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("next token")
    ax.set_ylabel("current token")
    ax.set_title(f"Transition probabilities (first {n} of {matrix.vocab_size} tokens, subset view)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig

# bar chart of stationary probabilities, top tokens by mass
def plot_stationary(distribution: np.ndarray, tokenizer: Tokenizer, *, top_n: int = 25) -> Figure:
    n = min(top_n, len(distribution))
    ranked = np.argsort(distribution)[::-1][:n]
    probs = distribution[ranked]
    labels = _token_labels(tokenizer, ranked.tolist())
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels, probs, color="steelblue")
    ax.set_ylabel("probability")
    ax.set_title(f"Stationary distribution (top {n} tokens)")
    ax.tick_params(axis="x", rotation=90)
    fig.tight_layout()
    return fig

# line plot of selected token probabilities across convergence steps
def plot_convergence(series: np.ndarray, tokenizer: Tokenizer, token_indices: list[int]) -> Figure:
    if series.ndim != 2:
        raise ValueError("series must be a 2D array of shape (steps, vocab_size)")
    steps, vocab_size = series.shape
    x = np.arange(1, steps + 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    for index in token_indices:
        if index < 0 or index >= vocab_size:
            raise ValueError(f"token index {index} out of range [0, {vocab_size})")
        label = tokenizer.index_to_word[index]
        ax.plot(x, series[:, index], marker="o", markersize=3, label=label)

    ax.set_xlabel("step")
    ax.set_ylabel("probability")
    ax.set_title("Convergence toward stationary distribution")
    ax.legend(loc="best", fontsize=9)
    ax.set_ylim(0.0, 1.0)
    fig.tight_layout()
    return fig

# bar charts of highest and lowest entropy states, 
# which are essentially the most uncertain and most decisive states
def plot_entropy(entropy: np.ndarray, tokenizer: Tokenizer, *, top_n: int = 20) -> Figure:
    n = min(top_n, len(entropy))
    high_ranked = np.argsort(entropy)[::-1][:n]
    low_ranked = np.argsort(entropy)[:n]

    high_labels = _token_labels(tokenizer, high_ranked.tolist())
    low_labels = _token_labels(tokenizer, low_ranked.tolist())
    high_values = entropy[high_ranked]
    low_values = entropy[low_ranked]

    fig, (ax_high, ax_low) = plt.subplots(1, 2, figsize=(14, 5))

    ax_high.bar(high_labels, high_values, color="coral")
    ax_high.set_title(f"Highest entropy (top {n}, most uncertain)")
    ax_high.set_ylabel("entropy (bits)")
    ax_high.tick_params(axis="x", rotation=90)

    ax_low.bar(low_labels, low_values, color="seagreen")
    ax_low.set_title(f"Lowest entropy (top {n}, most decisive)")
    ax_low.set_ylabel("entropy (bits)")
    ax_low.tick_params(axis="x", rotation=90)

    fig.suptitle("Per-state transition entropy", y=1.02)
    fig.tight_layout()
    return fig