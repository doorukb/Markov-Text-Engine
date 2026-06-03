from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from markov.matrix import Markov_Matrix
from markov.tokenizer import Tokenizer


def _token_labels(tokenizer: Tokenizer, indices: list[int] | range) -> list[str]:
    return [tokenizer.index_to_word[int(i)] for i in indices]


def _row_entropy(transition: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(transition > 0, -transition * np.log2(transition), 0.0)
    return terms.sum(axis=1)


def plot_matrix(matrix: Markov_Matrix, tokenizer: Tokenizer, *, max_tokens: int = 40) -> Figure:
    """Transition matrix heatmap for a subset of tokens (order-1 only).

    Vocabulary indices follow tokenizer insertion order (first-seen in corpus),
    not frequency. The displayed subset is the top *max_tokens* by row entropy so
    the most structurally interesting (uncertain) contexts are shown.
    """
    if matrix.order != 1 or matrix._matrix is None:
        raise ValueError("plot_matrix requires an order-1 matrix")

    transition = matrix._matrix
    vocab_size = matrix.vocab_size
    n = min(max_tokens, vocab_size)
    ranked = np.argsort(_row_entropy(transition))[::-1][:n]
    subset = transition[np.ix_(ranked, ranked)]
    labels = _token_labels(tokenizer, ranked.tolist())

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(subset, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("next token")
    ax.set_ylabel("current token")
    ax.set_title(
        f"Transition probabilities (top {n} by row entropy, of {vocab_size} tokens)"
    )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


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
    fig.tight_layout()
    return fig


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
