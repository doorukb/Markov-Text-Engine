from __future__ import annotations

import argparse
import re
import textwrap
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from markov import analysis
from markov.generator import generate
from markov.matrix import Markov_Matrix
from markov.tokenizer import Tokenizer
from source.loader import load_text
from visualization.heatmap import (
    plot_convergence,
    plot_entropy,
    plot_matrix,
    plot_stationary,
)

def _fix_punctuation(text: str) -> str:
    return re.sub(r"\s([.?!])", r"\1", text)

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Markov Engine")
    parser.add_argument("data", help="Path to the data .txt file")
    parser.add_argument(
        "--order",
        type=int,
        default=1,
        help="Order of the Markov chain",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Max tokens for generation",
    )
    parser.add_argument(
        "--save-figures",
        action="store_true",
        help="Save figures to outputs/ instead of displaying",
    )
    parser.add_argument(
        "--outputs-dir",
        type=str,
        default="outputs",
        help="Directory for output_text.txt and figures when --save-figures is set",
    )
    return parser.parse_args()


def _format_output_lines(text: str, *, width: int = 80) -> str:
    """Break generated text into lines (by sentence, then word-wrap long runs)."""
    stripped = text.strip()
    if not stripped:
        return ""

    sentences = re.split(r"(?<=[.?!])\s+", stripped)
    lines: list[str] = []
    for sentence in sentences:
        if len(sentence) <= width:
            lines.append(sentence)
        else:
            lines.extend(textwrap.wrap(sentence, width=width))
    return "\n".join(lines) + "\n"


def _write_output_text(text: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "output_text.txt"
    path.write_text(_format_output_lines(text), encoding="utf-8")
    return path

def _save_figures(figures: list[tuple[str, plt.Figure]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, figure in figures:
        figure.savefig(output_dir / f"{name}.png", dpi=150, bbox_inches="tight")
        plt.close(figure)
    print(f"Figures saved to {output_dir.resolve()}")

def _show_figures(_figures: list[tuple[str, plt.Figure]]) -> None:
    plt.show()

def main() -> None:
    args = _parse_args()

    text = load_text(args.data)
    tokenizer = Tokenizer()
    indices = tokenizer.fit_encode(text)

    matrix = Markov_Matrix(args.order)
    matrix.fit(indices, tokenizer.vocab_size)

    generated = generate(matrix, tokenizer, max_tokens=args.max_tokens)
    output_text = _fix_punctuation(generated)
    print(output_text)

    output_dir = Path(args.outputs_dir)
    text_path = _write_output_text(output_text, output_dir)
    print(f"Generated text saved to {text_path.resolve()}")

    if args.order > 1:
        print(
            "Analysis and visualizations are only available for order-1 models; "
            f"skipping (--order={args.order})."
        )
        return

    stationary = analysis.stationary_distribution(matrix)
    series = analysis.convergence_series(matrix, steps=50)
    entropy = analysis.entropy_per_state(matrix)
    track_indices = np.argsort(stationary)[::-1][:5].tolist()

    figures: list[tuple[str, plt.Figure]] = [
        ("matrix", plot_matrix(matrix, tokenizer)),
        ("stationary", plot_stationary(stationary, tokenizer)),
        ("convergence", plot_convergence(series, tokenizer, track_indices)),
        ("entropy", plot_entropy(entropy, tokenizer)),
    ]
    if args.save_figures:
        _save_figures(figures, output_dir)
    else:
        _show_figures(figures)

if __name__ == "__main__":
    main()