from __future__ import annotations
import numpy as np
from markov.matrix import Markov_Matrix, State
from markov.tokenizer import Tokenizer

def _require_order_one(matrix: Markov_Matrix, function_name: str) -> np.ndarray:
    if matrix.order != 1 or matrix._matrix is None:
        raise ValueError(f"{function_name} requires an order-1 matrix")
    return matrix._matrix

# left eigenvector with eigenvalue 1
def stationary_distribution(matrix: Markov_Matrix) -> np.ndarray:
    transition = _require_order_one(matrix, "stationary_distribution")
    eigenvalues, eigenvectors = np.linalg.eig(transition.T)
    index = int(np.argmin(np.abs(eigenvalues - 1.0)))
    vector = np.abs(eigenvectors[:, index].real)
    return vector / vector.sum()

# state distribution after each step from a uniform start : (steps, vocab_size)
def convergence_series(matrix: Markov_Matrix, steps: int) -> np.ndarray:
    if steps < 1:
        raise ValueError("steps must be at least 1")
    transition = _require_order_one(matrix, "convergence_series")
    vocab_size = matrix.vocab_size
    distribution = np.full(vocab_size, 1.0 / vocab_size, dtype=np.float64)
    rows: list[np.ndarray] = []
    for _ in range(steps):
        distribution = distribution @ transition
        rows.append(distribution.copy())
    return np.array(rows)

# top-k next tokens and probabilities for a given token
def top_transitions(matrix: Markov_Matrix, tokenizer: Tokenizer, token: str, k: int) -> list[tuple[str, float]]:
    if k < 1:
        raise ValueError("k must be at least 1")
    index = tokenizer.word_to_index[token]
    if matrix.order == 1:
        state: State = (index,)
    else:
        state = (index,) * matrix.order
    row = matrix.get_row(state)
    top_indices = np.argsort(row)[::-1][:k]
    return [(tokenizer.index_to_word[int(i)], float(row[i])) for i in top_indices]

# base 2 entropy of each transition row
def entropy_per_state(matrix: Markov_Matrix) -> np.ndarray:
    transition = _require_order_one(matrix, "entropy_per_state")
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(transition > 0, -transition * np.log2(transition), 0.0)
    return terms.sum(axis=1)
