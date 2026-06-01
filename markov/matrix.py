from __future__ import annotations
from typing import TypeAlias
import numpy as np
State: TypeAlias = tuple[int, ...]

# our transition probability model will start as order 1 2D numpy array
class Markov_Matrix:
    def __init__(self, order: int) -> None:
        if order < 1:
            raise ValueError("order must be at least 1")
        self.order = order
        self.vocab_size = 0
        self._matrix: np.ndarray | None = None
        self._counts: dict[State, dict[int, int]] = {}
        self._rows: dict[State, np.ndarray] = {}
    
    # build transition counts from encoded indices
    def fit(self, indices: list[int], vocab_size: int) -> None:
        if vocab_size < 1:
            raise ValueError("vocab_size must be at least 1")
        if len(indices) < self.order + 1:
            raise ValueError(
                f"need at least {self.order + 1} indices for order-{self.order}, got {len(indices)}"
            )
        self.vocab_size = vocab_size
        self._rows.clear()
        if self.order == 1:
            self._counts = {}
            self._matrix = self._fit_order_one(indices, vocab_size)
        else:
            self._matrix = None
            self._counts = self._fit_order_n(indices)

    # return a probability row for the given state
    def get_row(self, state: State) -> np.ndarray:
        if self.vocab_size < 1:
            raise ValueError("matrix is not fitted. Call fit() first.")
        if len(state) != self.order:
            raise ValueError(f"state length must be {self.order}, got {len(state)}")

        if self.order == 1:
            return self._matrix[state[0]].copy()

        if state in self._rows:
            return self._rows[state].copy()

        inner = self._counts.get(state)
        if inner is None:
            return self._uniform_row()

        row = self._normalize_counts(inner)
        self._rows[state] = row
        return row.copy()

    def fit_order_one(self, indices: list[int], vocab_size: int) -> np.ndarray:
        counts = np.zeros((vocab_size, vocab_size), dtype=np.float64)
        for i in range(len(indices) - 1):
            current, nxt = indices[i], indices[i + 1]
            counts[current, nxt] += 1.0

        row_sums = counts.sum(axis=1, keepdims=True)
        matrix = np.divide(
            counts,
            row_sums,
            out=np.zeros_like(counts),
            where=row_sums != 0,
        )
        uniform = self._uniform_row()
        zero_rows = row_sums.flatten() == 0
        matrix[zero_rows] = uniform
        return matrix

    def fit_order_n(self, indices: list[int]) -> dict[State, dict[int, int]]:
        counts: dict[State, dict[int, int]] = {}
        for i in range(len(indices) - self.order):
            state = tuple(indices[i : i + self.order])
            nxt = indices[i + self.order]
            if state not in counts:
                counts[state] = {}
            bucket = counts[state]
            bucket[nxt] = bucket.get(nxt, 0) + 1
        return counts

    def normalize_counts(self, inner: dict[int, int]) -> np.ndarray:
        row = np.zeros(self.vocab_size, dtype=np.float64)
        for index, count in inner.items():
            row[index] = float(count)
        total = row.sum()
        if total == 0:
            return self._uniform_row()
        return row / total

    def uniform_row(self) -> np.ndarray:
        return np.full(self.vocab_size, 1.0 / self.vocab_size, dtype=np.float64)