from __future__ import annotations
import re

_SENTENCE_END_PATTERN = re.compile(r"([.?!])")
_NON_TOKEN_CHAR_PATTERN = re.compile(r"[^a-z\s.?!]")

# our class to clean, tokenize, and index the input text for our model
class Tokenizer:
    def __init__(self) -> None:
        self.word_to_index: dict[str, int] = {}
        self.index_to_word: dict[int, str] = {}

    @property
    def vocab_size(self) -> int:
        return len(self.word_to_index)

    def fit(self, text: str) -> None:
        tokens = self.tokenize(text)
        unique_tokens = list(dict.fromkeys(tokens))
        self.word_to_index = {word: index for index, word in enumerate(unique_tokens)}
        self.index_to_word = {index: word for word, index in self.word_to_index.items()}

    # keep the tokens as lowercase and do not ignore punctuations
    def tokenize(self, text: str) -> list[str]:
        lowered = text.lower()
        padded = _SENTENCE_END_PATTERN.sub(r" \1 ", lowered)
        cleaned = _NON_TOKEN_CHAR_PATTERN.sub("", padded)
        return cleaned.split()

    def fit_encode(self, text: str) -> list[int]:
        self.fit(text)
        return self.encode(text)

    # tokenize the text and map tokens to indices
    def encode(self, text: str) -> list[int]:
        if not self.word_to_index:
            raise ValueError("tokenizer currently has no vocabulary. Try calling fit() first.")
        tokens = self.tokenize(text)
        unknown = [token for token in tokens if token not in self.word_to_index]
        if unknown:
            raise KeyError(f"unknown token : {unknown[:5]}")
        return [self.word_to_index[token] for token in tokens]

    # opposite of encode
    def decode(self, indices: list[int]) -> list[str]:
        if not self.index_to_word:
            raise ValueError("tokenizer currently has no vocabulary. Try calling fit() first.")
        unknown = [index for index in indices if index not in self.index_to_word]
        if unknown:
            raise KeyError(f"unknown index : {unknown[:5]}")
        return [self.index_to_word[index] for index in indices]