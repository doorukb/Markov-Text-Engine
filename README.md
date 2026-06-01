# Markov Engine

- Python 3.10+
- NumPy
- Matplotlib

<br>

A text generation engine built on stochastic matrix mechanics, implemented from scratch using NumPy methods and functions. The project models token transitions as a proper probability matrix rather than a lookup table, which makes the underlying Markov structure visible and mathematically inspectable rather than hidden inside a dict.

The motivating question is not "can this generate useful text" but "what does statistical sequence modeling actually capture, and where does it structurally fail." The engine is built to answer that question concretely. Particularly for outputs of order 1, we can make an analysis of relationship between each word.

<br>

## How it works

The input text is tokenized into a flat sequence of indices. For order-1 models, a square stochastic transition matrix is built directly as a NumPy array where entry `T[i, j]` is the probability of token `j` following token `i`. For order-n models, the state space becomes n-gram tuples and the transition structure is stored as a sparse dict of normalized probability vectors. Generation works by sampling from the appropriate row at each step using `numpy.random.choice` with probability weights.

The matrix representation is not the most memory-efficient approach for large vocabularies, and it is not meant to be. The point is that the matrix is a real mathematical object you can operate on directly, unlike a Python dictionary.

<br>

## Order and input text size

Higher order means more context and more coherent output, but it also means the chain is more likely to reproduce the source text verbatim. For a small input, the crossover happens quickly. 
For example, order 1 will be more generative and unique compared to the original output, yet it can be gibberish. For order 10, it will be less unique compared to the input, yet more coherent and meaningful.

The vocabulary size directly determines matrix dimensions. For order-1, the matrix is `vocab_size x vocab_size`. 

<br>

## Usage

You can either drag your TXT file into the root directory or copy the path of your TXT file, both works- just make sure to insert the right address. Replace "address_of_input_TXT"

```bash
# You want an output of order 1 with maximum size of 150 tokens, and save figures to outputs/ instead of displaying
python main.py address_of_input_TXT --order 1 --max-tokens 150 --save-figures
```

```bash
# You want an output of order 2 with maximum size of 150 tokens, and display the result in terminal & save as TXT
python main.py address_of_input_TXT --order 2 --max-tokens 150
```

<br>

## Analysis layer (Only for outputs of order 1)

Available for order-1 models only, where the full stochastic matrix exists as a NumPy array.

**Stationary distribution** finds the left eigenvector of the transition matrix corresponding to eigenvalue 1. This is the long-run probability of being at each token if the chain runs indefinitely.

**Convergence series** computes the state distribution after each of n multiplication steps starting from a uniform prior, showing the chain losing memory of its starting point.

**Entropy per state** computes the Shannon entropy of each row in the transition matrix. High-entropy rows correspond to tokens with uncertain futures. Low-entropy rows are tokens that almost always lead somewhere specific.

**Top transitions** returns the k most probable next tokens for any given input token, with their probabilities.
