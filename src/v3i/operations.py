"""Custom operation definitions for neural networks."""

import torch


def create_structure_constants() -> torch.Tensor:
    """Create structure constants.

    Constructs a tensor f of shape (7, 7, 7) that encodes the
    structure constants for the unique cross product in ℝ⁷. We use
    the Fano plane convention with these seven cyclic triples:

        (0, 1, 2)
        (0, 3, 4)
        (0, 6, 5)
        (1, 3, 5)
        (1, 4, 6)
        (2, 3, 6)
        (2, 5, 4)

    For each triple (a, b, c) we set:
      f[a,b,c] = f[b,c,a] = f[c,a,b] = +1,
      f[b,a,c] = f[a,c,b] = f[c,b,a] = -1.
    """
    f = torch.zeros(7, 7, 7)
    triples = [
        (0, 1, 2),
        (0, 3, 4),
        (0, 6, 5),
        (1, 3, 5),
        (1, 4, 6),
        (2, 3, 6),
        (2, 5, 4),
    ]
    for a, b, c in triples:
        # set cyclic permutations to +1
        f[a, b, c] = 1
        f[b, c, a] = 1
        f[c, a, b] = 1
        # set anti-cyclic permutations to -1
        f[b, a, c] = -1
        f[a, c, b] = -1
        f[c, b, a] = -1
    return f


# Precompute the constant tensor
STRUCTURE_CONSTANTS = create_structure_constants()


def cross_7d(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Compute the 7-dimensional cross product of two 7-d vectors.

    Args:
        u: A tensor of shape (7,).
        v: A tensor of shape (7,).

    Returns:
        Cross product of `u` and `v`, a tensor of shape (7,) defined as
        ```
            (u x v)_k = sum_{i,j} f[i,j,k] * u[i] * v[j]
        ```
    """
    return torch.einsum("ijk,i,j->k", STRUCTURE_CONSTANTS, u, v)


def cumulative_cross_product(vectors: torch.Tensor) -> torch.Tensor:
    """Computes the cumulative cross product of a sequence of 7-d vectors.

    For a word with vectors [v1, v2, ..., vn], we compute:

         output = v1 × v2, then output = output × v3, and so on.

    Args:
        vectors: A tensor of shape (L, 7), where L is the number of characters.

    Returns:
        A 7-d cumulative cross product vector representing the word.
    """
    if vectors.shape[0] == 0:
        msg = "Empty sequence of vectors."
        raise ValueError(msg)
    result = vectors[0]
    for i in range(1, vectors.shape[0]):
        result = cross_7d(result, vectors[i])
    return result
