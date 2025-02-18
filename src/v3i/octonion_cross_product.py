import torch

def create_structure_constants():
    """
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
        (2, 5, 4)
    ]
    for (a, b, c) in triples:
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

def cross_7d(u, v):
    """
    Compute the 7-dimensional cross product of two 7-d vectors.
    
    Args:
        u, v (Tensor): each is a tensor of shape (7,)
    
    Returns:
        Tensor: cross product, a tensor of shape (7,)
                 defined as (u x v)_k = sum_{i,j} f[i,j,k] * u[i] * v[j]
    """
    return torch.einsum('ijk,i,j->k', STRUCTURE_CONSTANTS, u, v) 