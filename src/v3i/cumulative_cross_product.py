import torch
from octonion_cross_product import cross_7d

def cumulative_cross_product(vectors):
    """
    Computes the cumulative cross product of a sequence of 7-d vectors.
    For a word with vectors [v1, v2, ..., vn], we compute:
    
         output = v1 × v2, then output = output × v3, and so on.
    
    Args:
        vectors (Tensor): shape (L, 7), where L is the number of characters.
    
    Returns:
        Tensor: a 7-d vector representing the word.
    """
    if vectors.shape[0] == 0:
        raise ValueError("Empty sequence of vectors.")
    result = vectors[0]
    for i in range(1, vectors.shape[0]):
        result = cross_7d(result, vectors[i])
    return result 