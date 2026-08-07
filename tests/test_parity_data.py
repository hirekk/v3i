"""The parity-ladder benchmark: k-bit parity on the centered n-cube.

Embedded on the sphere so that a degree-<k readout provably cannot solve it.
"""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures

from v3i.make_data import generate_parity


def test_shapes_norm_and_labels() -> None:
    """Rows are unit vectors of the right dimension with ±1 labels."""
    rng = np.random.default_rng(0)
    x_tr, y_tr, x_te, _ = generate_parity(400, 100, 0.1, rng, bits=3, dim=8)
    assert x_tr.shape == (400, 8)
    assert x_te.shape == (100, 8)
    assert np.allclose(np.linalg.norm(x_tr, axis=1), 1.0)
    assert set(np.unique(y_tr).tolist()) <= {-1, 1}


def test_labels_are_parity_of_the_bits() -> None:
    """With no noise, the label is the parity of the sign pattern of the bits."""
    rng = np.random.default_rng(1)
    x, y, _, _ = generate_parity(512, 0, 0.0, rng, bits=4, dim=8)
    bits = (x[:, 1:5] > 0).astype(int)
    parity = np.where(bits.sum(1) % 2 == 1, 1, -1)
    assert np.array_equal(parity, y)


def test_embedding_confined_to_first_k_imaginary_coords() -> None:
    """Real part and unused imaginary coords are zero (degree-preserving)."""
    rng = np.random.default_rng(2)
    x, _, _, _ = generate_parity(256, 0, 0.0, rng, bits=3, dim=8)
    assert np.allclose(x[:, 0], 0.0)  # real part
    assert np.allclose(x[:, 4:], 0.0)  # imaginary coords beyond bit 3


def test_classes_are_balanced() -> None:
    """Parity yields a ~balanced label distribution."""
    rng = np.random.default_rng(3)
    _, y, _, _ = generate_parity(2000, 0, 0.1, rng, bits=5, dim=8)
    assert abs(y.mean()) < 0.1


def test_bits_out_of_range_is_rejected() -> None:
    """Reject bits outside 1..dim-1.

    Too many bits (no imaginary room) or fewer than one (degenerate — a
    zero-width cube would normalize 0/0) both raise.
    """
    rng = np.random.default_rng(4)
    with pytest.raises(ValueError, match="bits"):
        generate_parity(100, 10, 0.1, rng, bits=8, dim=8)  # only 7 imaginary dims
    with pytest.raises(ValueError, match="bits"):
        generate_parity(100, 10, 0.1, rng, bits=4, dim=4)  # quaternion: only 3
    with pytest.raises(ValueError, match="bits"):
        generate_parity(100, 10, 0.1, rng, bits=0, dim=8)  # degenerate


def test_degree_staircase_survives_the_embedding() -> None:
    """Quadratic readout stays near chance on 3-bit parity; cubic solves it.

    This is the benchmark's whole point, and it depends on the embedding
    preserving degree — a rational embedding (inverse stereographic) would
    smear it and break this test.
    """
    rng = np.random.default_rng(5)
    x_tr, y_tr, x_te, y_te = generate_parity(2000, 500, 0.02, rng, bits=3, dim=8)

    def acc(degree: int) -> float:
        feats = PolynomialFeatures(degree, include_bias=True)
        f_tr = feats.fit_transform(x_tr[:, 1:4])
        f_te = feats.transform(x_te[:, 1:4])
        clf = LogisticRegression(max_iter=5000, C=1e4).fit(f_tr, y_tr)
        return clf.score(f_te, y_te)

    assert acc(2) < 0.70  # quadratic cannot separate 3-bit parity
    assert acc(3) > 0.95  # cubic can
