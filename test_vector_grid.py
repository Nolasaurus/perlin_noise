import numpy as np
import pytest

from vector_grid import create_grid


def test_create_grid():
    grid = create_grid(3, 3, 1)
    assert grid.shape == (3, 3, 1)
    grid = create_grid(3, 3, 2)
    assert grid.shape == (3, 3, 2)

def test_random_values():
    grid = create_grid(1000, 1000, 2)
    assert grid.min() >= -1
    assert grid.max() <= 1
    assert np.allclose(grid.mean(), 0, atol = 0.05), "Mean is not approximately zero"
    # sigma approx 0.577 for uniform distr between -1 and 1
    assert np.allclose(grid.std(), 0.577, atol=0.05), "Standard deviation is not approximately uniform"