"""Tests for utility classes and functions."""
import pytest
import torch
from scipy.stats import spearmanr

from bayesian_stats.utils import get_spearman_corrcoef, get_wasserstein_distance


@pytest.mark.parametrize("shape", [(100,), (2, 100), (2, 3, 100)])
def test_get_spearman_corrcoef_diff_shapes(shape: tuple[int, ...]) -> None:
    """Test that different shaped inputs are handled correctly."""
    a = torch.randn(*shape)
    b = torch.randn(*shape)
    result = get_spearman_corrcoef(a, b)
    assert result.shape == shape[:-1]


@pytest.mark.parametrize("seed", range(10))
def test_get_spearman_corrcoef_values(seed: int) -> None:
    """Test computed correlation statistic matches `scipy.stats.spearmanr`."""
    torch.manual_seed(seed)
    a = torch.randn(100)
    b = torch.randn(100)
    result = get_spearman_corrcoef(a, b)
    assert torch.allclose(
        result.to(torch.float64), torch.tensor(spearmanr(a, b).statistic)
    )


def test_get_wasserstein_distance_1d() -> None:
    """Test that calculation on 1D tensors works and result shape is correct."""
    a = torch.randn(100)
    b = torch.randn(100)
    result = get_wasserstein_distance(a, b)
    assert result.shape == ()


@pytest.mark.parametrize("batch", [1, 2, 3])
def test_get_wasserstein_distance_2d(batch: int) -> None:
    """Test that calculation on 2D tensors works and result shape is correct."""
    a = torch.randn(batch, 100)
    b = torch.randn(batch, 100)
    result = get_wasserstein_distance(a, b)
    assert result.shape == (batch,)


@pytest.mark.parametrize("scale", [0.1, 0.5, 1.5, 10.0])
def test_get_wasserstein_distance_normalized(scale: float) -> None:
    """Test that distance is the same irrespective of scaling."""
    a = torch.randn(100)
    b = torch.randn(100)
    assert torch.allclose(
        get_wasserstein_distance(a, b), get_wasserstein_distance(a * scale, b * scale)
    )
