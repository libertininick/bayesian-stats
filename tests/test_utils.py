"""Tests for utility classes and functions."""
import pytest
import torch
from scipy.stats import spearmanr

from bayesian_stats.utils import get_spearman_corrcoef


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
