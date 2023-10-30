"""Tests for utility classes and functions."""

import pytest
import torch
from torch import Tensor

from bayesian_stats.utils import CumulativeDistribution


@pytest.fixture(scope="module")
def cumulative_distribution() -> CumulativeDistribution:
    """Build and instance of `CumulativeDistribution` for testing."""
    torch.manual_seed(1234)
    samples = torch.randn(10_000, 5) * (torch.arange(5) + 1)
    return CumulativeDistribution(samples)


@pytest.mark.parametrize(
    "values,expected",
    [
        (
            torch.tensor([[-1.0, -1.0, -1.0, -1.0, -1.0]]),
            torch.tensor([[0.1587, 0.3085, 0.3694, 0.4013, 0.4207]]),
        ),
        (
            torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0]]),
            torch.tensor([[0.5000, 0.5000, 0.5000, 0.5000, 0.5000]]),
        ),
        (
            torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0]]),
            torch.tensor([[0.8413, 0.6915, 0.6306, 0.5987, 0.5793]]),
        ),
    ],
)
def test_cumulative_distribution_get_prob(
    cumulative_distribution: CumulativeDistribution,
    values: Tensor,
    expected: Tensor,
) -> None:
    """Test that estimated cumulative probabilities are reasonably close to the \
        expected true values."""
    probs = cumulative_distribution.get_prob(values)
    assert torch.allclose(probs, expected, atol=0.01)


@pytest.mark.parametrize(
    "probs,expected",
    [
        (
            torch.full(size=(1, 5), fill_value=0.33),
            torch.tensor([[-0.4399, -0.8798, -1.3197, -1.7597, -2.1996]]),
        ),
        (
            torch.full(size=(1, 5), fill_value=0.5),
            torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0]]),
        ),
        (
            torch.full(size=(1, 5), fill_value=0.67),
            torch.tensor([[0.4399, 0.8798, 1.3197, 1.7597, 2.1996]]),
        ),
    ],
)
def test_cumulative_distribution_get_value(
    cumulative_distribution: CumulativeDistribution,
    probs: Tensor,
    expected: Tensor,
) -> None:
    """Test that estimated cumulative probabilities are reasonably close to the \
        expected true values."""
    values = cumulative_distribution.get_value(probs)
    std_norm = torch.arange(5) + 1
    diff = torch.abs((values - expected) / std_norm)
    assert torch.all(diff <= 0.05)


@pytest.mark.parametrize("shape", [(10,), (10, 2, 3)])
def test_cumulative_distribution_init_raises(shape: tuple[int, int]) -> None:
    """Test that samples of the incorrect shape raise ValueError."""
    with pytest.raises(ValueError):
        CumulativeDistribution(torch.rand(*shape))


@pytest.mark.parametrize("shape", [(5,), (10,), (10, 3), (10, 6), (10, 2, 5)])
def test_cumulative_distribution_get_raises(
    cumulative_distribution: CumulativeDistribution, shape: tuple[int, int]
) -> None:
    """Test that probs | values of the incorrect shape raise ValueError for gets."""
    x = torch.rand(*shape)

    with pytest.raises(ValueError):
        cumulative_distribution.get_prob(x)

    with pytest.raises(ValueError):
        cumulative_distribution.get_value(x)
