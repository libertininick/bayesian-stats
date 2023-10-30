"""Test for MCMC module."""

import pyro.distributions as dist
import pytest
import torch
from torch import Tensor

from bayesian_stats.mcmc import Bounds, get_proposal_distribution, initialize_samples


# Test constants
NUM_SAMPLES = 1_024
NUM_PARAMETERS = 3
PROPOSAL_WIDTH = 0.1


# Test fixtures
@pytest.fixture(scope="module")
def parameter_bounds() -> dict[str, Bounds]:
    """Get a dictionary of parameter bounds test fixture."""
    return dict(pi=Bounds(0.0, 1.0), mu=Bounds(-5.0, 2.0), sigma=Bounds(1e-6, 3.0))


@pytest.fixture(scope="module")
def samples(parameter_bounds: dict[str, Bounds]) -> Tensor:
    """Get parameter samples test fixture."""
    return initialize_samples(
        parameter_bounds.values(), num_samples=NUM_SAMPLES, seed=456
    )


@pytest.fixture(scope="module")
def proposal_dist(samples: Tensor, parameter_bounds: dict[str, Bounds]) -> dist.Uniform:
    """Get a uniform proposal distribution text fixture."""
    lower_bounds, upper_bounds = torch.tensor(list(parameter_bounds.values())).T
    return get_proposal_distribution(
        samples, lower_bounds, upper_bounds, proposal_width=PROPOSAL_WIDTH
    )


# Tests
@pytest.mark.parametrize("num_samples", [3, 5, 6, 7])
def test_initialize_samples_raise_on_bad_num_samples(num_samples: int) -> None:
    """Test that a `num_samples` that isn't a power of 2 raises an exception."""
    with pytest.raises(ValueError):
        initialize_samples([(0, 1)], num_samples)


def test_initialize_samples_seeded() -> None:
    """Test that seed samples reproduce."""
    expected = torch.tensor(
        [
            [0.9936113953590393, -3.014742136001587],
            [0.26954954862594604, -0.8952667117118835],
            [0.04656250774860382, -3.7364935874938965],
            [0.6903061270713806, 1.5815576314926147],
        ]
    )
    samples = initialize_samples(
        bounds=[(0.0, 1.0), (-5.0, 2.0)], num_samples=4, seed=1234
    )
    assert torch.allclose(expected, samples)


def test_proposal_distribution_shape(proposal_dist: dist.Uniform) -> None:
    """Test that proposal distribution has correct shape."""
    assert proposal_dist.shape() == (NUM_SAMPLES, NUM_PARAMETERS)


def test_proposal_distribution_width(
    proposal_dist: dist.Uniform, parameter_bounds: dict[str, Bounds]
) -> None:
    """Test that each parameter sample's proposal distribution has correct width."""
    for i, bounds in enumerate(parameter_bounds.values()):
        # Samples were randomly initialized so expected sampled dist is ~ U(lb, ub)
        expected_dist = dist.Uniform(*bounds)

        # Look up cumulative probability of lower and upper bounds of proposal dist
        lower_prob = expected_dist.cdf(proposal_dist.low[:, i])
        upper_prob = expected_dist.cdf(proposal_dist.high[:, i])

        # Check width of proposal dist is roughly = PROPOSAL_WIDTH
        diff = torch.abs((upper_prob - lower_prob) - PROPOSAL_WIDTH)
        assert torch.all(diff <= 0.01)


def test_proposal_distribution_sample(
    proposal_dist: dist.Uniform, parameter_bounds: dict[str, Bounds]
) -> None:
    """Test a sample from the proposal distribution is correct shape and all proposed \
        values are within bounds."""
    proposals = proposal_dist.sample()
    assert proposals.shape == (NUM_SAMPLES, NUM_PARAMETERS)

    lower_bounds, upper_bounds = torch.tensor(list(parameter_bounds.values())).T
    assert torch.all(proposals >= lower_bounds[None])
    assert torch.all(proposals <= upper_bounds[None])
