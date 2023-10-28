"""Test for MCMC module."""

import pytest

from bayesian_stats.mcmc import initialize_samples


@pytest.mark.parametrize("num_samples", [3, 5, 6, 7])
def test_initialize_samples_raise_on_bad_num_samples(num_samples: int) -> None:
    """Test that a `num_samples` that isn't a power of 2 raises an exception."""
    with pytest.raises(ValueError):
        initialize_samples([(0, 1)], num_samples)
