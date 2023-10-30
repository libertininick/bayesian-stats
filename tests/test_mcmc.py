"""Test for MCMC module."""

import pytest
import torch

from bayesian_stats.mcmc import initialize_samples


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
