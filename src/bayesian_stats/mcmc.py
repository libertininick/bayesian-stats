"""Markov chain Monte Carlo implementation.

This is a parallel MCMC sampler that evolves a population of `N` samples over `M`
iterations, so at the end of the iterations you have `N` samples approximating
the posterior distribution.
"""
import math
from collections import Counter
from dataclasses import dataclass
from functools import partial
from typing import Callable, Iterable, NamedTuple, ParamSpec

import pyro.distributions as dist
import torch
from scipy.stats import qmc, wasserstein_distance
from torch import Tensor

from bayesian_stats.utils import get_spearman_corrcoef


P = ParamSpec("P")


class Bounds(NamedTuple):
    """Upper and lower bounds for a model parameter.

    Attributes
    ----------
    lower: float
        Lower bound.
    upper: float
        Upper bound.
    """

    lower: float
    upper: float


@dataclass(frozen=True, kw_only=True)
class MCMCResult:
    """Stores the output of a parallel MCMC run.

    Attributes
    ----------
    parameters: list[str]
        Model parameter names.
    map_values: Tensor, shape=(num_parameters,)
        Maximum a posteriori parameter values observed during sampling.
    init_samples: Tensor, shape=(num_samples, num_parameters)
        Initial state of samples.
    samples: Tensor, shape=(num_samples, num_parameters)
        Samples after running MCMC steps.
    max_wasserstein_trace: Tensor, shape=(num_iter, num_parameters)
        Trace of the Wasserstein distance between consecutive sample distributions.
    sample_counter: dict[str, Counter]
        Counters of iterations spent at a specific value for each parameter.
    seed: int | None
        Seed for MCMC run.
    """

    parameters: list[str]
    map_values: Tensor
    init_samples: Tensor
    samples: Tensor
    wasserstein_distance_trace: Tensor
    sample_counter: dict[str, Counter]
    seed: int | None

    @property
    def num_iter(self) -> int:
        """Number of MCMC iterations completed."""
        return int(self.wasserstein_distance_trace.shape[0])

    @property
    def num_parameters(self) -> int:
        """Number of model parameters."""
        return len(self.parameters)

    @property
    def num_samples(self) -> int:
        """Number of MCMC samples per parameter."""
        return int(self.samples.shape[0])

    def __post_init__(self) -> None:
        """Validate result attributes."""
        if len(self.map_values) != self.num_parameters:
            raise ValueError("Number of MAP values != number of parameters")
        if self.samples.shape[1] != self.num_parameters:
            raise ValueError(
                "Number of parameters in samples != number of parameter names"
            )

    def get_samples(self, parameter: str, init: bool = False) -> Tensor:
        """Get a parameter's samples by name.

        Parameters
        ----------
        parameter: str
            Name of parameter.
        init: bool, optional
            Whether to get final samples or initial samples.
            (default = False)

        Returns
        -------
        samples: Tensor, shape=(num_samples,)
        """
        if init:
            return self.init_samples[:, self.parameters.index(parameter)]
        else:
            return self.samples[:, self.parameters.index(parameter)]

    def get_map_estimate(self, parameter: str) -> float:
        """Get the maximum a posteriori estimate for a parameter.

        Parameters
        ----------
        parameter: str
            Name of parameter.

        Returns
        -------
        map_estimate: float
        """
        return self.map_values[self.parameters.index(parameter)].item()

    def get_correlation_to_init(
        self, num_bootstraps: int | None = 100
    ) -> dict[str, Tensor]:
        """Get the correlation of each parameter's samples with initial distribution.

        This is useful as a convergence diagnostic b/c we want the final distribution
        of samples to have forgotten its initialization. Material positive correlation
        would therefore indicate we haven't reached convergence yet.

        Parameters
        ----------
        num_bootstraps: int | None, optional
            Estimate the distribution of each parameter's correlation coefficient
            with `num_bootstraps` resamples. Hey, we are trying to be Bayesian...
            If `None`, return the point estimate for each parameter.
            (default = 100)

        Returns
        -------
        corr_to_init: dict[str, Tensor],
            Distribution of correlation to initialization for each parameter.
        """
        if num_bootstraps is not None:
            bootstrap_idxs = torch.randint(
                low=0, high=self.num_samples, size=(num_bootstraps, self.num_samples)
            )
            a = self.init_samples[bootstrap_idxs].permute(0, 2, 1)
            b = self.samples[bootstrap_idxs].permute(0, 2, 1)
        else:
            a = self.init_samples.T
            a = self.samples.T

        corr_to_init = get_spearman_corrcoef(a, b)

        if corr_to_init.ndim == 2:
            corr_to_init = corr_to_init.T

        return dict(zip(self.parameters, corr_to_init))


def get_wasserstein_distance(
    samples_a: Tensor,
    samples_b: Tensor,
    bounds: Iterable[Bounds | tuple[float, float]],
) -> Tensor:
    """Get the Wasserstein distance of between two sets of samples for each parameter.

    The Wasserstein distance is the minimum amount of distribution weight that must
    be moved, multiplied by the distance it has to be moved to transform `dist_a`
    into `dist_b`.

    Parameters
    ----------
    samples_a: Tensor, shape(num_samples, num_parameters)
        Sample distributions A.
    samples_b: Tensor, shape(num_samples, num_parameters)
        Sample distributions B.
    bounds: Iterable[Bounds]
        Parameter bounds.

    Returns
    -------
    distances: Tensor, shape=(num_parameters)
        Wasserstein distance between samples A and B for each parameter.
    """
    distances: list[float] = []
    for i, (lb, ub) in enumerate(bounds):
        distances.append(
            wasserstein_distance(
                (samples_a[:, i].cpu().numpy() - lb) / (ub - lb),
                (samples_b[:, i].cpu().numpy() - lb) / (ub - lb),
            )
        )
    return torch.tensor(distances)


def get_proposal_distribution(samples: Tensor) -> dist.Normal:
    """Get a normal proposal distribution centered at each parameter sample with \
        scale equal to the standard deviation of each parameter's samples.

    As the sampling distribution of each parameter evolves, the standard deviation
    may shrink, thus annealing the sample movement from iteration to iteration.

    Parameters
    ----------
    samples: Tensor, shape=(num_samples, num_parameters)

    Returns
    -------
    proposal_dist: pyro.distributions.Normal, shape=(num_samples, num_parameters)
        Normal proposal distribution for each parameter sample.
    """
    # Calculate standard deviation of each parameter distribution
    proposal_scales = samples.std(dim=0, keepdim=True)

    # Build normal proposal distribution centered at each sample
    proposal_dist = dist.Normal(loc=samples, scale=proposal_scales)

    return proposal_dist


def get_sample_log_prob(
    parameters: list[str],
    samples: Tensor,
    prior: Callable[P, Tensor],
    likelihood: Callable[P, Tensor] | None,
) -> Tensor:
    """Get un-normalized log probability of samples from likelihood and prior functions.

    Parameters
    ----------
    parameters: list[str], len=num_parameters
        Model parameter names.
    samples: Tensor, shape=(num_samples, num_parameters)
        Samples for each parameter.
    prior: Callable[..., Tensor]
        Function that takes parameter samples as Tensors and returns a single Tensor \
            for the joint prior log probability.
    likelihood: Callable[..., Tensor] | None
        Function w/ same parameter signature as `prior`, that takes parameter samples \
            as Tensors and returns a single Tensor for the joint log likelihood. \
            If `None`, then only prior joint probability will be returned.

    Returns
    -------
    log_probs: : Tensor, shape=(num_samples,)
        Un-normalized log probability of samples.
    """
    # Convert sample tensor to sample dict
    sample_dict = dict(zip(parameters, samples.T))

    # Get joint prior log probability given parameter samples
    log_probs = prior(**sample_dict)

    if likelihood is not None:
        # Add the joint log likelihood given parameter samples
        log_probs += likelihood(**sample_dict)

    return log_probs


def initialize_samples(
    bounds: Iterable[Bounds | tuple[float, float]],
    num_samples: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = torch.float32,
    seed: int | None = None,
) -> Tensor:
    """Randomly initialize samples for each parameter using scrambled Sobol sequences.

    Parameters
    ----------
    bounds: Iterable[Bounds]
        Parameter bounds.
    num_samples: int
        Number of samples to generate for each parameter.
        NOTE: number of samples must be a power of 2.
    device: torch.device | None, optional
        Compute device for samples.
        (default = None)
    dtype: torch.dtype | None, optional
        Datatype for samples.
        (default = torch.float32)
    seed: int | None, optional
        Random state seed.
        (default = None)

    Returns
    -------
    samples: Tensor, shape=(num_samples, num_parameters)

    Raises
    ------
    ValueError
        If `num_samples` is not a power of 2.
    """
    # Check number of samples is a power of 2 for Sobol sampling
    m = math.log2(num_samples)
    if not m.is_integer():
        raise ValueError(f"{num_samples} is not a power of 2")

    # Unpack lower and upper bounds
    l_bounds, u_bounds = zip(*bounds)

    # Draw scrambled Sobol samples
    sampler = qmc.Sobol(d=len(l_bounds), scramble=True, seed=seed)
    samples = sampler.random_base2(int(m))

    # Scale samples
    samples = qmc.scale(samples, l_bounds, u_bounds)

    return torch.from_numpy(samples).to(device=device, dtype=dtype)


def run_mcmc(
    parameter_bounds: dict[str, Bounds],
    prior: Callable[P, Tensor],
    likelihood: Callable[P, Tensor] | None,
    num_samples: int,
    *,
    max_iter: int = 1_000,
    tol: float | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype | None = torch.float32,
    seed: int | None = None,
) -> MCMCResult:
    """Run parallel MCMC sampling.

    Parameters
    ----------
    parameter_bounds: dict[str, Bounds]
        Bounds for each parameter in `prior` and `likelihood` spec.
    prior: Callable[..., Tensor]
        Function that takes parameter samples as Tensors and returns a single Tensor \
            for the joint prior log probability.
    likelihood: Callable[..., Tensor] | None
        Function w/ same parameter signature as `prior`, that takes parameter samples \
            as Tensors and returns a single Tensor for the joint log likelihood. \
            If `None`, then only prior joint probability will be returned.
    num_samples: int
        Number of MCMC samples.
    max_iter: int, optional
        Maximum number of iterations to evolve samples for.
        (default = 1_000)
    tol: float | None, optional
        Wasserstein distance for early stopping. If the maximum change in sampling \
            distributions from one iteration to the next is less than this value \
            iterations will be stopped and the sampling distribution returned. \
            If `None`, then will run to `max_iter`.
            (default = None)
    device: torch.device | None, optional
        Compute device for samples.
        (default = None)
    dtype: torch.dtype | None, optional
        Datatype for samples.
        (default = torch.float32)
    seed: int | None, optional
        Random state seed.
        (default = None)


    Returns
    -------
    samples: dict[str, Tensor]
        Samples for each parameter.
    """
    if seed is not None:
        torch.manual_seed(seed)

    # Initialize samples
    lower_bounds, upper_bounds = torch.tensor(
        list(parameter_bounds.values()),
        device=device,
        dtype=dtype,
    ).T
    samples = initialize_samples(
        bounds=parameter_bounds.values(),
        num_samples=num_samples,
        device=device,
        dtype=dtype,
        seed=seed,
    )
    init_samples = samples.clone()

    # Evaluate samples
    _eval_fxn = partial(
        get_sample_log_prob,
        parameters=list(parameter_bounds.keys()),
        prior=prior,
        likelihood=likelihood,
    )
    sample_scores: Tensor = _eval_fxn(samples=samples)

    # Set MAP parameter values
    map_idx = sample_scores.argmax()
    map_score = sample_scores[map_idx]
    map_values = samples[map_idx]

    # Update iterations
    wasserstein_distance_trace: list[Tensor] = []
    sample_counter: dict[str, Counter] = {
        param: Counter() for param in parameter_bounds
    }
    for _ in range(max_iter):
        # Get proposed samples
        proposed_samples = (
            get_proposal_distribution(samples)
            .sample()
            .clip(min=lower_bounds, max=upper_bounds)
        )

        # Evaluate proposed samples
        proposed_sample_scores: Tensor = _eval_fxn(samples=proposed_samples)

        # Decide whether to move to proposed for each sample
        move_prob = torch.sigmoid(proposed_sample_scores - sample_scores)
        move_mask = move_prob >= torch.rand_like(move_prob)
        new_samples = torch.where(move_mask[:, None], proposed_samples, samples)

        # Calculate Wasserstein distance between samples and new_sample:
        wasserstein_distance_trace.append(
            get_wasserstein_distance(samples, new_samples, parameter_bounds.values())
        )

        # Check for early stopping based on max Wasserstein distance
        if tol is not None and wasserstein_distance_trace[-1].max() <= tol:
            # Stop iterations
            break

        # Update samples, scores, MAP parameter values, and counts
        samples = new_samples
        sample_scores = torch.where(move_mask, proposed_sample_scores, sample_scores)
        map_idx = sample_scores.argmax()
        if sample_scores[map_idx] > map_score:
            map_score = sample_scores[map_idx]
            map_values = samples[map_idx]

        for i, param in enumerate(sample_counter):
            sample_counter[param].update(samples[:, i].cpu().tolist())

    # Collate results
    result = MCMCResult(
        parameters=list(parameter_bounds.keys()),
        map_values=map_values,
        init_samples=init_samples,
        samples=samples,
        wasserstein_distance_trace=torch.stack(wasserstein_distance_trace, dim=0),
        sample_counter=sample_counter,
        seed=seed,
    )

    return result
