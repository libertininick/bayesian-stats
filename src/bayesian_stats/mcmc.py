"""Markov chain Monte Carlo implementation.

This is a parallel MCMC sampler that evolves a population of `N` samples over `M`
iterations, so at the end of the iterations you have `N` samples approximating
the posterior distribution.
"""
import math
from collections import Counter
from dataclasses import dataclass
from functools import partial
from itertools import accumulate, chain
from typing import Callable, Iterable, NamedTuple, ParamSpec

import pandas as pd
import pyro.distributions as dist
import torch
from scipy.stats import qmc
from torch import Tensor

from bayesian_stats.utils import (
    get_spearman_corrcoef,
    get_wasserstein_distance,
)


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
    parameters: list[str], len=num_parameters
        Model parameter names.
    map_values: Tensor, shape=(num_parameters,)
        Maximum a posteriori parameter values observed during sampling.
    samples: Tensor, shape=(num_samples, num_parameters)
        Samples for each parameter after running MCMC iterations.
    split_a_idxs: Tensor, shape=(num_samples / 2, num_parameters), dtype=int64
        Split A indexes.
    split_b_idxs: Tensor, shape=(num_samples / 2, num_parameters), dtype=int64
        Split B indexes.
    correlation_traces: Tensor, shape=(num_iter, num_parameters)
        Trace of the Spearman Rank Correlation between current sample distribution
        and the initial sample distribution for each parameter.
    wasserstein_distance_traces: Tensor, shape=(num_iter, num_parameters)
        Trace of the Wasserstein distance between split A and B of the sample
        distributions for each parameter.
    sample_counter: dict[str, Counter]
        Counters of iterations spent at a specific value for each parameter.
    seed: int | None
        Seed for MCMC run.
    """

    parameters: list[str]
    map_values: Tensor
    samples: Tensor
    split_a_idxs: Tensor
    split_b_idxs: Tensor
    correlation_traces: Tensor
    wasserstein_distance_traces: Tensor
    sample_counter: dict[str, Counter]
    seed: int | None

    @property
    def num_iter(self) -> int:
        """Number of MCMC iterations completed."""
        return int(self.correlation_traces.shape[0])

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

    def get_posterior_summary(self) -> pd.DataFrame:
        """Get a summary of posterior parameter samples."""
        return (
            pd.DataFrame(data=self.samples.cpu().numpy(), columns=self.parameters)
            .describe()
            .drop("count")
            .T
        )

    def get_samples(self, parameter: str, split: str | None = None) -> Tensor:
        """Get a parameter's samples by name.

        Parameters
        ----------
        parameter: str
            Name of parameter.
        split: {'a', 'b'} | None, optional
            Get samples for split A or B.
            If `None`, return all samples.
            (default = None)

        Returns
        -------
        samples: Tensor, shape=(num_samples,)
        """
        samples = self.samples
        if split is not None:
            if split == "a":
                index = self.split_a_idxs
            elif split == "b":
                index = self.split_b_idxs
            else:
                raise ValueError(f"Expecting 'a' or 'b'; got {split}")
            samples = torch.gather(samples, dim=0, index=index)
        return samples[:, self.parameters.index(parameter)]

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

    def get_correlation_trace(self, parameter: str) -> Tensor:
        """Get trace of a parameter's samples correlation with initial its distribution.

        This is useful as a convergence diagnostic b/c we want the final distribution
        of samples to have forgotten its initialization. Material positive correlation
        would therefore indicate we haven't reached convergence yet.

        Parameters
        ----------
        parameter: str
            Name of parameter.

        Returns
        -------
        correlation: Tensor, shape=(num_iter,)
        """
        return self.correlation_traces[:, self.parameters.index(parameter)]

    def get_wasserstein_distance_trace(self, parameter: str) -> Tensor:
        """Get trace of a parameter's Wasserstein distance between split A and B.

        This is useful as a convergence diagnostic because at initialization,
        split A contains rank 1 to N/2 values (smallest half) while split B
        contains rank N/2 + 1 to N values (largest half). By definition these splits
        are completely unmixed and have maximal Wasserstein. As sampling progresses
        we want to see A and B mix well so their Wasserstein distance approaches 0,
        indicating the parameter's sample distribution has reach convergence.

        Parameters
        ----------
        parameter: str
            Name of parameter.

        Returns
        -------
        avg_quantile_diff: Tensor, shape=(num_iter,)
        """
        return self.wasserstein_distance_traces[:, self.parameters.index(parameter)]


class ParameterSamples:
    """..."""

    def __init__(
        self,
        parameter_bounds: dict[str, Bounds],
        parameter_shapes: dict[str, torch.Size | tuple[int, ...]],
        num_samples: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = torch.float32,
        seed: int | None = None,
    ) -> None:
        if param_diff := parameter_bounds.keys() ^ parameter_shapes.keys():
            raise ValueError(
                f"Parameter names in bounds and shapes must match; {param_diff}"
            )

        self.shapes = {
            param: torch.Size(shape) for param, shape in parameter_shapes.items()
        }

        num_cumulative_params = list(
            accumulate(v.numel() for v in self.shapes.values())
        )
        self.num_parameters = num_cumulative_params[-1]

        self.slices = {
            param: slice(st, stop)
            for param, st, stop in zip(
                self.shapes, [0] + num_cumulative_params[:-1], num_cumulative_params
            )
        }

        self.num_samples = num_samples
        self.samples = initialize_samples(
            bounds=chain.from_iterable(
                [parameter_bounds[param]] * shape.numel()
                for param, shape in self.shapes.items()
            ),
            num_samples=num_samples,
            device=device,
            dtype=dtype,
            seed=seed,
        )

    @property
    def sample_views(self) -> dict[str, torch.Tensor]:
        return {
            param: self.samples[:, self.slices[param]].view(self.num_samples, *shape)
            for param, shape in self.shapes.items()
        }


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
    prior_fun: Callable[P, Tensor] | None,
    likelihood_fun: Callable[P, Tensor] | None,
) -> Tensor:
    """Get log probability of samples.

    Parameters
    ----------
    parameters: list[str], len=num_parameters
        Model parameter names.
    samples: Tensor, shape=(num_samples, num_parameters)
        Samples for each parameter.
    prior_fun: Callable[..., Tensor] | None
        Function that takes parameter samples as Tensors and returns a single Tensor \
            for the joint prior log probability.
    likelihood_fun: Callable[..., Tensor] | None
        Function that takes parameter samples as Tensors and returns a single Tensor \
            for the joint log likelihood.

    Returns
    -------
    log_prob: : Tensor, shape=(num_samples,)
        Log probability of samples.

    Raises
    ------
    ValueError
        If both `prior_fun` and `likelihood` are None.
    """
    # Convert sample tensor to sample dict
    sample_dict = dict(zip(parameters, samples.T))

    # Compute log probabilities
    if prior_fun is None and likelihood_fun is None:
        raise ValueError("Both `prior_fun` and `likelihood_fun` are None")
    elif prior_fun is None:
        log_prob = likelihood_fun(**sample_dict)
    elif likelihood_fun is None:
        log_prob = prior_fun(**sample_dict)
    else:
        log_prob = prior_fun(**sample_dict) + likelihood_fun(**sample_dict)

    return log_prob


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
    prior_fun: Callable[P, Tensor] | None,
    likelihood_fun: Callable[P, Tensor] | None,
    num_samples: int,
    *,
    max_iter: int = 1_000,
    max_corr: float = 0.025,
    max_split_distance: float = 0.005,
    device: torch.device | None = None,
    dtype: torch.dtype | None = torch.float32,
    seed: int | None = None,
    verbose: bool = True,
) -> MCMCResult:
    """Run parallel MCMC sampling.

    Parameters
    ----------
    parameter_bounds: dict[str, Bounds]
        Bounds for each parameter in `prior` and `likelihood` spec.
    prior_fun: Callable[..., Tensor] | None
        Function that takes parameter samples as Tensors and returns a single Tensor \
            for the joint prior log probability.
    likelihood_fun: Callable[..., Tensor] | None
        Function w/ same parameter signature as `prior`, that takes parameter samples \
            as Tensors and returns a single Tensor for the joint log likelihood.
    num_samples: int
        Number of MCMC samples.
    max_iter: int, optional
        Maximum number of iterations to evolve samples for.
        (default = 1_000)
    max_corr: float, optional
        Maximum correlation between current sample distribution and initial sample \
        distribution for any parameter to allow early stopping of MCMC iterations.
        (default = 5%)
    max_split_distance: float, optional
        If the max Wasser the sampling distribution from \
        one iteration to the next is less than this value for all parameters, and
        the `max_corr` constraint has been satisfied, MCMC iterations will be stopped.
        (default = 0.005)
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
    init_sample_isort = init_samples.argsort(0)
    split_a_idxs = init_sample_isort[: num_samples // 2]
    split_b_idxs = init_sample_isort[num_samples // 2 :]

    # Evaluate samples
    _eval_fxn = partial(
        get_sample_log_prob,
        parameters=list(parameter_bounds.keys()),
        prior_fun=prior_fun,
        likelihood_fun=likelihood_fun,
    )
    sample_scores = _eval_fxn(samples=samples)

    # Set MAP parameter values
    map_idx = sample_scores.argmax()
    map_score = sample_scores[map_idx]
    map_values = samples[map_idx]

    # Update iterations
    correlation_traces: list[Tensor] = []
    wasserstein_distance_traces: list[Tensor] = []
    sample_counter: dict[str, Counter] = {
        param: Counter() for param in parameter_bounds
    }

    for i in range(max_iter):
        # Get proposed samples
        proposed_samples = (
            get_proposal_distribution(samples)
            .sample()
            .clip(min=lower_bounds, max=upper_bounds)
        )

        # Evaluate proposed samples
        proposed_sample_scores = _eval_fxn(samples=proposed_samples)

        # Decide whether to move to proposed for each sample
        move_prob = torch.sigmoid(proposed_sample_scores - sample_scores)
        move_mask = move_prob >= torch.rand_like(move_prob)
        samples = torch.where(move_mask[:, None], proposed_samples, samples)

        # Update samples scores, MAP parameter values, and counts
        sample_scores = torch.where(move_mask, proposed_sample_scores, sample_scores)
        map_idx = sample_scores.argmax()
        if sample_scores[map_idx] > map_score:
            map_score = sample_scores[map_idx]
            map_values = samples[map_idx]

        for j, param in enumerate(sample_counter):
            sample_counter[param].update(samples[:, j].cpu().tolist())

        # Calculation correlation between initial samples and current samples
        correlation_traces.append(get_spearman_corrcoef(init_samples.T, samples.T))

        # Calculate Wasserstein distance between split A and B for each parameter
        wasserstein_distance_traces.append(
            get_wasserstein_distance(
                a=torch.gather(samples, 0, split_a_idxs).T,
                b=torch.gather(samples, 0, split_b_idxs).T,
            )
        )

        # Check for early stopping
        max_corr_i = correlation_traces[-1].max()
        max_splt_dist_i = wasserstein_distance_traces[-1].max()
        if max_corr_i <= max_corr and max_splt_dist_i <= max_split_distance:
            # Stop iterations
            break

        if verbose:
            # Print progress bar
            n_done = int((i + 1) / max_iter * 100)
            print(
                f"""{"█" * n_done}{" " * (100 - n_done)} | {(i+1):>6,}/{max_iter:,}"""
                f" | {max_corr_i:>5.1%} | {max_splt_dist_i:>7.5f}",
                flush=True,
                end="\r",
            )

    # Collate results
    result = MCMCResult(
        parameters=list(parameter_bounds.keys()),
        map_values=map_values,
        samples=samples,
        split_a_idxs=split_a_idxs,
        split_b_idxs=split_b_idxs,
        correlation_traces=torch.stack(correlation_traces, dim=0),
        wasserstein_distance_traces=torch.stack(wasserstein_distance_traces, dim=0),
        sample_counter=sample_counter,
        seed=seed,
    )

    return result
