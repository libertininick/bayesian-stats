"""Markov chain Monte Carlo implementation.

This is a parallel MCMC sampler that evolves a population of `N` samples over `M`
iterations, so at the end of the iterations you have `N` samples approximating
the posterior distribution.
"""
import math
from collections.abc import KeysView, Mapping
from dataclasses import dataclass
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


class ParameterSamples(Mapping):
    """Holds the samples of latent model parameters for parallel MCMC sampling.

    Attributes
    ----------
    bounds: dict[str, Bounds]
        (lower, upper) bounds for parameters.
    lower_bounds: Tensor, shape=(P,)
        Parameter lower bounds.
    num_parameter_elements: dict[str, int]
        Number of elements per parameter.
    num_samples: int
        Number of samples for each parameter.
    sample_matrix: Tensor, shape=(S, P)
        2D matrix of samples from each parameter concatenated together.
    slices: dict[str, slice]
         Each parameter's slice into columns of `sample_matrix`.
    shapes: dict[str, torch.Size]
        Shape of each parameter.
    upper_bounds: Tensor, shape=(P,)
        Parameter upper bounds.
    """

    def __init__(
        self,
        bounds: dict[str, Bounds | tuple[int, int]],
        shapes: dict[str, torch.Size | tuple[int, ...]],
        sample_matrix: Tensor,
    ) -> None:
        """Initialize a `ParameterSamples` instance.

        Parameters
        ----------
        bounds: dict[str, Bounds | tuple[int, int]]
            (lower, upper) bounds for parameters: {param_name: (lb, ub), ...}
        shapes: dict[str, torch.Size | tuple[int, ...]]
            Shape of each parameter: {param_name: shape, ...}
        sample_matrix: Tensor, shape=(S, P)
            2D matrix of samples from each parameter concatenated together.
        """
        if diff := bounds.keys() ^ shapes.keys():
            raise ValueError(f"Parameter names in bounds and shapes must match; {diff}")

        self.bounds = {param: Bounds(*bounds) for param, bounds in bounds.items()}
        self.shapes = {param: torch.Size(shape) for param, shape in shapes.items()}
        self.sample_matrix = sample_matrix
        self.num_samples: int = sample_matrix.shape[0]

        # Number of sample elements per parameter
        self.num_parameter_elements: dict[str, int] = {
            param: shape.numel() for param, shape in self.shapes.items()
        }
        total_numel = sum(self.num_parameter_elements.values())
        if sample_matrix.shape[-1] != total_numel:
            raise ValueError(
                f"# of columns in `sample_matrix` ({sample_matrix.shape[-1]})"
                f"doesn't match # of parameter elements {total_numel}"
            )

        # Each parameter's slice into columns of `sample_matrix`
        self.slices = {
            param: slice(stop - numel, stop)
            for (param, numel), stop in zip(
                self.num_parameter_elements.items(),
                accumulate(self.num_parameter_elements.values()),
            )
        }

        self.lower_bounds: Tensor = torch.tensor(
            data=list(
                chain.from_iterable(
                    [bounds.lower] * self.num_parameter_elements[param]
                    for param, bounds in self.bounds.items()
                )
            ),
            device=self.sample_matrix.device,
            dtype=self.sample_matrix.dtype,
        )
        self.upper_bounds: Tensor = torch.tensor(
            data=list(
                chain.from_iterable(
                    [bounds.upper] * self.num_parameter_elements[param]
                    for param, bounds in self.bounds.items()
                )
            ),
            device=self.sample_matrix.device,
            dtype=self.sample_matrix.dtype,
        )

    def __getitem__(self, key: str) -> Tensor:
        """Get a view of a parameter's samples by name."""
        return self.sample_matrix[:, self.slices[key]].view(
            self.num_samples, *self.shapes[key]
        )

    def __iter__(self) -> Iterable[tuple[str, Tensor]]:
        """Iterate across views of parameter samples."""
        return ((k, self[k]) for k in self.keys())

    def __len__(self) -> int:
        """Get number of parameters."""
        return len(self.bounds)

    def clone(self) -> "ParameterSamples":
        """Return a copy `ParameterSamples` instance."""
        return self.__class__(
            bounds=self.bounds.copy(),
            shapes=self.shapes.copy(),
            sample_matrix=self.sample_matrix.clone(),
        )

    def get_proposed_samples(self) -> "ParameterSamples":
        """Get proposed samples by sampling from proposal distribution.

        Returns
        -------
        proposed_samples: ParameterSamples
        """
        proposed_sample_matrix = (
            self.proposal_distribution()
            .sample()
            .clip(min=self.lower_bounds, max=self.upper_bounds)
        )
        return self.__class__(
            bounds=self.bounds.copy(),
            shapes=self.shapes.copy(),
            sample_matrix=proposed_sample_matrix,
        )

    def items(self) -> Iterable[tuple[str, Tensor]]:
        """Return an iter over parameter names and samples."""
        return iter(self)

    def keys(self) -> KeysView[str]:
        """Get parameter names."""
        return self.bounds.keys()

    def values(self) -> Iterable[Tensor]:
        """Return an iter over parameter samples."""
        return (self[k] for k in self.keys())

    def log_prob(
        self,
        likelihood_func: Callable[P, Tensor],
        prior_func: Callable[P, Tensor] | None,
    ) -> Tensor:
        """Get log probability of samples.

        Parameters
        ----------
        likelihood_func: Callable[..., Tensor] | None
            Function that takes parameter samples as Tensors and returns a single
            Tensor for the joint log likelihood.
        prior_func: Callable[..., Tensor] | None
            Function that takes parameter samples as Tensors and returns a single
            Tensor for the joint prior log probability.

        Returns
        -------
        log_prob: : Tensor, shape=(S,)
            Log probability of samples.

        """
        log_prob = likelihood_func(**self)
        if prior_func is not None:
            log_prob += prior_func(**self)
        return log_prob

    def proposal_distribution(self) -> dist.Normal:
        """Get a normal proposal distribution centered at each parameter sample \
        with scale equal to the standard deviation of each parameter's samples.

        As the sampling distribution of each parameter evolves, the standard
        deviation may shrink, thus annealing the sample movement from iteration
        to iteration.

        Returns
        -------
        proposal_dist: pyro.distributions.Normal, shape=(S, P)
            Normal proposal distribution for each parameter sample.
        """
        # Calculate standard deviation of each parameter distribution
        proposal_scales = self.sample_matrix.std(dim=0, keepdim=True)

        # Build normal proposal distribution centered at each sample
        proposal_dist = dist.Normal(loc=self.sample_matrix, scale=proposal_scales)

        return proposal_dist

    def to(self, *args, **kwargs) -> "ParameterSamples":
        """Perform `ParameterSamples` dtype and/or device conversion."""
        self.sample_matrix = self.sample_matrix.to(*args, **kwargs)
        self.lower_bounds = self.lower_bounds.to(*args, **kwargs)
        self.upper_bounds = self.upper_bounds.to(*args, **kwargs)
        return self

    @classmethod
    def random_initialization(
        cls,
        bounds: dict[str, Bounds],
        shapes: dict[str, torch.Size | tuple[int, ...]],
        num_samples: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = torch.float32,
        seed: int | None = None,
    ) -> "ParameterSamples":
        """Randomly initialize a `ParameterSamples` instance.

        Parameters
        ----------
        bounds: dict[str, Bounds]
            (lower, upper) bounds for parameters: {param_name: (lb, ub), ...}
        shapes: dict[str, torch.Size | tuple[int, ...]]
            Shape of each parameter: {param_name: shape, ...}
        num_samples: int
            Number of MCMC samples for each parameter.
        device: torch.device | None, optional
            Compute device for samples.
            (default = None)
        dtype: torch.dtype | None, optional
            Datatype for samples.
            (default = torch.float32)
        seed: int | None, optional
            Random state seed.
            (default = None)
        """
        sample_matrix = initialize_samples(
            bounds=chain.from_iterable(
                [bounds[param]] * torch.Size(shape).numel()
                for param, shape in shapes.items()
            ),
            num_samples=num_samples,
            device=device,
            dtype=dtype,
            seed=seed,
        )
        return cls(bounds, shapes, sample_matrix)


@dataclass(frozen=True, kw_only=True)
class MCMCResult:
    """Stores the output of a parallel MCMC run.

    Attributes
    ----------
    parameter_samples: ParameterSamples
        Samples of latent model parameters.
    map_values: dict[str, Tensor]
        Maximum a posteriori parameter values observed during sampling.
    correlation_traces: dict[str, Tensor]
        Trace of the Spearman Rank Correlation between current sample distribution
        and the initial sample distribution for each parameter.
    wasserstein_distance_traces: dict[str, Tensor]
        Trace of the Wasserstein distance between split A and B of the sample
        distributions for each parameter.
    split_a_idxs: Tensor, shape=(S / 2, P), dtype=int64
        Split A indexes.
    split_b_idxs: Tensor, shape=(S / 2, P), dtype=int64
        Split B indexes.

    """

    parameter_samples: ParameterSamples
    map_values: dict[str, Tensor]
    correlation_traces: dict[str, Tensor]
    wasserstein_distance_traces: dict[str, Tensor]

    def get_posterior_summary(self) -> pd.DataFrame:
        """Get a summary of posterior parameter samples."""
        columns = []
        for param, numel in self.parameter_samples.num_parameter_elements.items():
            if numel > 1:
                for i in range(numel):
                    columns.append(f"{param}[{i}]")
            else:
                columns.append(param)

        return (
            pd.DataFrame(
                data=self.parameter_samples.sample_matrix.cpu().numpy(),
                columns=columns,
            )
            .describe()
            .drop("count")
            .T
        )


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
    parameter_samples: ParameterSamples,
    likelihood_func: Callable[P, Tensor],
    *,
    prior_func: Callable[P, Tensor] | None = None,
    max_iter: int = 1_000,
    max_corr: float = 0.025,
    max_split_distance: float = 0.005,
    seed: int | None = None,
    verbose: bool = True,
) -> MCMCResult:
    """Run parallel MCMC sampling.

    Parameters
    ----------
    parameter_samples: ParameterSamples
        Samples of latent model parameters.
    likelihood_func: Callable[..., Tensor]
        Function that takes parameter samples as Tensors and returns a single
        Tensor for the joint log likelihood.
    prior_func: Callable[..., Tensor] | None, optional
        Function w/ same parameter signature as `likelihood_func`, that takes
        parameter samples as Tensors and returns a single Tensor for the joint
        prior log probability.
        (default = None)
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

    # Initial samples
    init_samples = parameter_samples
    init_sample_isort = init_samples.sample_matrix.argsort(0)
    split_a_idxs = init_sample_isort[: init_samples.num_samples // 2]
    split_b_idxs = init_sample_isort[init_samples.num_samples // 2 :]

    # Evaluate samples
    samples = parameter_samples.clone()
    sample_scores = samples.log_prob(likelihood_func, prior_func).squeeze(-1)

    # Set MAP parameter values
    map_idx = sample_scores.argmax()
    map_score = sample_scores[map_idx]
    map_values = samples.sample_matrix[map_idx]

    # Update iterations
    correlation_traces = []
    wasserstein_distance_traces = []
    for i in range(max_iter):
        # Generate and evaluate proposed samples
        proposed_samples = samples.get_proposed_samples()
        proposed_sample_scores = proposed_samples.log_prob(
            likelihood_func, prior_func
        ).squeeze(-1)

        # Decide whether to move to proposed for each sample
        move_prob = torch.sigmoid(proposed_sample_scores - sample_scores)
        move_mask = move_prob >= torch.rand_like(move_prob)
        if move_mask.ndim == 1:
            move_mask = move_mask[:, None]
        samples.sample_matrix = torch.where(
            move_mask,
            proposed_samples.sample_matrix,
            samples.sample_matrix,
        )

        # Update samples scores, MAP parameter values, and counts
        sample_scores = torch.where(
            move_mask.squeeze(-1), proposed_sample_scores, sample_scores
        )
        map_idx = sample_scores.argmax()
        if sample_scores[map_idx] > map_score:
            map_score = sample_scores[map_idx]
            map_values = samples.sample_matrix[map_idx]

        # Calculation correlation between initial samples and current samples
        correlation_traces.append(
            get_spearman_corrcoef(init_samples.sample_matrix.T, samples.sample_matrix.T)
        )

        # Calculate Wasserstein distance between split A and B for each parameter
        wasserstein_distance_traces.append(
            get_wasserstein_distance(
                a=torch.gather(samples.sample_matrix, 0, split_a_idxs).T,
                b=torch.gather(samples.sample_matrix, 0, split_b_idxs).T,
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
    correlation_traces = torch.stack(correlation_traces, dim=0)
    wasserstein_distance_traces = torch.stack(wasserstein_distance_traces, dim=0)
    result = MCMCResult(
        parameter_samples=samples,
        map_values={
            param: map_values[sl].view(*samples.shapes[param])
            for param, sl in samples.slices.items()
        },
        correlation_traces={
            param: correlation_traces[:, sl].view(-1, *samples.shapes[param])
            for param, sl in samples.slices.items()
        },
        wasserstein_distance_traces={
            param: wasserstein_distance_traces[:, sl].view(-1, *samples.shapes[param])
            for param, sl in samples.slices.items()
        },
    )

    return result
