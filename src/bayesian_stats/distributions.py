"""Extension of PyTorch and Pyro parameterizable probability distributions."""
from enum import StrEnum
from typing import Callable, NamedTuple

import torch
import pyro.distributions as dist
from einops import rearrange
from torch.distributions import constraints
from torch import Tensor

from bayesian_stats.utils import Bounds, encode_shape, iqr


class DistributionShapes(NamedTuple):
    """Container for distribution shapes.

    ```
          |      iid     | independent | dependent
    ------+--------------+-------------+------------
    shape = sample_shape + batch_shape + event_shape
    ```

    https://pyro.ai/examples/tensor_shapes.html

    Attributes
    ----------
    sample_shape: torch.Size
        The shape of iid samples drawn from the distribution.
    batch_shape: torch.Size
        The non-identical (independent) parameterizations of the distribution,
        inferred from the distribution's parameter shapes.
    event_shape: torch.Size
        The atomic shape of a single event/observation from the
        distribution (or batch of distributions of the same family).
    """

    sample_shape: torch.Size
    batch_shape: torch.Size
    event_shape: torch.Size

    @property
    def shape(self) -> torch.Size:
        """Get combined shape of `sample`, `batch` and `event` dimensions."""
        return self.sample_shape + self.batch_shape + self.event_shape

    def numel(self) -> int:
        """Get number of elements implied by shape."""
        return self.shape.numel()

    @classmethod
    def from_sample_data_shape(
        cls,
        data_shape: torch.Size,
        *,
        sample_shape: torch.Size | None = None,
        batch_shape: torch.Size | None = None,
        event_shape: torch.Size | None = None,
    ) -> "DistributionShapes":
        """Factor the shape of samples from a distribution into `sample`, \
            `batch`, and `event` components.

        Parameters
        ----------
        data_shape: torch.Size
            Shape of sample data.
        sample_shape: torch.Size, optional
            Known sample dimensions.
            If `None`, assumes the first dimension.
        batch_shape: torch.Size, optional
            Known batch dimensions.
            If `None`, assumes the remaining dimensions after `sample_shape`.
        event_shape: torch.Size, optional
            Known event dimensions.
            If `None`, assumes the remaining dimensions after `sample_shape`
            and `batch_shape`.

        Returns
        -------
        DistributionShapes
        """
        # Copy of input data shape to check shape constituents
        exp_shape = data_shape[:]

        if sample_shape is None:
            # Assume first dimension is sample shape
            sample_shape = data_shape[:1]
            data_shape = data_shape[1:]
        else:
            # Remove sample shape elements
            data_shape = data_shape[len(sample_shape) :]

        if batch_shape is None:
            # Assume remainder of elements are batch shape
            batch_shape = data_shape[:]
            data_shape = torch.Size()
        else:
            # Remove batch shape elements
            data_shape = data_shape[len(batch_shape) :]

        if event_shape is None:
            # Assume remainder of elements are event shape
            event_shape = data_shape[:]

        if exp_shape != (sample_shape + batch_shape + event_shape):
            raise ValueError(
                f"Shapes {(sample_shape + batch_shape + event_shape)} "
                f"don't add up to expected {exp_shape}"
            )

        return cls(
            sample_shape=sample_shape,
            batch_shape=batch_shape,
            event_shape=event_shape,
        )


class Kernel(StrEnum):
    """Kernels for density estimation."""

    GAUSSIAN = "gaussian"

    @property
    def log_func(self) -> Callable[[Tensor], Tensor]:
        """Get (log) kernel function."""
        match self:
            case self.GAUSSIAN:
                return gaussian_kernel
            case _:
                raise NotImplementedError


class KDEDistribution(dist.TorchDistribution):
    """Kernel density estimated (KDE) distribution from sample data."""

    arg_constraints = {}
    has_enumerate_support = True

    def __init__(
        self,
        samples: Tensor,
        bounds: Bounds,
        *,
        kernel: Kernel = Kernel.GAUSSIAN,
        bandwidth: float | Tensor | None = None,
        sample_shape: torch.Size | None = None,
        batch_shape: torch.Size | None = None,
        event_shape: torch.Size | None = None,
        validate_args: bool | None = None,
    ) -> None:
        """Initialize a KDEDistribution.

        Parameters
        ----------
        samples: Tensor
            Samples from distribution to use for density estimation.
        bounds: Bounds
            Bounds of distribution.
        kernel: Kernel, optional
            Kernel to use for density estimation.
            (default = Kernel.GAUSSIAN)
        bandwidth: float | Tensor, optional
            Bandwidth for density estimation.
            If `None` will use Silverman's rule of thumb.
            (default = None)
        sample_shape: torch.Size, optional
            Known sample dimensions. If None, assumes the first dimension.
            (default = None)
        batch_shape: torch.Size, optional
            Known batch dimensions.
            If `None`, assumes the remaining dimensions after `sample_shape`.
        event_shape: torch.Size, optional
            Known event dimensions.
            If `None`, assumes the remaining dimensions after `sample_shape`
            and `batch_shape`.
        validate_args: bool, optional
            (default = None)
        """
        # Init TorchDistribution parent class & shape and bounds properties
        (
            sample_shape,
            batch_shape,
            event_shape,
        ) = DistributionShapes.from_sample_data_shape(
            data_shape=samples.shape,
            sample_shape=sample_shape,
            batch_shape=batch_shape,
            event_shape=event_shape,
        )
        if event_shape != torch.Size():
            raise NotImplementedError(
                "KDEDistribution currently supports univariate distributions; "
                f"got sample data with event shape {event_shape}"
            )
        super().__init__(
            batch_shape=batch_shape,
            event_shape=event_shape,
            validate_args=validate_args,
        )
        self.num_samples: int = sample_shape.numel()
        self.ex_samples_shape: torch.Size = batch_shape + event_shape
        self.bounds = Bounds(*bounds)
        self.lower_bound = torch.full(
            size=self.ex_samples_shape,
            fill_value=self.bounds.lower,
            dtype=samples.dtype,
            device=samples.device,
        )
        self.upper_bound = torch.full(
            size=self.ex_samples_shape,
            fill_value=self.bounds.upper,
            dtype=samples.dtype,
            device=samples.device,
        )

        # Flatten & sort sample values
        samples = (
            samples.reshape((-1, *self.ex_samples_shape)).sort(dim=0).values
        )
        # Move sample dim to inner most dim (for torch.searchsorted)
        self.samples = rearrange(samples, "S ... -> ... S").contiguous()

        # Set kernel, calculate bandwidth, and estimate density of samples
        self._init_sample_density(samples, kernel, bandwidth)

    def __repr__(self) -> str:
        """Get string representation."""
        return f"{self.__class__.__name__}()"

    def sample(self, sample_shape: torch.Size = torch.Size([])) -> Tensor:
        """Get `sample_shape` randomly selected samples from distribution.

        Parameters
        ----------
        sample_shape: torch.Size
            Shape of samples to return.

        Returns
        -------
        values: Tensor
        """
        # Draw random samples from sample pool
        idxs = torch.randint(low=0, high=self.num_samples, size=sample_shape)
        values = self.samples[..., idxs]

        # Rearrange so sampling dimensions are first (outermost) dims
        ss = encode_shape(sample_shape)
        values = rearrange(values, f"... {ss} -> {ss} ...").contiguous()

        return values

    def log_prob(self, value: Tensor, interpolate: bool = True) -> Tensor:
        """Evaluate the log probability density of each event in a batch.

        Parameters
        ----------
        value: Tensor
            Values to estimate log-density for.
        interpolate: bool, optional
            True: interpolate density from stored samples & densities
            False: apply kernel density estimation to values
            (default = True)

        Returns
        -------
        log_density: Tensor
        """
        value_shape = value.shape
        num_sample_elements = len(value_shape) - len(self.ex_samples_shape)
        if (
            num_sample_elements < 0
            or value.shape[num_sample_elements:] != self.ex_samples_shape
        ):
            raise ValueError(
                "Incorrect shape for `value`; expecting (..., "
                + ", ".join(str(d) for d in self.ex_samples_shape)
                + f"); got {tuple(value.shape)}"
            )

        return (
            self._log_prob_interp(value, num_sample_elements)
            if interpolate
            else self._log_prob_est(value)
        )

    def to(self, *args, **kwargs) -> "KDEDistribution":
        """Perform dtype and/or device conversion for Tensor properties."""
        self.bandwidth = self.bandwidth.to(*args, **kwargs)
        self.samples = self.samples.to(*args, **kwargs)
        self.lower_bound = self.lower_bound.to(*args, **kwargs)
        self.upper_bound = self.upper_bound.to(*args, **kwargs)
        self.sample_density = self.samples.to(*args, **kwargs)
        self.lower_bound_density = self.lower_bound_density.to(*args, **kwargs)
        self.upper_bound_density = self.upper_bound_density.to(*args, **kwargs)

    def _init_sample_density(
        self, samples: Tensor, kernel: Kernel, bandwidth: float | Tensor | None
    ) -> None:
        """Initialize sample density estimates for distribution."""
        # Set kernel function for density estimation
        self.kernel = kernel

        # Set bandwidth for density estimation
        if bandwidth is None:
            self.bandwidth = get_bandwidth(
                sample_size=self.num_samples,
                data_dim=self.event_shape.numel(),
                sample_std=self.samples.std(dim=-1),
                sample_iqr=iqr(self.samples, dim=-1),
            )
        elif isinstance(bandwidth, float):
            self.bandwidth = torch.full(
                size=self.ex_samples_shape,
                fill_value=bandwidth,
            )
        else:
            self.bandwidth = bandwidth
        self.bandwidth = self.bandwidth.to(samples)

        # Append bounds to samples to get density estimates for them
        samples_n_bounds = torch.cat(
            (
                self.lower_bound[None, ...],
                samples,
                self.upper_bound[None, ...],
            ),
            dim=0,
        )

        # Estimate log-density for each sample & bounds of distribution
        density = self._log_prob_est(samples_n_bounds)
        self.lower_bound_density = density[..., 0]
        self.upper_bound_density = density[..., -1]
        density = density[..., 1:-1]

        # Move sample dim to inner most dim (for torch.searchsorted)
        self.sample_density = rearrange(density, "S ... -> ... S").contiguous()

    def _log_prob_est(self, value: Tensor) -> Tensor:
        """Estimate log-density of values using distribution's kernel."""
        # Diff of value to every sample
        diffs = value[..., None] - self.samples[None, ...]
        diffs = diffs / (self.bandwidth[None, ..., None])

        # Apply kernel to diffs to estimate density
        log_density = torch.logsumexp(self.kernel.log_func(diffs), dim=-1)
        log_density = log_density - torch.log(
            self.num_samples * self.bandwidth[None, ...]
        )

        return log_density

    def _log_prob_interp(
        self, value: Tensor, num_sample_elements: int
    ) -> Tensor:
        """Interpolate log-density of values from density of stored samples."""
        sample_shape = value.shape[:num_sample_elements]
        if not sample_shape:
            # Add phantom sample dimension
            value = value.unsqueeze(0)

        # Flatten and move sample dim to inner most dim for torch.searchsorted
        value = value.reshape((-1, *self.ex_samples_shape))
        value = rearrange(value, "S ... -> ... S").contiguous()

        # Append bounds to sorted samples & density
        sorted_samples = torch.cat(
            (
                self.lower_bound[..., None],
                self.samples,
                self.upper_bound[..., None],
            ),
            dim=-1,
        )
        sample_density = torch.cat(
            (
                self.lower_bound_density[..., None],
                self.sample_density,
                self.upper_bound_density[..., None],
            ),
            dim=-1,
        )

        # Find index of value in distribution's sorted samples
        # samples[i-1] <= value < samples[i]
        idxs = torch.searchsorted(sorted_samples, value, right=True)
        lb_idxs = (idxs - 1).clip(min=0, max=self.num_samples + 1)
        ub_idxs = idxs.clip(min=0, max=self.num_samples + 1)

        # Look up lower and upper density bounds by index
        lb_density = torch.gather(sample_density, dim=-1, index=lb_idxs)
        ub_density = torch.gather(sample_density, dim=-1, index=ub_idxs)
        density_rng = ub_density - lb_density

        # Calculate interpolation % between lower and upper density
        lb_value = torch.gather(sorted_samples, dim=-1, index=lb_idxs)
        ub_value = torch.gather(sorted_samples, dim=-1, index=ub_idxs)
        interp_p = torch.where(
            ub_value > lb_value,
            (value - lb_value) / (ub_value - lb_value),
            0.0,
        )

        # Calculate interpolated density
        log_density = lb_density + density_rng * interp_p

        # Move densities from inner most to outer most dimension
        log_density = rearrange(log_density, "... S -> S ...")

        # Restore original sample dimension
        log_density = log_density.reshape(
            *sample_shape, *self.ex_samples_shape
        )

        return log_density

    @property
    def support(self) -> constraints.Constraint:
        """Get distribution's support from bounds."""
        return constraints.interval(*self.bounds)


def gaussian_kernel(x: Tensor) -> Tensor:
    """Apply Gaussian kernel to input."""
    return dist.Normal(loc=0.0, scale=1.0).log_prob(x)


def get_bandwidth(
    sample_size: int,
    data_dim: int,
    sample_std: Tensor,
    sample_iqr: Tensor,
) -> Tensor:
    """Use Silverman's rule of thumb to estimate bandwidth for kernel \
        density estimation.

    Parameters
    ----------
    sample_size: int
        Number of i.i.d. samples to estimate bandwidth from.
    data_dim: int
        Dimension of the data distribution (e.g. univariate = 1)
    sample_std: Tensor, shape(B?, D)
        Sample standard deviation.
    sample_iqr: Tensor, shape(B?, D)
        Sample interquartile range.

    Returns
    -------
    bandwidth: Tensor, shape(B?, D)

    Notes
    -----
    - The bandwidth of the kernel is a free parameter which exhibits a
    strong influence on the resulting estimate.
    - Silverman's rule of thumb may result in oversmoothed density estimations
    if the sample size is small and/or if the data is multi-modal.
    """
    return (
        0.9
        * torch.minimum(sample_std, sample_iqr / 1.34)
        * sample_size ** (-1.0 / (data_dim + 4))
    )
