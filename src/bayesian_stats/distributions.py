"""Extension of PyTorch and Pyro parameterizable probability distributions."""
from enum import StrEnum
from typing import Callable, NamedTuple

import torch
from pyro.distributions import TorchDistribution
from torch.distributions import constraints
from torch import Tensor

from bayesian_stats.utils import Bounds, iqr


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
    def func(self) -> Callable[[Tensor], Tensor]:
        """Get kernel function."""
        match self:
            case self.GAUSSIAN:
                return gaussian_kernel
            case _:
                raise NotImplementedError


class KDEDistribution(TorchDistribution):
    """Kernel density estimated (KDE) distribution from sample data."""

    arg_constraints = {}
    has_enumerate_support = True

    def __init__(
        self,
        samples: Tensor,
        bounds: Bounds,
        *,
        kernel: Kernel = Kernel.GAUSSIAN,
        bandwidth: float | None = None,
        sample_shape: torch.Size | None = None,
        batch_shape: torch.Size | None = None,
        event_shape: torch.Size | None = None,
        validate_args: bool | None = None,
    ) -> None:
        self.bounds = bounds

        # Get shape properties of distribution
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
        self.num_samples: int = sample_shape.numel()
        self.exsamples_shape: torch.Size = batch_shape + event_shape

        super().__init__(
            batch_shape=batch_shape,
            event_shape=event_shape,
            # support=support,
            validate_args=validate_args,
        )

        # Flatten samples
        self._samples = samples.reshape((-1, *self.exsamples_shape))

        # Move sample dim to inner most dim (for distance calculations)
        dims = range(self._samples.ndim)
        self._samples = self._samples.permute(*dims[1:], 0).contiguous()

        # Set kernel and bandwidth for density estimation
        self.kernel = kernel
        if bandwidth is None:
            # Use rule of thumb for bandwidth
            s_std = self._samples.std(dim=-1)
            s_iqr = iqr(self._samples, dim=-1) / 1.34
            d = self.event_shape.numel()
            self.bandwidth: Tensor = (
                0.9
                * torch.minimum(s_std, s_iqr)
                * self.num_samples ** (-1.0 / (d + 4))
            ).to(self._samples)
        else:
            self.bandwidth = torch.tensor(bandwidth).to(self._samples)

    def sample(self, sample_shape: torch.Size = torch.Size([])) -> Tensor:
        """Get `sample_shape` randomly selected samples from distribution."""
        # Draw random sample indexes
        ridxs = torch.randint(low=0, high=self.num_samples, size=sample_shape)
        samples = self._samples[..., ridxs]

        # Permute so sampling dimensions are first (outermost) dims
        dims = range(samples.ndim)
        samples = samples.permute(
            *dims[-len(sample_shape) :], dims[: -len(sample_shape)]
        ).contiguous()

        return samples

    def log_prob(self, value: Tensor) -> Tensor:
        """Evaluate the log probability density of each event in a batch."""
        diffs = value[..., None] - self._samples[None, ...]
        density = self.kernel.func(diffs / self.bandwidth).sum(-1)
        density = density / (self.num_samples * self.bandwidth)
        return torch.log(density)

    @property
    def support(self) -> constraints.Constraint:
        """Get distribution's support from bounds."""
        return constraints.interval(*self.bounds)


def gaussian_kernel(x: Tensor) -> Tensor:
    """Apply Gaussian kernel to input."""
    return 1 / (2 * torch.pi) ** 0.5 * torch.exp(-(x**2) / 2)
