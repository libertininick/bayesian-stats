from dataclasses import dataclass
from typing import Union, List

import numpy as np
from numpy.typing import NDArray


@dataclass
class MarketDecline:
    start: np.datetime64
    end: np.datetime64
    length: int
    size: float


def find_market_declines(
    dates: NDArray[np.datetime64],
    prices: NDArray[np.float64],
    min_length: int = 21,
    min_size: float = -0.075,
    market_declines: Union[List[MarketDecline], None] = None,
) -> List[MarketDecline]:
    """Builds market decline objects from a sequence of prices.

    Parameters
    ----------
    dates: NDArray[np.datetime64], shape=(N,)
        Sequence of dates.
    prices: NDArray[np.float64], shape=(N,)
        Sequence of prices
    min_length: int, optional
        Minimum number of days from peak to trough to be considered a decline.
        (default = 21)
    min_size: float, optional
        Minimum size of drawdown to be considered a decline (expressed as a negative number)
        (default = -7.5%)
    market_declines: List[MarketDecline], optional
        Running set of declines.
        Use for recursive call when finding additional declines in a long drawdown period.
        (default = None)

    Returns
    -------
    market_declines: List[MarketDecline]
      List of declines in price sequence that meet min length/size filter.
    """

    if len(dates) != len(prices):
        raise ValueError("Number of dates and prices must match")

    if market_declines is None:
        market_declines = []

    # Find all drawdowns
    high_water = np.maximum.accumulate(prices)
    drawdown_from_hw = prices / high_water - 1

    # Assign prices to a group
    # group := [new high, next new high)
    at_new_high = drawdown_from_hw == 0
    grp_ids = np.cumsum(at_new_high)

    # Build market decline objs from drawdown groups
    for grp_id in np.unique(grp_ids):
        mask = grp_ids == grp_id
        grp_dates = dates[mask]
        grp_prices = prices[mask]

        if mask.sum() < min_length + 1:
            # Decline is too short
            continue

        # Size of decline = max drawdown
        size = grp_prices.min() / grp_prices[0] - 1

        # Length of decline = # days to reach max drawdown point
        length = np.argmin(grp_prices)

        if size <= min_size and length >= min_length:
            # Add decline to list
            market_declines.append(
                MarketDecline(
                    start=grp_dates[1],
                    end=grp_dates[length],
                    size=size,
                    length=length,
                )
            )

            # Check if recovery period had any declines
            market_declines = find_market_declines(
                dates=grp_dates[length + 1 :],
                prices=grp_prices[length + 1 :],
                min_length=min_length,
                min_size=min_size,
                market_declines=market_declines,
            )

    return market_declines


def in_market_decline(
    dates: NDArray[np.datetime64],
    market_declines: List[MarketDecline],
) -> NDArray[np.bool_]:
    "Flag whether a date is in a market decline period"
    masks = np.stack(
        [np.logical_and(dates >= md.start, dates <= md.end) for md in market_declines],
        axis=-1,
    )
    return masks.any(-1)
