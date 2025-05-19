"""Calculates the delta of a binary option."""

import numba as nb
import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, Field


class BinaryDeltaSettings(BaseModel):
    """Settings for the binary delta calculation."""

    s_min: float = Field(..., gt=0, description="Minimum asset price.")
    s_max: float = Field(..., gt=0, description="Maximum asset price.")
    n_step: int = Field(..., gt=0, description="Number of grid points.")
    strike: float = Field(..., description="Strike price of the option.")
    risk_free_rate: float = Field(..., description="Risk-free rate of interest.")
    t_end: float = Field(..., description="End time of the option.")
    sigma: float = Field(..., description="Volatility of the asset price process.")


class BinaryOptionDelta:
    """Calculates the delta of a binary option."""

    def __init__(self, config: BinaryDeltaSettings) -> None:
        """Initialize BinaryOptionDelta object."""
        self.config = config

    @classmethod
    def delta_calculation(
        cls, config: BinaryDeltaSettings
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Calculates the delta of a binary opt. as a function of asset price and maturity time."""
        bin_delta = cls(config)
        prices = np.linspace(config.s_min, config.s_max, config.n_step, dtype=np.float64)
        tau_values = np.linspace(0.01, config.t_end, config.n_step, dtype=np.float64)

        delta_values = bin_delta.calc_delta(
            prices, tau_values, config.risk_free_rate, config.sigma, config.strike
        )
        return prices, tau_values, delta_values

    @staticmethod
    @nb.njit(fastmath=True)
    def calc_delta(
        prices: npt.NDArray[np.float64],
        tau_values: npt.NDArray[np.float64],
        r: float,
        sigma: float,
        strike: float,
    ) -> npt.NDArray[np.float64]:
        """Calculates the delta of a binary option."""
        delta_values = np.zeros((prices.shape[0], tau_values.shape[0]), dtype=np.float64)
        for i, price in enumerate(prices):
            for j, tau in enumerate(tau_values):
                d2 = (np.log(price / strike) + (r - 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
                pdf_d2 = np.exp(-0.5 * d2**2) / np.sqrt(2 * np.pi)
                delta_values[i, j] = np.exp(-r * tau) * pdf_d2 / (price * sigma * np.sqrt(tau))
        return delta_values


# config = BinaryDeltaSettings(
#     s_min=0.2, s_max=3, n_step=1000,
#     strike=1, risk_free_rate=0.01,
#     t_end=1.0, sigma=0.2
# )
#
# prices, taus, delta = BinaryOptionDelta.delta_calculation(config)
#
# alma = 1
