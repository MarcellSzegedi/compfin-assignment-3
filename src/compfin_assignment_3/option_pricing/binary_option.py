"""Simulate and analytically price a binary option."""

import math
from functools import partial

import numpy as np
import numpy.typing as npt
import scipy.stats as stats

from compfin_assignment_3.option_pricing.mc_sim_settings import BSModelSettings
from compfin_assignment_3.option_pricing.numerical_schemes import NumScheme


class BinaryOption:
    """Simulate and analytically price a binary option."""

    def __init__(self, config: BSModelSettings, numerical_scheme: str = "euler") -> None:
        """Initialize BinaryOption object."""
        self.config = config
        self.num_scheme = numerical_scheme

    @classmethod
    def simulate_binary_option(
        cls,
        config: BSModelSettings,
        numerical_scheme: str,
        stochastic_increments: npt.NDArray[np.float64] = None,
    ) -> tuple[float, float, float]:
        """Simulate and price a binary option."""
        model_sim = cls(config, numerical_scheme)

        num_sim_func = model_sim.set_up_numerical_scheme()
        simulated_trajectories = num_sim_func(config, stochastic_increments=stochastic_increments)

        payoffs = model_sim.calculate_payoffs(simulated_trajectories)
        discounted_payoffs = model_sim.discount_payoffs(payoffs)

        price, conf_u, conf_l = model_sim.calculate_statistics(discounted_payoffs)
        return price, conf_u, conf_l

    @classmethod
    def compute_binary_option_price_analytical(
        cls,
        config: BSModelSettings,
    ) -> float:
        """Compute the price of a binary option."""
        model_sim = cls(config)
        return model_sim.calculate_analytical_price()

    def calculate_analytical_price(self) -> float:
        """Calculate the price of a binary option using analytical formula."""
        d2 = self.d2_calculation()
        return np.exp(-self.config.risk_free_rate * self.config.t_end) * stats.norm.cdf(d2)

    def d2_calculation(self) -> float:
        """Calculate the d2 term for the analytical formula."""
        return (
            math.log(self.config.s_0 / self.config.strike)
            + (self.config.risk_free_rate - 0.5 * self.config.sigma**2) * self.config.t_end
        ) / (self.config.sigma * math.sqrt(self.config.t_end))

    def set_up_numerical_scheme(self) -> callable:
        """Sets up the numerical scheme."""
        match self.num_scheme:
            case "euler":
                return partial(NumScheme.asset_price_simulation, numerical_scheme="euler")
            case "milstein":
                return partial(NumScheme.asset_price_simulation, numerical_scheme="milstein")
            case _:
                raise ValueError("Invalid numerical scheme.")

    def calculate_payoffs(self, trajectories: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Calculates payoffs for each simulated trajectory."""
        return np.array(trajectories[:, -1] > self.config.strike, dtype=np.float64)

    def discount_payoffs(self, payoffs: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Calculates discounted payoffs for each simulated trajectory."""
        return payoffs * np.exp(-self.config.risk_free_rate * self.config.t_end)

    def calculate_statistics(
        self, measurements: npt.NDArray[np.float64]
    ) -> tuple[float, float, float]:
        """Calculates statistics for the simulated trajectories (mean and confidence interval)."""
        std = float(np.std(measurements, ddof=1))
        standard_error = std / np.sqrt(len(measurements))

        mean = np.mean(measurements)
        conf_upper_bound = mean + stats.norm.ppf(1 - self.config.alpha / 2) * standard_error
        conf_lower_bound = mean - stats.norm.ppf(1 - self.config.alpha / 2) * standard_error
        return float(mean), float(conf_upper_bound), float(conf_lower_bound)
