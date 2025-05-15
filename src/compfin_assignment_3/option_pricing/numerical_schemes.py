"""Numerical schemes for solving Heston model."""

import math
from typing import Optional

import numpy as np
import numpy.typing as npt
from tqdm import trange

from compfin_assignment_3.option_pricing.model_settings import HestonModelSettings


class NumScheme:
    """Numerical methods for solving Heston model."""

    def __init__(
        self,
        config: HestonModelSettings,
        asset_stochastic_increments: Optional[npt.NDArray[np.float64]],
    ) -> None:
        """Initialize EulerScheme object."""
        if asset_stochastic_increments is not None:
            assert asset_stochastic_increments.shape[0] == config.n_trajectories
            assert asset_stochastic_increments.shape[1] == config.num_steps

        self.config = config
        self.num_scheme = None
        self.asset_stochastic_increments = asset_stochastic_increments

    @classmethod
    def heston_model_simulation(
        cls,
        config: HestonModelSettings,
        numerical_scheme: str,
        stochastic_increments: Optional[npt.NDArray[np.float64]] = None,
        logging: bool = False,
    ) -> npt.NDArray[np.float64]:
        """Simulate trajectories of the Heston model using the chosen numerical scheme."""
        model_sim = cls(config, stochastic_increments)
        model_sim.set_numerical_scheme(numerical_scheme)

        iterator = (
            trange(
                model_sim.config.n_trajectories,
                desc=f"{numerical_scheme.upper()} scheme simulation",
            )
            if logging
            else range(model_sim.config.n_trajectories)
        )

        return np.array([model_sim.simulate_trajectory(traj_idx) for traj_idx in iterator])

    @classmethod
    def gbm_model_simulation(
        cls,
        config: HestonModelSettings,
        numerical_scheme: str = "euler",
        stochastic_increments: Optional[npt.NDArray[np.float64]] = None,
        logging: bool = False,
    ) -> npt.NDArray[np.float64]:
        """Simulate trajectories of the GBM model using the chosen numerical scheme."""
        model_sim = cls(config, stochastic_increments)
        model_sim.set_numerical_scheme(numerical_scheme)

        model_sim._reducing_heston_to_gbm()

        iterator = (
            trange(
                model_sim.config.n_trajectories,
                desc=f"{numerical_scheme.upper()} scheme simulation",
            )
            if logging
            else range(model_sim.config.n_trajectories)
        )

        return np.array([model_sim.simulate_trajectory(traj_idx) for traj_idx in iterator])

    def simulate_trajectory(self, traj_idx: int, min_var: float = 0) -> list:
        """Simulate a single trajectory of the Heston model using the chosen numerical scheme."""
        s_t = [self.config.s_0]
        v_t = [self.config.v_0]
        stochastic_increments = self._simulate_stochastic_increments(
            self.asset_stochastic_increments[traj_idx]
            if self.asset_stochastic_increments is not None
            else None
        )

        for i in range(self.config.num_steps):
            next_s, next_v = self.num_scheme(
                s_t[-1], v_t[-1], stochastic_increments[i, :], min_var
            )
            s_t.append(next_s)
            v_t.append(next_v)

        return s_t

    def _simulate_stochastic_increments(
        self, asset_stoch_inc: Optional[npt.NDArray[np.float64]]
    ) -> npt.NDArray[np.float64]:
        """Simulate correlated stochastic increments for the asset and its variance processes."""
        if asset_stoch_inc is not None:
            z_1 = asset_stoch_inc / math.sqrt(self.config.step_size)
            z_2 = np.random.normal(size=asset_stoch_inc.shape[0])
            x_2 = z_1 * self.config.stoc_inc_corr + z_2 * math.sqrt(
                1 - self.config.stoc_inc_corr**2
            )
            x_2 = x_2 * math.sqrt(self.config.step_size)
            return np.vstack((asset_stoch_inc, x_2)).T
        else:
            cov_mat = np.array([[1, self.config.stoc_inc_corr], [self.config.stoc_inc_corr, 1]])
            stochastic_increments = np.random.multivariate_normal(
                mean=np.zeros(2), cov=cov_mat, size=self.config.num_steps
            )
            return stochastic_increments * np.sqrt(self.config.step_size)

    def set_numerical_scheme(self, numerical_scheme: str) -> None:
        """Get the function for updating the asset and variance process."""
        match numerical_scheme:
            case "euler":
                self.num_scheme = self.euler_scheme_update
            case "milstein":
                self.num_scheme = self.milstein_scheme_update
            case _:
                raise ValueError("Invalid numerical scheme.")

    def euler_scheme_update(
        self, curr_s: float, curr_v: float, stoc_inc: npt.NDArray[np.float64], min_var: float
    ) -> tuple[float, float]:
        """Calculates the next asset and the variance process element using Euler scheme."""
        next_s = float(
            curr_s
            + self.config.drift * curr_s * self.config.step_size
            + math.sqrt(curr_v) * curr_s * stoc_inc[0]
        )
        next_v = max(
            min_var,
            float(
                curr_v
                + self.config.kappa * (self.config.theta - curr_v) * self.config.step_size
                + self.config.vol_of_vol * math.sqrt(curr_v) * stoc_inc[1]
            ),
        )
        return next_s, next_v

    def milstein_scheme_update(
        self, curr_s: float, curr_v: float, stoc_inc: npt.NDArray[np.float64], min_var: float
    ) -> tuple[float, float]:
        """Calculates the next asset and the variance process element using Milstein scheme."""
        next_s = float(
            curr_s
            + self.config.drift * curr_s * self.config.step_size
            + math.sqrt(curr_v) * curr_s * stoc_inc[0]
            + 0.5 * curr_v * curr_s * (stoc_inc[0] * stoc_inc[0] - self.config.step_size)
        )
        next_v = max(
            min_var,
            float(
                curr_v
                + self.config.kappa * (self.config.theta - curr_v) * self.config.step_size
                + self.config.vol_of_vol * math.sqrt(curr_v) * stoc_inc[1]
                + 0.25
                * self.config.vol_of_vol
                * self.config.vol_of_vol
                * (stoc_inc[1] * stoc_inc[1] - self.config.step_size)
            ),
        )
        return next_s, next_v

    def _reducing_heston_to_gbm(self) -> None:
        """Reduces the Heston model to the GBM model by changing the parameters.."""
        self.config.kappa = 0
        self.config.v_0 = self.config.theta
        self.config.vol_of_vol = 0
