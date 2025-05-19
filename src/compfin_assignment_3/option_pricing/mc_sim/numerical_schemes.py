"""Numerical schemes for solving GBM model."""

from typing import Optional

import numba as nb
import numpy as np
import numpy.typing as npt

from compfin_assignment_3.option_pricing.mc_sim.settings import BSModelSettings


class NumScheme:
    """Numerical methods for solving GBM model."""

    def __init__(
        self,
        config: BSModelSettings,
        asset_stochastic_increments: Optional[npt.NDArray[np.float64]],
    ) -> None:
        """Initialize NumScheme object."""
        if asset_stochastic_increments is not None:
            assert asset_stochastic_increments.shape[0] == config.n_trajectories
            assert asset_stochastic_increments.shape[1] == config.num_steps

        self.config = config
        self.num_scheme = None
        self.asset_stochastic_increments = (
            asset_stochastic_increments
            if asset_stochastic_increments is not None
            else self._simulate_stochastic_increments()
        )

    @classmethod
    def asset_price_simulation(
        cls,
        config: BSModelSettings,
        numerical_scheme: str,
        stochastic_increments: Optional[npt.NDArray[np.float64]] = None,
    ) -> npt.NDArray[np.float64]:
        """Simulate trajectories of the GBM model using the chosen numerical scheme."""
        model_sim = cls(config, stochastic_increments)
        model_sim.set_numerical_scheme(numerical_scheme)

        return model_sim.num_scheme(
            s_0=config.s_0,
            drift=config.drift,
            sigma=config.sigma,
            step_size=config.step_size,
            n_step=config.num_steps,
            n_trajectories=config.n_trajectories,
            stoc_increments=model_sim.asset_stochastic_increments,
        )
        # return model_sim.num_scheme()

    def _simulate_stochastic_increments(self) -> npt.NDArray[np.float64]:
        """Simulate stochastic increments for the asset price process."""
        return np.random.normal(
            loc=0, scale=1, size=(self.config.n_trajectories, self.config.num_steps)
        ) * np.sqrt(self.config.step_size)

    def set_numerical_scheme(self, numerical_scheme: str) -> None:
        """Get the function for updating the asset and variance process."""
        match numerical_scheme:
            case "euler":
                self.num_scheme = self.euler_scheme
            case "milstein":
                self.num_scheme = self.milstein_scheme
            case _:
                raise ValueError("Invalid numerical scheme.")

    @staticmethod
    @nb.njit(fastmath=True)
    def euler_scheme(
        s_0: float,
        drift: float,
        sigma: float,
        step_size: int,
        n_step: int,
        n_trajectories: int,
        stoc_increments: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Calculates n_trajectories number of trajectories using Euler scheme."""
        s_t = np.zeros((n_trajectories, n_step + 1))
        s_t[:, 0] = s_0

        for col_idx in range(1, n_step + 1):
            s_t[:, col_idx] = (
                s_t[:, col_idx - 1]
                + drift * s_t[:, col_idx - 1] * step_size
                + sigma * s_t[:, col_idx - 1] * stoc_increments[:, col_idx - 1]
            )
        return s_t

    @staticmethod
    @nb.njit(fastmath=True)
    def milstein_scheme(
        s_0: float,
        drift: float,
        sigma: float,
        step_size: int,
        n_step: int,
        n_trajectories: int,
        stoc_increments: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Calculates n_trajectories number of trajectories using Milstein scheme."""
        s_t = np.zeros((n_trajectories, n_step + 1))
        s_t[:, 0] = s_0

        for col_idx in range(1, n_step + 1):
            s_t[:, col_idx] = (
                s_t[:, col_idx - 1]
                + drift * s_t[:, col_idx - 1] * step_size
                + sigma * s_t[:, col_idx - 1] * stoc_increments[:, col_idx - 1]
                + 0.5
                * sigma
                * sigma
                * s_t[:, col_idx - 1]
                * (stoc_increments[:, col_idx - 1] * stoc_increments[:, col_idx - 1] - step_size)
            )
        return s_t

    # def euler_scheme(self) -> npt.NDArray[np.float64]:
    #     """Calculates n_trajectories number of trajectories using Euler scheme."""
    #     drift = self.config.drift * self.config.step_size
    #     stochastic_increments = self.asset_stochastic_increments * self.config.sigma
    #
    #     price_increments = drift + stochastic_increments + 1
    #     price_increments = np.hstack((np.ones((self.config.n_trajectories, 1)) * self.config.s_0,
    #                                   price_increments))
    #
    #     price = np.cumprod(price_increments, axis=1)
    #     return price
    #
    # def milstein_scheme(self) -> npt.NDArray[np.float64]:
    #     """Calculates n_trajectories number of trajectories using Milstein scheme."""
    #     drift = self.config.drift * self.config.step_size
    #     stochastic_increments = (self.asset_stochastic_increments * self.config.sigma
    #                              + 0.5 * (self.config.sigma * self.config.sigma)
    #                              * (self.asset_stochastic_increments
    #                                 * self.asset_stochastic_increments
    #                                 - self.config.step_size))
    #
    #     price_increments = drift + stochastic_increments + 1
    #     price_increments = np.hstack((np.ones((self.config.n_trajectories, 1)) * self.config.s_0,
    #                                   price_increments))
    #
    #     price = np.cumprod(price_increments, axis=1)
    #     return price
