"""Pricing binary option using heat equation simulation."""

import math

import numpy as np
import numpy.typing as npt
from scipy.linalg import solve_banded
from tqdm import trange

from compfin_assignment_3.option_pricing.heat_equation_sim.settings import HeatEquationSettings


class HeatEquationPricing:
    """Pricing binary option using heat equation simulation."""

    def __init__(self, config: HeatEquationSettings) -> None:
        """Initialize HeatEquationPricing object."""
        self.config = config

    @classmethod
    def calculate_binary_option_price(
        cls, config: HeatEquationSettings
    ) -> npt.NDArray[np.float64]:
        """Calculate the price of a binary option using heat equation simulation."""
        heat_eq_pricing = cls(config)

        spatial_grid = heat_eq_pricing.set_up_heat_equation()

        heat_eq_solution_disc = heat_eq_pricing.calc_heat_eq_solution(spatial_grid)
        return heat_eq_solution_disc[-1, :]

    def set_up_heat_equation(self) -> npt.NDArray[np.float64]:
        """Sets up discrete time and spatial grids for the heat equation."""
        spatial_grid = np.linspace(
            math.log(self.config.min_s),
            math.log(self.config.max_s),
            self.config.n_step_x + 1,
            dtype=np.float64,
        )
        return spatial_grid

    def calc_heat_eq_solution(
        self,
        spatial_grid: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Solves the heat equation using finite difference method."""
        initial_cond = self.initial_condition_calc(spatial_grid)
        banded_mat = self.banded_mat_imp_calc()
        heat_eq_solution = [initial_cond]
        for _ in trange(
            self.config.n_step_t,
            desc=f"Heat equation simulation K: {self.config.strike}: ",
            position=1,
            leave=True,
        ):
            heat_eq_solution.append(solve_banded((1, 1), banded_mat, heat_eq_solution[-1]))
        return np.array(heat_eq_solution)

    def initial_condition_calc(
        self,
        spatial_grid: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Calculates the initial condition for the heat equation."""
        return np.array(spatial_grid > math.log(self.config.strike), dtype=np.float64) * np.exp(
            -self.config.alpha * spatial_grid
        )

    def banded_mat_imp_calc(self) -> npt.NDArray[np.float64]:
        """Calculates the banded matrix to solve the heat equation using implicit scheme."""
        coeff = self.config.sigma**2 * self.config.step_size_t / self.config.step_size_x**2
        banded_mat = np.zeros((3, self.config.n_step_x + 1), dtype=np.float64)

        upper_diag = -coeff * np.ones(self.config.n_step_x)
        main_diag = (1 + 2 * coeff) * np.ones(self.config.n_step_x + 1)
        lower_diag = -coeff * np.ones(self.config.n_step_x)

        banded_mat[0, 1:] = upper_diag
        banded_mat[1, :] = main_diag
        banded_mat[2, :-1] = lower_diag
        return banded_mat


model_settings = {
    "n_step_x": 1000,
    "n_step_t": 2000,
    "t_end": 1.0,
    "s_0": 1.0,
    "min_s": 0.2,
    "max_s": 5.0,
    "strike": 1.0,
    "sigma": 0.2,
    "risk_free_rate": 0.01,
    "drift": 0.01,
}

prices = HeatEquationPricing.calculate_binary_option_price(HeatEquationSettings(**model_settings))
alma = 1
