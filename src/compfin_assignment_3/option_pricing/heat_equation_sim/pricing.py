"""Pricing binary option using heat equation simulation."""

import math

import numpy as np
import numpy.typing as npt
from scipy.linalg import solve_banded
from scipy.sparse import dia_matrix, diags

from compfin_assignment_3.option_pricing.heat_equation_sim.settings import HeatEquationSettings


class HeatEquationPricing:
    """Pricing binary option using heat equation simulation."""

    def __init__(self, config: HeatEquationSettings, boundary_cond: str) -> None:
        """Initialize HeatEquationPricing object."""
        self.config = config
        self.boundary_cond = boundary_cond

    @classmethod
    def calculate_binary_option_price(
        cls,
        config: HeatEquationSettings,
        boundary_cond: str = "dirichlet",
        numeric_scheme: str = "implicit",
    ) -> npt.NDArray[np.float64]:
        """Calculate the price of a binary option using heat equation simulation."""
        heat_eq_pricing = cls(config, boundary_cond)

        spatial_grid = heat_eq_pricing.set_up_heat_equation()
        num_solver = heat_eq_pricing._set_numeric_scheme_solver(numeric_scheme)

        heat_eq_solution_disc = num_solver(spatial_grid)

        transformed_prices = heat_eq_pricing.transform_heat_to_s(
            heat_eq_solution_disc[-1], spatial_grid, config.t_end
        )
        return transformed_prices

    def set_up_heat_equation(self) -> npt.NDArray[np.float64]:
        """Sets up discrete time and spatial grids for the heat equation."""
        spatial_grid = np.linspace(
            math.log(self.config.min_s),
            math.log(self.config.max_s),
            self.config.n_step_x + 1,
            dtype=np.float64,
        )
        return spatial_grid

    def calc_heat_eq_solution_imp(
        self,
        spatial_grid: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Solves the heat equation using finite difference method."""
        initial_cond = self.initial_condition_calc(spatial_grid)
        banded_mat, coeff = self.banded_mat_imp_scheme_calc()
        heat_eq_solution = [initial_cond]
        for i in range(self.config.n_step_t):
            goal_vec = heat_eq_solution[-1].copy()
            match self.boundary_cond:
                case "dirichlet":
                    goal_vec[0] = 0
                    goal_vec[-1] = self.transform_s_to_heat(
                        1, spatial_grid[-1], i * self.config.step_size_t
                    )
                    next_solution = solve_banded((1, 1), banded_mat, goal_vec)
                    next_solution[0] = 0
                    next_solution[-1] = self.transform_s_to_heat(
                        1, spatial_grid[-1], (i + 1) * self.config.step_size_t
                    )
                    heat_eq_solution.append(next_solution)
                case "neumann":
                    goal_vec[-1] += 2 * coeff * self.config.step_size_x
                    heat_eq_solution.append(solve_banded((1, 1), banded_mat, goal_vec))
                case _:
                    raise ValueError("Invalid boundary condition.")

        return np.array(heat_eq_solution)

    def calc_heat_eq_solution_crank_nicolson(
        self,
        spatial_grid: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Solves the heat equation using Crank-Nicolson method."""
        initial_cond = self.initial_condition_calc(spatial_grid)
        banded_mat_1, banded_mat_2, _ = self.banded_mat_crank_nic_scheme_calc()
        heat_eq_solution = [initial_cond]
        for i in range(self.config.n_step_t):
            goal_vec = heat_eq_solution[-1].copy()
            match self.boundary_cond:
                case "dirichlet":
                    goal_vec[0] = 0
                    goal_vec[-1] = self.transform_s_to_heat(
                        1, spatial_grid[-1], i * self.config.step_size_t
                    )
                    goal_vec = banded_mat_2 @ goal_vec
                    next_solution = solve_banded((1, 1), banded_mat_1, goal_vec)
                    next_solution[0] = 0
                    next_solution[-1] = self.transform_s_to_heat(
                        1, spatial_grid[-1], (i + 1) * self.config.step_size_t
                    )
                    heat_eq_solution.append(next_solution)
                case "neumann":
                    raise ValueError("Neumann boundary condition not implemented yet.")
                case _:
                    raise ValueError("Invalid boundary condition.")

        return np.array(heat_eq_solution)

    def initial_condition_calc(
        self,
        spatial_grid: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Calculates the initial condition for the heat equation."""
        payoffs = np.array(spatial_grid > math.log(self.config.strike), dtype=np.float64)
        initial_cond_heat = self.transform_s_to_heat(payoffs, spatial_grid, 0)
        return initial_cond_heat

    def banded_mat_imp_scheme_calc(self) -> tuple[npt.NDArray[np.float64], float]:
        """Calculates the banded matrix to solve the heat equation using implicit scheme."""
        banded_mat, coeff = self._base_imp_trans_mat_calc()

        match self.boundary_cond:
            case "dirichlet":
                banded_mat[0, 1] = 0
                banded_mat[1, 0] = 1
                banded_mat[1, -1] = 1
                banded_mat[2, -2] = 0
            case "neumann":
                banded_mat[0, 1] = -2 * coeff
                banded_mat[2, -2] = -2 * coeff
            case _:
                raise ValueError("Invalid boundary condition.")

        return banded_mat, coeff

    def banded_mat_crank_nic_scheme_calc(
        self,
    ) -> tuple[npt.NDArray[np.float64], dia_matrix, float]:
        """Calculates the banded matrices to solve the heat eq using Crank-Nicolson scheme."""
        banded_mat_1, banded_mat_2, coeff = self._base_crank_nic_trans_mat_calc()

        match self.boundary_cond:
            case "dirichlet":
                banded_mat_1[0, 1] = 0
                banded_mat_1[1, 0] = 1
                banded_mat_1[1, -1] = 1
                banded_mat_1[2, -2] = 0

                banded_mat_2[0, :] = 0
                banded_mat_2[0, 0] = 1
                banded_mat_2[-1, :] = 0
                banded_mat_2[-1, -1] = 1
            case "neumann":
                raise ValueError("Neumann boundary condition not implemented yet.")

        return banded_mat_1, dia_matrix(banded_mat_2), coeff

    def transform_s_to_heat(
        self,
        price: float | npt.NDArray[np.float64],
        spatial_grid: float | npt.NDArray[np.float64],
        tau: float,
    ) -> float | npt.NDArray[np.float64]:
        """Transforms the normal var space of the option to the heat equation var space."""
        return price * np.exp(
            -self.config.alpha * spatial_grid
            + (self.config.risk_free_rate - self.config.beta) * tau
        )

    def transform_heat_to_s(
        self,
        heat_eq_solution: npt.NDArray[np.float64],
        spatial_grid: npt.NDArray[np.float64],
        tau: float,
    ) -> npt.NDArray[np.float64]:
        """Transforms the heat equation solution to the price of the option."""
        return heat_eq_solution * np.exp(
            self.config.alpha * spatial_grid
            + (self.config.beta - self.config.risk_free_rate) * tau
        )

    def _set_numeric_scheme_solver(self, numeric_scheme: str) -> callable:
        """Sets up the numeric scheme solver."""
        match numeric_scheme:
            case "implicit":
                return self.calc_heat_eq_solution_imp
            case "crank-nicolson":
                return self.calc_heat_eq_solution_crank_nicolson
            case _:
                raise ValueError("Invalid numeric scheme.")

    def _base_imp_trans_mat_calc(self) -> tuple[npt.NDArray[np.float64], float]:
        """Calculates the base matrix for the implicit scheme."""
        coeff = self.config.sigma**2 / 2 * self.config.step_size_t / self.config.step_size_x**2
        banded_mat = np.zeros((3, self.config.n_step_x + 1), dtype=np.float64)

        upper_diag = -coeff * np.ones(self.config.n_step_x)
        main_diag = (1 + 2 * coeff) * np.ones(self.config.n_step_x + 1)
        lower_diag = -coeff * np.ones(self.config.n_step_x)

        banded_mat[0, 1:] = upper_diag
        banded_mat[1, :] = main_diag
        banded_mat[2, :-1] = lower_diag
        return banded_mat, coeff

    def _base_crank_nic_trans_mat_calc(
        self,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float]:
        """Calculates the base matrices for the Crank-Nicolson scheme."""
        coeff = self.config.sigma**2 / 2 * self.config.step_size_t / self.config.step_size_x**2
        banded_mat_1 = np.zeros((3, self.config.n_step_x + 1), dtype=np.float64)

        main_diag_1 = (1 + coeff) * np.ones(self.config.n_step_x + 1)
        subdiag_1 = -coeff / 2 * np.ones(self.config.n_step_x)

        banded_mat_1[0, 1:] = subdiag_1
        banded_mat_1[1, :] = main_diag_1
        banded_mat_1[2, :-1] = subdiag_1

        main_diag_2 = (1 - coeff) * np.ones(self.config.n_step_x + 1)
        subdiag_2 = coeff / 2 * np.ones(self.config.n_step_x)

        banded_mat_2 = diags(
            diagonals=[subdiag_2, main_diag_2, subdiag_2],
            offsets=[-1, 0, 1],
            shape=(self.config.n_step_x + 1, self.config.n_step_x + 1),
            dtype=np.float64,
        ).toarray()

        return banded_mat_1, banded_mat_2, coeff


model_settings = {
    "n_step_x": 1000,
    "n_step_t": 4000,
    "t_end": 1.0,
    "s_0": 1.0,
    "min_s": 0.2,
    "max_s": 5.0,
    "strike": 1.0,
    "sigma": 0.2,
    "risk_free_rate": 0.01,
    "drift": 0.01,
}

prices = HeatEquationPricing.calculate_binary_option_price(
    HeatEquationSettings(**model_settings), "dirichlet", "crank-nicolson"
)

alma = 1
# # Create dummy 2D array
#
# # Create X and Y coordinates
# x = np.arange(prices.shape[1])
# y = np.arange(prices.shape[0])
# X, Y = np.meshgrid(x, y)
#
# # Plot
# fig = plt.figure(figsize=(10, 7))
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(X, Y, prices, cmap='viridis')
# ax.set_title("3D Surface Plot")
# plt.show()
