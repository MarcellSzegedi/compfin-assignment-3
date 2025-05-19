"""Sensitivity test for risk-free rate."""

import math
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import typer
from tqdm import tqdm

from compfin_assignment_3.option_pricing.heat_equation_sim.pricing import HeatEquationPricing
from compfin_assignment_3.option_pricing.heat_equation_sim.settings import HeatEquationSettings

app = typer.Typer()


@app.command(name="riskfree")
def plot_price_sensitivity_to_risk_free_rate(
    boundary_cond: Annotated[
        str, typer.Option("--boundary-cond", help="Boundary condition to use.")
    ] = "dirichlet",
) -> None:
    """Plot binary option price sensitivity to risk-free rate as 4 subplots with 3D surfaces."""
    model_settings = {
        "n_step_x": 3000,
        "n_step_t": 10000,
        "t_end": 1.0,
        "s_0": 1.0,
        "min_s": 0.05,
        "max_s": 5.0,
        "sigma": 0.2,
        "risk_free_rate": 0.01,  # base rate, will be overridden
        "drift": 0.01,
    }
    risk_free_rates = [0.0, 0.04, 0.1, 0.2]  # Different risk-free rates to test
    strike_prices = np.linspace(0.2, 1.8, 200)
    starting_asset_prices = np.exp(
        np.linspace(
            math.log(model_settings["min_s"]),
            math.log(model_settings["max_s"]),
            model_settings["n_step_x"] + 1,
        )
    )
    X, Y = np.meshgrid(starting_asset_prices, strike_prices)

    fig = plt.figure(figsize=(20, 6))
    for i, rfr in enumerate(risk_free_rates, start=1):
        prices = []
        for strike in tqdm(strike_prices, desc=f"risk_free_rate={rfr}", position=0, leave=True):
            curr_model_settings = HeatEquationSettings(
                **{**model_settings, "strike": strike, "risk_free_rate": rfr}
            )
            price = HeatEquationPricing.calculate_binary_option_price(
                curr_model_settings, boundary_cond, "implicit"
            )
            prices.append(price)
        prices = np.array(prices)

        ax = fig.add_subplot(1, 4, i, projection="3d")
        ax.plot_surface(X, Y, prices, cmap="coolwarm")
        ax.set_title(f"Risk-Free Rate = {rfr}")
        ax.set_xlabel("Price")
        ax.set_ylabel("Strike Price")
        ax.view_init(azim=120, elev=30)

    plt.tight_layout()
    plt.savefig("figures/riskfree_sensitivity.png", dpi=600)
    plt.show()
