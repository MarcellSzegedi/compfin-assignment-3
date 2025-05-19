"""Plot binary option price sensitivity to volatility."""

import math
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import typer
from tqdm import tqdm

from compfin_assignment_3.option_pricing.heat_equation_sim.pricing import HeatEquationPricing
from compfin_assignment_3.option_pricing.heat_equation_sim.settings import HeatEquationSettings

app = typer.Typer()


@app.command(name="volatility")
def plot_price_sensitivity_to_volatility(
    boundary_cond: Annotated[
        str, typer.Option("--boundary-cond", help="Boundary condition to use.")
    ] = "dirichlet",
) -> None:
    """Plot binary option price sensitivity to volatility as 4 subplots with 3D surfaces."""
    model_settings = {
        "n_step_x": 3000,
        "n_step_t": 10000,
        "t_end": 1.0,
        "s_0": 1.0,
        "min_s": 0.05,
        "max_s": 5.0,
        "sigma": 0.1,
        "risk_free_rate": 0.01,
        "drift": 0.01,
    }
    volatilities = [0.1, 0.4, 0.8, 1.5]  # Increasing volatility values
    strike_prices = np.linspace(0.2, 1.8, 200)
    starting_asset_prices = np.exp(
        np.linspace(
            math.log(model_settings["min_s"]),
            math.log(model_settings["max_s"]),
            model_settings["n_step_x"] + 1,
        )
    )
    X, Y = np.meshgrid(starting_asset_prices, strike_prices)

    fig = plt.figure(figsize=(20, 7))
    for i, sigma in enumerate(volatilities, start=1):
        prices = []
        for strike in tqdm(strike_prices, desc=f"Volatility={sigma}", position=0, leave=True):
            curr_model_settings = HeatEquationSettings(
                **{**model_settings, "strike": strike, "sigma": sigma}
            )
            price = HeatEquationPricing.calculate_binary_option_price(
                curr_model_settings, boundary_cond, "implicit"
            )
            prices.append(price)
        prices = np.array(prices)

        ax = fig.add_subplot(1, 4, i, projection="3d")
        ax.plot_surface(X, Y, prices, cmap="coolwarm")
        ax.set_title(f"Volatility = {sigma}")
        ax.set_xlabel("Price")
        ax.set_ylabel("Strike Price")
        ax.view_init(azim=120, elev=30)

    plt.tight_layout()
    plt.savefig("figures/volatility_sensitivity.png", dpi=600)
    plt.show()
