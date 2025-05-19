"""Plots the delta as a function of price and time to maturity."""

import matplotlib.pyplot as plt
import numpy as np
import typer

from compfin_assignment_3.option_pricing.binary_option_delta import (
    BinaryDeltaSettings,
    BinaryOptionDelta,
)

app = typer.Typer()


@app.command(name="delta-calc")
def main():
    """Plots the delta as a function of price and time to maturity."""
    model_settings = {
        "t_end": 1.0,
        "s_min": 0.05,
        "s_max": 5.0,
        "n_step": 5000,
        "sigma": 0.2,
        "risk_free_rate": 0.01,
        "strike": 1.0,
    }

    prices, taus, deltas = BinaryOptionDelta.delta_calculation(
        BinaryDeltaSettings(**model_settings)
    )

    X, Y = np.meshgrid(prices, taus, indexing="ij")

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X, Y, deltas, cmap="coolwarm")
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    ax.set_title("Binary Call Option Delta")
    ax.set_xlabel("Price")
    ax.set_ylabel("Time to Maturity")
    ax.set_zlabel("Delta")

    ax.view_init(azim=120, elev=30)
    plt.tight_layout()

    plt.savefig("figures/binary_option_delta.png", dpi=600)
    plt.show()
