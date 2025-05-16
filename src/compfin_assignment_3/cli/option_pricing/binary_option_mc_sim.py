"""Plot the price of a binary option using Monte Carlo simulation compared to analytic solution."""

import math
from collections import defaultdict
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import typer
from tqdm import tqdm

from compfin_assignment_3.option_pricing.binary_option import BinaryOption
from compfin_assignment_3.option_pricing.mc_sim_settings import BSModelSettings

app = typer.Typer()


@app.command(name="binary-mc-sim")
def main(
    n_trajectories: Annotated[
        int, typer.Option("--n-traj", min=1, help="Number of trajectories to simulate.")
    ] = 1000,
):
    """Plot the price of a binary option using Monte Carlo method compared to analytic solution."""
    model_settings = {
        "n_trajectories": n_trajectories,
        "s_0": 1,
        "sigma": 0.2,
        "t_end": 1,
        "drift": 0.02,
        "num_steps": 10000,
        "risk_free_rate": 0.02,
        "alpha": 0.05,
    }
    strike_prices = np.linspace(0.2, 1.8, 200)
    prices, conf_us, conf_ls = defaultdict(list), defaultdict(list), defaultdict(list)

    for strike in tqdm(strike_prices, desc="Strike price:"):
        curr_model_settings = BSModelSettings(**{**model_settings, "strike": strike})
        stoch_increments = np.random.normal(
            loc=0,
            scale=math.sqrt(curr_model_settings.step_size),
            size=(n_trajectories, curr_model_settings.num_steps),
        )

        price, conf_u, conf_l = BinaryOption.simulate_binary_option(
            curr_model_settings, "euler", stoch_increments
        )
        prices["euler"].append(price)
        conf_us["euler"].append(conf_u)
        conf_ls["euler"].append(conf_l)

        price, conf_u, conf_l = BinaryOption.simulate_binary_option(
            curr_model_settings, "milstein"
        )
        prices["milstein"].append(price)
        conf_us["milstein"].append(conf_u)
        conf_ls["milstein"].append(conf_l)

    true_prices = [
        BinaryOption.compute_binary_option_price_analytical(
            BSModelSettings(**{**model_settings, "strike": strike})
        )
        for strike in strike_prices
    ]

    plt.figure(figsize=(10, 5))
    plt.plot(
        strike_prices,
        true_prices,
        label="Analytic solution",
        linewidth=3,
        color="black",
        linestyle="--",
    )
    plt.plot(strike_prices, prices["euler"], label="Euler scheme", linewidth=1, color="red")
    plt.plot(strike_prices, prices["milstein"], label="Milstein scheme", linewidth=1, color="blue")

    plt.fill_between(strike_prices, conf_us["euler"], conf_ls["euler"], alpha=0.2, color="red")
    plt.fill_between(
        strike_prices, conf_us["milstein"], conf_ls["milstein"], alpha=0.2, color="blue"
    )
    plt.plot(strike_prices, conf_us["euler"], color="red", linestyle="--", linewidth=0.5)
    plt.plot(strike_prices, conf_ls["euler"], color="red", linestyle="--", linewidth=0.5)
    plt.plot(strike_prices, conf_us["milstein"], color="blue", linestyle="--", linewidth=0.5)
    plt.plot(strike_prices, conf_ls["milstein"], color="blue", linestyle="--", linewidth=0.5)

    plt.legend()
    plt.xlabel("Strike price")
    plt.ylabel("Price")
    plt.title("Binary option price simulation")
    plt.tight_layout()

    plt.savefig("figures/binary_option_mc_sim_1000.png", dpi=600)
    plt.show()
