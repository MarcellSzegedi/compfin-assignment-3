"""
Course: Computational Finance
Names: Marcell Szegedi; Tika van Bennekum; Michael MacFarlane Glasow
Student IDs: 15722635; 13392425; 12317217

File description:
    This file contains two pricing engines.
    A Monte carlo engine and a semi-closed Heston formula engine.
    we compare their all-price surfaces and their mplied-volatility surfaces.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.stats import norm
from scipy.optimize import brentq


def parameters():
    """
    Chosen parameteres for Heston model.
    """

    kappa = 2.0  # mean reversion speed
    theta = 0.04  # long-term variance
    sigma = 0.3  # volatility of volatility
    rho = -0.7  # correlation price and volatity
    v0 = 0.04  # initial variance

    return (kappa, theta, sigma, rho, v0)


def strike_maturity(nr_timesteps):
    """
    Chosen strike and maturity range.
    """

    strikes = np.arange(50, 150, nr_timesteps)
    maturities = np.arange(0, 3, nr_timesteps)  # In years

    return (strikes, maturities)


def brownian_motion(t, rho):
    """
    Calculates current values of W and B.
    """
    # Two random variables from standard normal distribution
    z1 = np.random.randn()
    z2 = np.random.randn()

    dW = z1 * np.sqrt(t)
    dB = rho * z1 * np.sqrt(t) + np.sqrt(1 - rho**2) * z2 * np.sqrt(t)

    return (dW, dB)


def euler_maruyama(s0, strike, r, total_time, nr_timesteps):
    """
    Simulates one path under the Heston model using Euler-Maruyama.
    Returns the payoff.
    """
    dt = total_time / nr_timesteps
    (k, theta, sigma, rho, v0) = parameters()

    s = s0
    v = v0
    for _ in range(1, nr_timesteps + 1):
        dW, dB = brownian_motion(dt, rho)

        # Update variance
        new_v = v + k * (theta - v) * dt + sigma * np.sqrt(max(v, 0)) * dW
        v = max(0, new_v)

        # Safe square root to avoid overflow
        v_safe = max(v, 1e-8)

        # Update asset with guard to avoid overflow
        exponent = (r - 0.5 * v_safe) * dt + np.sqrt(v_safe) * dB
        if exponent > 700:
            exponent = 700
        s = s * np.exp(exponent)

    payoff = max(s - strike, 0)
    return payoff


def simulations_monte_carlo(s0, strike, r, maturity, nr_timesteps, nr_runs):
    """
    Simulates multiple paths to find the average payoff for a certain strike level and maturity.
    """
    avg_payoff = 0
    for _ in range(nr_runs):
        payoff = euler_maruyama(s0, strike, r, maturity, nr_timesteps)
        avg_payoff += payoff
    avg_payoff /= nr_runs
    price = np.exp(-r * maturity) * avg_payoff

    return price


def heston_characteristic_function(u, maturity, S0, r):
    """
    Stable version of the Heston characteristic function.
    """
    kappa, theta, sigma, rho, v0 = parameters()
    i = complex(0, 1)
    x = np.log(S0)

    xi = kappa - sigma * rho * u * i
    d = np.sqrt((xi) ** 2 + sigma**2 * (u**2 + i * u))
    g2 = (xi - d) / (xi + d)

    C = (kappa * theta / sigma**2) * (
        (xi - d) * maturity - 2 * np.log((1 - g2 * np.exp(-d * maturity)) / (1 - g2))
    )
    D = ((xi - d) / sigma**2) * (
        (1 - np.exp(-d * maturity)) / (1 - g2 * np.exp(-d * maturity))
    )

    return np.exp(i * u * (x + r * maturity) + C + D * v0)


def P_integral(j, S0, strike, r, maturity):
    """
    Calculating the P integral formula.
    """
    i = complex(0, 1)
    F = S0 * np.exp(r * maturity)

    def integrand(u):
        if j == 1: # P1
            phi = heston_characteristic_function(u - i, maturity, S0, r)
            denom = i * u * F
        else: # P2
            phi = heston_characteristic_function(u, maturity, S0, r)
            denom = i * u

        return np.real(np.exp(-i * u * np.log(strike)) * phi / denom)

    integral, _ = quad(integrand, 1e-5, 100)
    return 0.5 + (1 / np.pi) * integral


def heston_call_price(S0, strike, r, maturity):
    """
    Full Heston semi-closed formula for European call.
    """
    P1 = P_integral(1, S0, strike, r, maturity)
    P2 = P_integral(2, S0, strike, r, maturity)

    return S0 * P1 - np.exp(-r * maturity) * strike * P2


def singular_comparison():
    """Function to test if the pricing engines are working as we expect."""
    s0 = 100
    strike = 110
    r = 0.05
    maturity = 1
    nr_timesteps = 200
    nr_runs = 1000

    price = simulations_monte_carlo(s0, strike, r, maturity, nr_timesteps, nr_runs)
    print(f"Monte Carlo Option Price: {price:.4f}")

    price = heston_call_price(s0, strike, r, maturity)
    print(f"Heston Analytic Price: {price:.4f}")


def compute_and_save_surface_data():
    S0 = 100
    r = 0.05
    n_paths = 10000
    n_steps = 200

    strikes = np.linspace(80, 120, 10)
    maturities = np.linspace(0.1, 2.0, 10)
    strike_grid, maturity_grid = np.meshgrid(strikes, maturities)

    analytic_prices = np.zeros_like(strike_grid)
    monte_carlo_prices = np.zeros_like(strike_grid)

    for i in range(strike_grid.shape[0]):
        for j in range(strike_grid.shape[1]):
            K = strike_grid[i, j]
            T = maturity_grid[i, j]
            analytic_prices[i, j] = heston_call_price(S0, K, r, T)
            monte_carlo_prices[i, j] = simulations_monte_carlo(S0, K, r, T, n_steps, n_paths)

    np.savez("surface_data.npz",
             strike_grid=strike_grid,
             maturity_grid=maturity_grid,
             analytic_prices=analytic_prices,
             monte_carlo_prices=monte_carlo_prices)



def load_surface_data():
    data = np.load("surface_data.npz")
    return (data["strike_grid"], data["maturity_grid"],
            data["analytic_prices"], data["monte_carlo_prices"])


def plot_call_price_surface():
    strike_grid, maturity_grid, analytic_prices, monte_carlo_prices = load_surface_data()

    fig = plt.figure(figsize=(16, 6))

    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.plot_surface(strike_grid, maturity_grid, analytic_prices, cmap='viridis')
    ax1.set_title("Heston Analytic Call Price")
    ax1.set_xlabel("Strike")
    ax1.set_ylabel("Maturity")
    ax1.set_zlabel("Call Price")

    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.plot_surface(strike_grid, maturity_grid, monte_carlo_prices, cmap='plasma')
    ax2.set_title("Monte Carlo Call Price")
    ax2.set_xlabel("Strike")
    ax2.set_ylabel("Maturity")
    ax2.set_zlabel("Call Price")

    plt.savefig("call_price_surface.png", dpi=400, bbox_inches="tight")


def black_scholes_call_price(S0, K, r, T, sigma):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def implied_volatility(price, S0, K, r, T):
    # Define a function that we want to find the root of
    def bs_price_error(sigma):
        bs_price = black_scholes_call_price(S0, K, r, T, sigma)
        return bs_price - price

    # Try to find the implied volatility between 0.0001 and 5.0
    try:
        implied_vol = brentq(bs_price_error, 1e-4, 5.0)
        return implied_vol
    except:
        return float('nan')  # if it can't find a solution

def plot_implied_volatility_surface():
    S0 = 100
    r = 0.05

    strike_grid, maturity_grid, analytic_prices, monte_carlo_prices = load_surface_data()
    analytic_iv = np.zeros_like(analytic_prices)
    monte_carlo_iv = np.zeros_like(monte_carlo_prices)

    for i in range(strike_grid.shape[0]):
        for j in range(strike_grid.shape[1]):
            K = strike_grid[i, j]
            T = maturity_grid[i, j]
            analytic_iv[i, j] = implied_volatility(analytic_prices[i, j], S0, K, r, T)
            monte_carlo_iv[i, j] = implied_volatility(monte_carlo_prices[i, j], S0, K, r, T)

    fig = plt.figure(figsize=(16, 6))

    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.plot_surface(strike_grid, maturity_grid, analytic_iv, cmap='viridis')
    ax1.set_title("Analytic Implied Volatility")
    ax1.set_xlabel("Strike")
    ax1.set_ylabel("Maturity")
    ax1.set_zlabel("Implied Volatility")

    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.plot_surface(strike_grid, maturity_grid, monte_carlo_iv, cmap='plasma')
    ax2.set_title("Monte Carlo Implied Volatility")
    ax2.set_xlabel("Strike")
    ax2.set_ylabel("Maturity")
    ax2.set_zlabel("Implied Volatility")

    plt.savefig("implied_volatility_surface.png", dpi=400, bbox_inches="tight")


if __name__ == "__main__":
    compute_and_save_surface_data()
    plot_call_price_surface()
    plot_implied_volatility_surface()
