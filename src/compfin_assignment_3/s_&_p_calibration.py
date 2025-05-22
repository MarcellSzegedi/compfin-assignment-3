import numpy as np
import pathlib as path
from scipy.integrate import quad
from numpy import real, pi, log
from scipy.optimize import minimize
from pricing import implied_volatility
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

script_dir = path.Path(__file__).resolve().parent
data_path_raw = script_dir / "data_s_p" / "raw_ivol_surfaces.npy"
data_path_interp = script_dir / "data_s_p" / "interp_ivol_surfaces.npy"

raw = np.load(data_path_raw, allow_pickle=True).item()
interp = np.load(data_path_interp, allow_pickle=True).item()

date ="2023 11 07"
raw_vols = raw[date]["vols"] #15 × N
interp_vols = interp[date]["vols"] #N × 100
strikes = interp[date]["strikes"] #N
maturities = interp[date]["tenors"] #15


strikes = strikes[::5]     # Take every 5th strike (reduce to ~20)
maturities = maturities[::3]  # Reduce to ~5
interp_vols = interp_vols[::3, ::5]  # Slice accordingly

#refactor the heston call price to take in the parameters



def plot_iv_surface_comparison(strikes, maturities, market_iv, model_iv, date_str):
    strike_grid, maturity_grid = np.meshgrid(strikes, maturities)

    fig = plt.figure(figsize=(14, 6))

    # Market surface
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.plot_surface(strike_grid, maturity_grid, market_iv, cmap='viridis')
    ax1.set_title(f"Market IV - {date_str}")
    ax1.set_xlabel("Strike")
    ax1.set_ylabel("Maturity")
    ax1.set_zlabel("Implied Volatility")

    # Model surface
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.plot_surface(strike_grid, maturity_grid, model_iv, cmap='plasma')
    ax2.set_title(f"Fitted IV - {date_str}")
    ax2.set_xlabel("Strike")
    ax2.set_ylabel("Maturity")
    ax2.set_zlabel("Implied Volatility")

    plt.tight_layout()
    plt.savefig(f"iv_surface_comparison_{date_str.replace(' ', '_')}.png", dpi=300)
    plt.show()

def heston_call_price(S0, K, r, maturity, v0, kappa, theta, sigma, rho):
    """
    Full Heston semi-closed formula for European call, with parameter inputs.
    """
    try:
        P1 = P_integral(1, S0, K, r, maturity, v0, kappa, theta, sigma, rho)
        P2 = P_integral(2, S0, K, r, maturity, v0, kappa, theta, sigma, rho)

        # Safety: Clamp P1, P2 to [0, 1]
        P1 = min(max(P1, 0.0), 1.0)
        P2 = min(max(P2, 0.0), 1.0)
        price = S0 * P1 - np.exp(-r * maturity) * K * P2

        # Safety: Clamp negative or invalid prices
        if not np.isfinite(price) or price <= 0:
            # print(f"[Invalid price] Computed price={price}, P1={P1}, P2={P2}, K={K}, T={maturity}")
            return 1e-6  # Small positive value as fallback
    
        return price
    except Exception as e:
        # print(f"[ERROR] Heston call price calculation failed for S0={S0}, K={K}, T={maturity} → {e}")
        return 1e-6  # Small positive value as fallback

#refactor the P_integral function to take in the parameters

def P_integral(j, S0, strike, r, maturity, v0, kappa, theta, sigma, rho):
    """
    Computes the integral for probability P1 or P2 in the Heston formula.
    """
    i = complex(0, 1)
    F = S0 * np.exp(r * maturity)

    def integrand(u):
        if j == 1:
            phi = heston_characteristic_function(u - i, maturity, S0, r, v0, kappa, theta, sigma, rho)
            denom = i * u * F
        else:
            phi = heston_characteristic_function(u, maturity, S0, r, v0, kappa, theta, sigma, rho)
            denom = i * u

        return real(np.exp(-i * u * np.log(strike)) * phi / denom)

    integral, _ = quad(integrand, 1e-5, 100)
    return 0.5 + (1 / pi) * integral

#refactor the heston_characteristic_function to take in the parameters

def heston_characteristic_function(u, maturity, S0, r, v0, kappa, theta, sigma, rho):
    """
    Computes the characteristic function φ(u) for the Heston model, with given parameters.
    """
    i = complex(0, 1)
    x = np.log(S0)
    EPS = 1e-8  # small number to avoid division by zero / log(0)

    xi = kappa - sigma * rho * u * i
    d = np.sqrt(xi**2 + sigma**2 * (u**2 + i * u))
    
     # Ensure d is not zero
    d_safe = d if abs(d) > EPS else EPS

    # g2 calculation with safe denominator
    g2_den = xi + d_safe
    g2 = (xi - d_safe) / (g2_den if abs(g2_den) > EPS else EPS)

    if not np.isfinite(d) or abs(d) < 1e-8:
        raise ValueError("Unstable 'd' in characteristic function")

    if not np.isfinite(g2):
        raise ValueError("Unstable 'g2' in characteristic function")
    
    # Prevent log or division by zero by clamping denominators
    exp_term = np.exp(-d_safe * maturity)
    denom1 = 1 - g2 * exp_term
    denom2 = 1 - g2

    # Clamp log argument to avoid log(0)
    log_arg = denom1 / (denom2 if abs(denom2) > EPS else EPS)
    log_arg = log_arg if abs(log_arg) > EPS else EPS

    C = (kappa * theta / sigma**2) * ((xi - d_safe) * maturity - 2 * np.log(log_arg))

    D_numerator = 1 - exp_term
    D_denominator = 1 - g2 * exp_term
    D_denominator = D_denominator if abs(D_denominator) > EPS else EPS

    D = ((xi - d_safe) / sigma**2) * (D_numerator / D_denominator)

    return np.exp(i * u * (x + r * maturity) + C + D * v0)


iteration_counter = 0  # Global variable to track optimizer iterations

def loss_function(params, S0, r, strikes, maturities, interp_vols):
    """
    Loss function for Heston model calibration:
    Computes the sum of squared errors between model and market implied vols.
    Includes robust error handling and logging.
    """
    v0, kappa, theta, sigma, rho = params
    loss = 0.0
    failure_count = 0
    total_count = 0

    try:
        for i in range(len(strikes)):
            for j in range(len(maturities)):
                K = strikes[i]
                T = maturities[j]
                market_iv = interp_vols[j, i]  # Assumes shape (len(maturities), len(strikes))

                # Skip invalid market data
                if not np.isfinite(market_iv):
                    continue

                # Compute Heston model price and implied vol
                try:
                    price = heston_call_price(S0, K, r, T, v0, kappa, theta, sigma, rho)
                    model_iv = implied_volatility(price, S0, K, T, r)

                    if np.isnan(model_iv) or not np.isfinite(model_iv):
                        failure_count += 1
                        continue  # Skip, don't penalize yet

                    # Accumulate squared error
                    loss += (model_iv - market_iv) ** 2
                    total_count += 1

                except Exception as e:
                    failure_count += 1
                    continue

    except Exception as e:
        print(f"[FATAL] Loss function failed for params {params} → {e}")
        return 1e6  # Fatal fallback penalty
    
    print(f"[Iteration {iteration_counter:03}] Loss={loss:.4f} | Valid={total_count} | Failed={failure_count}")
    # If everything failed, return penalty to avoid "success" on garbage
    if total_count == 0 or failure_count > 0.8 * (len(strikes) * len(maturities)):
        return 1e6

    return loss / total_count  # normalize by number of valid points


# Define the initial guess for the parameters
initial_params = [0.04, 1.0, 0.04, 0.3, -0.7]
bounds = [(0.0005, 0.3), (0.5, 5.0), (0.01, 0.2), (0.05, 1.0), (-0.95, -0.1)]

S0 = 4300
r = 0.05

result = minimize(
    loss_function,
    initial_params,
    args=(S0, r, strikes, maturities, interp_vols),
    bounds=bounds,
)

print(result)
 