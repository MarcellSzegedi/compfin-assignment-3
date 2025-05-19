from mc_method import up_and_out_call_montecarlo
from barrier_option import up_and_out_call_analytical
import numpy as np
import matplotlib.pyplot as plt
# Parameters
S0 = 100     # spot price
K = 90       # strike price
B = 120      # barrier
T = 1        # maturity in years
r = 0.05     # risk-free rate
sigma = 0.2  # volatility

N_paths_list = [1_000, 5_000, 10_000, 50_000, 100_000, 200_000, 500_000]
beta1 = 0.5826
m = N_steps = 252
H_eff = B * np.exp(beta1 * sigma * np.sqrt(T/m))
# Analytical price
price_analytical = up_and_out_call_analytical(S0, K, H_eff, T, r, sigma)
print(f"Analytical Price: {price_analytical:.6f}")
mc_prices = []
errors = []

# Monte Carlo price convergence analysis
for N in N_paths_list:
    price_mc = up_and_out_call_montecarlo(S0, K, B, T, r, sigma, N_paths=N, N_steps=m)
    print(f"Monte Carlo Price with {N} paths: {price_mc:.6f}")
    error = abs(price_mc - price_analytical)
    errors.append(error)
    print(f"Absolute Error: {error:.6f}")



# Output comparison
# print(f"Analytical Price   : {price_analytical:.6f}")
# print(f"Monte Carlo Price  : {price_mc:.6f}")
# print(f"Absolute Difference: {abs(price_mc - price_analytical):.6f}")
# print(f"Relative Difference: {abs(price_mc - price_analytical) / price_analytical:.2%}")


plt.figure(figsize=(10, 6))
plt.plot(N_paths_list, errors, marker='o')
plt.axhline(0, color='gray', linestyle='--')
plt.title('Convergence of Monte Carlo Estimate')
plt.xlabel('Number of Monte Carlo Paths')
plt.ylabel('Absolute Error vs Analytical')
plt.xscale('log')  # Because convergence is O(1/√N)
plt.grid(True)
plt.show()