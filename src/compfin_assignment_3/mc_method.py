import numpy as np

def up_and_out_call_montecarlo(S0, K, B, T, r, sigma, N_paths=10000, N_steps=252):
    """
    Computes the price of an up-and-out barrier call option using Monte Carlo simulation.
    
    Parameters:
    S0     : Initial stock price
    K      : Strike price
    B      : Barrier level (B > S0)
    T      : Time to maturity (in years)'
    r      : Risk-free interest rate
    sigma  : Volatility
    N_paths: Number of Monte Carlo paths (default is 10,000)
    N_steps: Number of time steps in each path (default is 252)
    """
    dt = T / N_steps
    discount_factor = np.exp(-r * T)
    # Initialize the array to hold the paths
    S_paths = np.zeros((N_paths, N_steps + 1))
    S_paths[:, 0] = S0

    # Generate paths using geometric Brownian motion
    np.random.seed(42)  # For reproducibility
    Z = np.random.normal(size=(N_paths, N_steps))
    for t in range(1, N_steps + 1):
        S_paths[:, t] = S_paths[:, t - 1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[:, t - 1])

    # Check if the barrier is breached
    knocked_out = np.zeros(N_paths, dtype=bool)
    log_B = np.log(B)


    for t in range(N_steps):
        S_t = S_paths[:, t]
        S_tp1 = S_paths[:, t+1]

        log_S_t = np.log(S_t)
        log_S_tp1 = np.log(S_tp1)

        # Step 1: Only apply bridge where both ends are below barrier
        mask = (log_S_t < log_B) & (log_S_tp1 < log_B)

        # Step 2: Compute crossing probability for valid segments
        log_S_t_masked = log_S_t[mask]
        log_S_tp1_masked = log_S_tp1[mask]

        exponent = -2 * (log_B - log_S_t_masked) * (log_B - log_S_tp1_masked) / (sigma**2 * dt)
        
        # Ensure numerical stability (prevent overflow or underflow)
        exponent = np.clip(exponent, -700, 0)
        p_cross = np.exp(exponent)

        # Step 3: Simulate whether a crossing occurred
        u = np.random.uniform(size=np.sum(mask))
        crossed = np.zeros(N_paths, dtype=bool)
        crossed_indices = np.where(mask)[0]
        crossed[crossed_indices] = u < p_cross

        # Step 4: Update knockout flags
        knocked_out |= crossed
        #print(f"Total KO ratio: {np.mean(knocked_out):.2%}")


    # Calculate the payoff for paths that did not breach the barrier
    ST = S_paths[:, -1]
    payoffs = np.where(knocked_out, 0, np.maximum(ST - K, 0))
    # Calculate the price
    price = discount_factor * np.mean(payoffs)

    return price