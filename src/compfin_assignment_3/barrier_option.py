import numpy as np
from scipy.stats import norm

def up_and_out_call_analytical(S, K, B, T, r, sigma):
    """
    Computes the analytical price of an up-and-out barrier call option under Black-Scholes.
    
    Parameters:
    S     : Spot price
    K     : Strike price
    B     : Barrier level (B > S)
    T     : Time to maturity (in years)
    r     : Risk-free interest rate
    sigma : Volatility

    Returns:
    price : Option price
    """
    if S >= B:
        return 0.0  # Barrier is breached, option is worthless
    tau = T
    # x1 = np.log(S/K)
    # x2 = np.log(S/B)
    # x3 = np.log((B**2)/(S*K))

    # def delta functions
    def delta_plus(z):
        return (np.log(z) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))

    def delta_minus(z):
        return (np.log(z) + (r - 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    
    d1 = delta_plus(S/K)
    d2 = delta_plus(B**2/ (S*K))
    d3 = delta_plus(S/B)
    d4 = delta_plus(B/S)
    lambda_ = (r/sigma**2) - 0.5

    d5 = delta_minus(S/K)
    d6 = delta_minus(B**2/ (S*K))
    d7 = delta_minus(S/B)  
    d8 = delta_minus(B/S)

    term1 = S * norm.cdf(d1)
    
    term2 = S * norm.cdf(d3) # take term1 - term2 for first row of deriv
    term3 = S * (B/S)**(2*lambda_)
    term4 = term3 * norm.cdf(d2)
    term5 = term3 * norm.cdf(d4) # take term4 - term5 for second row of deriv

    term6 = K * np.exp(-r*tau) * norm.cdf(d5)
    term7 = K * np.exp(-r*tau) * norm.cdf(d7) # take term6 - term7 for third row of deriv

    term8 = K * np.exp(-r*tau) * (B/S)**(2*lambda_ - 2)
    term9 = term8 * norm.cdf(d6)
    term10 = term8 * norm.cdf(d8) # take term9 - term10 for fourth row of deriv
    
    price = (term1 - term2) \
      - (term4 - term5) \
      - (term6 - term7) \
      + (term9 - term10)
    return price




    