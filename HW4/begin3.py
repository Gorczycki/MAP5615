import scipy.stats as stats
import numpy as np
import random
from scipy.stats import norm

r = 0.035
sigma = 0.2
S0 = 100
K = 90
T = 1
n = 20
N = 1000
experiments = 50

def european_call_price(S0, K, T, r, sigma):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return np.exp(-r * T) * (S0 * norm.cdf(d1) - K * norm.cdf(d2))

def asian_geo_price(r, sigma, S0, K, T, n):
    h = T / n
    mu_bar = np.log(S0) + (r - 0.5 * sigma**2) * h * (n + 1) / 2
    sigma_bar_sq = sigma**2 * h * (n + 1) * (2 * n + 1) / (6 * n)
    d1 = (mu_bar - np.log(K) + sigma_bar_sq) / np.sqrt(sigma_bar_sq)
    d2 = (mu_bar - np.log(K)) / np.sqrt(sigma_bar_sq)
    return np.exp(-r * T) * (np.exp(mu_bar + 0.5 * sigma_bar_sq) * norm.cdf(d1) - K * norm.cdf(d2))

def arithmetic_asian_price_crude(r, sigma, S0, K, T, n, N):
    h = T / n
    dt = np.full((N, n), h)
    Z = np.random.normal(size=(N, n))
    S = np.full((N, n), S0)
    for j in range(1, n):
        S[:, j] = S[:, j - 1] * np.exp((r - 0.5 * sigma**2) * h + sigma * np.sqrt(h) * Z[:, j])
    avg_price = np.mean(S, axis=1)
    payoff = np.exp(-r * T) * np.maximum(avg_price - K, 0)
    return np.mean(payoff)

def arithmetic_asian_price_control(r, sigma, S0, K, T, n, N):
    h = T / n
    geo_price = asian_geo_price(r, sigma, S0, K, T, n)
    Z = np.random.normal(size=(N, n))
    S = np.full((N, n), S0)

    for j in range(1, n):
        S[:, j] = S[:, j - 1] * np.exp((r - 0.5 * sigma**2) * h + sigma * np.sqrt(h) * Z[:, j])

    arith_mean = np.mean(S, axis=1)
    geo_mean = np.exp(np.mean(np.log(S), axis=1))

    Y = np.exp(-r * T) * np.maximum(arith_mean - K, 0)
    C = np.exp(-r * T) * np.maximum(geo_mean - K, 0)

    beta_hat = np.cov(Y, C)[0, 1] / np.var(C)
    control_estimate = Y - beta_hat * (C - geo_price)
    return np.mean(control_estimate), np.var(control_estimate, ddof=1)

crude_estimates = []
control_estimates = []

for _ in range(experiments):
    crude = arithmetic_asian_price_crude(r, sigma, S0, K, T, n, N)
    control, _ = arithmetic_asian_price_control(r, sigma, S0, K, T, n, N)
    crude_estimates.append(crude)
    control_estimates.append(control)

var_Y = np.var(crude_estimates, ddof=1)
var_Y_beta = np.var(control_estimates, ddof=1)

print(f"Variance of Crude Monte Carlo (Var(Y)): {var_Y:.6f}")
print(f"Variance of Control Variate Estimator (Var(Y(β))): {var_Y_beta:.6f}")
