import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy.stats import norm


def simulate_brownian_motion(S0, mu, sigma, T, dt, paths):
    """Simulates stock price paths using Geometric Brownian Motion (GBM)."""
    N = int(T / dt)  # Number of time steps
    t = np.linspace(0, T, N)
    dt_sqrt = np.sqrt(dt)

    prices = np.zeros((paths, N))
    prices[:, 0] = S0

    for i in range(1, N):
        dW = np.random.randn(paths) * dt_sqrt  # Brownian motion
        dS = mu * prices[:, i - 1] * dt + sigma * prices[:, i - 1] * dW
        prices[:, i] = prices[:, i - 1] + dS

    return t, prices


def simulate_jump_diffusion(S0, mu, sigma, lambda_, jump_mean, jump_std, T, dt, paths):
    """Simulates stock price paths with Merton's Jump-Diffusion Model."""
    N = int(T / dt)
    t = np.linspace(0, T, N)
    dt_sqrt = np.sqrt(dt)

    prices = np.zeros((paths, N))
    prices[:, 0] = S0

    for i in range(1, N):
        dW = np.random.randn(paths) * dt_sqrt
        jumps = np.random.poisson(lambda_ * dt, paths) * np.random.normal(
            jump_mean, jump_std, paths
        )

        dS = mu * prices[:, i - 1] * dt + sigma * prices[:, i - 1] * dW + jumps
        prices[:, i] = prices[:, i - 1] + dS

    return t, prices


def simulate_heston(S0, mu, v0, kappa, theta, sigma_v, rho, T, dt, paths):
    """Simulates stock price paths with the Heston Stochastic Volatility Model."""
    N = int(T / dt)
    t = np.linspace(0, T, N)
    dt_sqrt = np.sqrt(dt)

    prices = np.zeros((paths, N))
    volatilities = np.zeros((paths, N))
    prices[:, 0] = S0
    volatilities[:, 0] = v0

    for i in range(1, N):
        dW1 = np.random.randn(paths) * dt_sqrt
        dW2 = np.random.randn(paths) * dt_sqrt
        dW2 = rho * dW1 + np.sqrt(1 - rho**2) * dW2

        volatilities[:, i] = np.maximum(
            0,
            volatilities[:, i - 1]
            + kappa * (theta - volatilities[:, i - 1]) * dt
            + sigma_v * np.sqrt(volatilities[:, i - 1]) * dW2,
        )
        dS = (
            mu * prices[:, i - 1] * dt
            + np.sqrt(volatilities[:, i]) * prices[:, i - 1] * dW1
        )
        prices[:, i] = prices[:, i - 1] + dS

    return t, prices


# Streamlit UI
st.set_page_config(page_title="Monte Carlo Stock Price Simulation")
st.title("Monte Carlo Stock Price Simulation")

# Model selection
model = st.selectbox(
    "Select Model",
    ["Brownian Motion", "Jump-Diffusion", "Heston Stochastic Volatility"],
)

# User inputs
S0 = st.number_input("Initial Stock Price", value=100.0)
mu = st.number_input("Drift (Mean Return)", value=0.05)
T = st.number_input("Time Horizon (years)", value=1.0)
dt = st.number_input("Time Step (days)", value=1.0) / 252
paths = st.number_input("Number of Simulations", value=100, step=10)

if model == "Brownian Motion":
    sigma = st.number_input("Volatility", value=0.2)
    if st.button("Run Simulation"):
        t, prices = simulate_brownian_motion(S0, mu, sigma, T, dt, paths)

elif model == "Jump-Diffusion":
    sigma = st.number_input("Volatility", value=0.2)
    lambda_ = st.number_input("Jump Intensity (per year)", value=0.5)
    jump_mean = st.number_input("Mean Jump Size", value=0.02)
    jump_std = st.number_input("Jump Size Volatility", value=0.05)

    if st.button("Run Simulation"):
        t, prices = simulate_jump_diffusion(
            S0, mu, sigma, lambda_, jump_mean, jump_std, T, dt, paths
        )

elif model == "Heston Stochastic Volatility":
    v0 = st.number_input("Initial Volatility", value=0.04)
    kappa = st.number_input("Mean Reversion Speed", value=2.0)
    theta = st.number_input("Long-term Variance", value=0.04)
    sigma_v = st.number_input("Volatility of Volatility", value=0.5)
    rho = st.number_input("Correlation between Price and Volatility", value=-0.7)

    if st.button("Run Simulation"):
        t, prices = simulate_heston(
            S0, mu, v0, kappa, theta, sigma_v, rho, T, dt, paths
        )

# Plot results
if "prices" in locals():
    plt.figure(figsize=(10, 5))
    for i in range(paths):
        plt.plot(t, prices[i], alpha=0.3)
    plt.xlabel("Time (years)")
    plt.ylabel("Stock Price")
    plt.title(f"Monte Carlo Simulation: {model}")
    st.pyplot(plt)

    # Compute statistical results
    final_prices = prices[:, -1]
    mean_price = np.mean(final_prices)
    median_price = np.median(final_prices)
    std_price = np.std(final_prices)
    min_price = np.min(final_prices)
    max_price = np.max(final_prices)

    # Plot histogram and normal distribution curve
    plt.figure(figsize=(10, 5))
    # Plot histogram of the final prices
    plt.hist(final_prices, bins=30, density=True, alpha=0.6, color="b")

    # Plot normal distribution curve
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mean_price, std_price)
    plt.plot(x, p, "k", linewidth=2)

    plt.title(f"Histogram of Final Prices with Normal Distribution Curve")
    plt.xlabel("Final Price")
    plt.ylabel("Density")
    st.pyplot(plt)

    st.subheader("Statistical Results")
    st.write(f"Mean Final Price: {mean_price:.2f}")
    st.write(f"Median Final Price: {median_price:.2f}")
    st.write(f"Standard Deviation: {std_price:.2f}")
    st.write(f"Minimum Final Price: {min_price:.2f}")
    st.write(f"Maximum Final Price: {max_price:.2f}")
