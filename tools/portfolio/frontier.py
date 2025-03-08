import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.optimize import minimize


def download_stock_data(tickers, start_date, end_date):
    """Download stock data for the given tickers and date range."""
    stock_data = yf.download(tickers, start=start_date, end=end_date)["Close"]
    returns = stock_data.pct_change().dropna()
    return returns


def portfolio_annualized_performance(weights, mean_returns, cov_matrix):
    """Calculate annualized return and volatility for a portfolio."""
    returns = np.sum(mean_returns * weights) * 252
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return std, returns


def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    """Calculate the negative Sharpe ratio (for minimization)."""
    p_std, p_ret = portfolio_annualized_performance(weights, mean_returns, cov_matrix)
    return -(p_ret - risk_free_rate) / p_std


def max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate):
    """Find the portfolio weights that maximize the Sharpe ratio."""
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate)
    constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
    bound = (0.0, 1.0)
    bounds = tuple(bound for asset in range(num_assets))

    result = minimize(
        neg_sharpe_ratio,
        num_assets * [1.0 / num_assets],
        args=args,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    return result


def portfolio_volatility(weights, mean_returns, cov_matrix):
    """Calculate portfolio volatility."""
    return portfolio_annualized_performance(weights, mean_returns, cov_matrix)[0]


def min_variance(mean_returns, cov_matrix):
    """Find the portfolio weights that minimize variance."""
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)
    constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
    bound = (0.0, 1.0)
    bounds = tuple(bound for asset in range(num_assets))

    result = minimize(
        portfolio_volatility,
        num_assets * [1.0 / num_assets],
        args=args,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    return result


def efficient_frontier(mean_returns, cov_matrix, returns_range):
    """Calculate efficient frontier."""
    efficient_portfolios = []
    num_assets = len(mean_returns)

    for ret in returns_range:
        constraints = (
            {
                "type": "eq",
                "fun": lambda x: portfolio_annualized_performance(
                    x, mean_returns, cov_matrix
                )[1]
                - ret,
            },
            {"type": "eq", "fun": lambda x: np.sum(x) - 1},
        )
        bounds = tuple((0.0, 1.0) for asset in range(num_assets))
        result = minimize(
            portfolio_volatility,
            num_assets * [1.0 / num_assets],
            args=(mean_returns, cov_matrix),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )
        efficient_portfolios.append(result["x"])

    return efficient_portfolios


def plot_efficient_frontier(
    mean_returns, cov_matrix, risk_free_rate=0.02, num_portfolios=10000
):
    """Plot the efficient frontier."""
    # Generate random portfolios
    results = np.zeros((3, num_portfolios))
    weights_record = []

    for i in range(num_portfolios):
        weights = np.random.random(len(mean_returns))
        weights /= np.sum(weights)
        weights_record.append(weights)
        portfolio_std, portfolio_return = portfolio_annualized_performance(
            weights, mean_returns, cov_matrix
        )
        results[0, i] = portfolio_std
        results[1, i] = portfolio_return
        results[2, i] = (
            portfolio_return - risk_free_rate
        ) / portfolio_std  # Sharpe ratio

    # Find optimal portfolios
    max_sharpe = max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate)
    sdp, rp = portfolio_annualized_performance(
        max_sharpe["x"], mean_returns, cov_matrix
    )

    min_vol = min_variance(mean_returns, cov_matrix)
    sdp_min, rp_min = portfolio_annualized_performance(
        min_vol["x"], mean_returns, cov_matrix
    )

    # Calculate efficient frontier
    target_returns = np.linspace(rp_min, np.max(results[1]), 100)
    efficient_portfolios = efficient_frontier(mean_returns, cov_matrix, target_returns)

    # Calculate efficient frontier curve
    volatilities = [
        portfolio_volatility(weights, mean_returns, cov_matrix)
        for weights in efficient_portfolios
    ]

    # Plot
    plt.figure(figsize=(12, 8))
    plt.scatter(
        results[0, :],
        results[1, :],
        c=results[2, :],
        cmap="viridis",
        marker="o",
        s=10,
        alpha=0.3,
    )
    plt.colorbar(label="Sharpe Ratio")
    plt.scatter(sdp, rp, marker="*", color="r", s=500, label="Maximum Sharpe Ratio")
    plt.scatter(
        sdp_min, rp_min, marker="*", color="g", s=500, label="Minimum Volatility"
    )
    plt.plot(
        volatilities, target_returns, "k--", linewidth=2, label="Efficient Frontier"
    )

    # Capital Market Line
    plt.plot([0, sdp], [risk_free_rate, rp], "r-", label="Capital Market Line")

    plt.title("Portfolio Optimization and Efficient Frontier")
    plt.xlabel("Annualized Volatility")
    plt.ylabel("Annualized Returns")
    plt.legend()
    plt.grid(True)
    return plt, max_sharpe, min_vol


def display_optimal_portfolios(tickers, max_sharpe, min_vol):
    """Display the assets allocation for optimal portfolios."""
    max_sharpe_allocation = pd.DataFrame(
        max_sharpe["x"], index=tickers, columns=["Allocation"]
    )
    max_sharpe_allocation["Allocation"] = [
        round(i * 100, 2) for i in max_sharpe_allocation["Allocation"]
    ]

    min_vol_allocation = pd.DataFrame(
        min_vol["x"], index=tickers, columns=["Allocation"]
    )
    min_vol_allocation["Allocation"] = [
        round(i * 100, 2) for i in min_vol_allocation["Allocation"]
    ]

    print("-" * 80)
    print("Maximum Sharpe Ratio Portfolio Allocation:")
    print(max_sharpe_allocation)
    print(
        "\nAnnualized Return:",
        round(
            portfolio_annualized_performance(max_sharpe["x"], mean_returns, cov_matrix)[
                1
            ]
            * 100,
            2,
        ),
        "%",
    )
    print(
        "Annualized Volatility:",
        round(
            portfolio_annualized_performance(max_sharpe["x"], mean_returns, cov_matrix)[
                0
            ]
            * 100,
            2,
        ),
        "%",
    )
    print("-" * 80)
    print("Minimum Volatility Portfolio Allocation:")
    print(min_vol_allocation)
    print(
        "\nAnnualized Return:",
        round(
            portfolio_annualized_performance(min_vol["x"], mean_returns, cov_matrix)[1]
            * 100,
            2,
        ),
        "%",
    )
    print(
        "Annualized Volatility:",
        round(
            portfolio_annualized_performance(min_vol["x"], mean_returns, cov_matrix)[0]
            * 100,
            2,
        ),
        "%",
    )
    print("-" * 80)


if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "META"]

    # Define parameters
    start_date = "2020-01-01"  # Starting date for historical data
    end_date = "2024-01-01"  # Ending date for historical data
    risk_free_rate = 0.02

    # Download stock data
    returns = download_stock_data(tickers, start_date, end_date)

    # Calculate mean returns and covariance matrix
    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    # Generate and plot efficient frontier
    plt, max_sharpe, min_vol = plot_efficient_frontier(
        mean_returns, cov_matrix, risk_free_rate, 5000
    )

    # Display optimal portfolio allocations
    display_optimal_portfolios(tickers, max_sharpe, min_vol)

    plt.show()
