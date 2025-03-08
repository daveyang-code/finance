import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


def get_option_data(ticker="SPY", days_to_expiry_min=10):
    """
    Fetch option data for a given ticker and filter by minimum days to expiry
    """
    # Get the ticker object
    stock = yf.Ticker(ticker)

    # Get current price for the stock
    current_price = stock.history(period="1d")["Close"][-1]
    print(f"Current {ticker} price: ${current_price:.2f}")

    # Get expiration dates
    expirations = stock.options

    # Create an empty list to store all options data
    all_options = []
    today = datetime.now().date()

    # Loop through each expiration date
    for expiry in expirations:
        # Convert expiry to datetime
        exp_date = datetime.strptime(expiry, "%Y-%m-%d").date()

        # Calculate days to expiration
        dte = (exp_date - today).days

        # Skip if days to expiry is less than minimum
        if dte < days_to_expiry_min:
            continue

        # Get options chain for this expiration
        opt = stock.option_chain(expiry)

        # Process calls
        calls = opt.calls.copy()
        calls["optionType"] = "call"
        calls["expiration"] = expiry
        calls["dte"] = dte

        # Process puts
        puts = opt.puts.copy()
        puts["optionType"] = "put"
        puts["expiration"] = expiry
        puts["dte"] = dte

        # Append to our list
        all_options.append(calls)
        all_options.append(puts)

    # Combine all options into a single DataFrame
    if not all_options:
        raise ValueError(
            f"No options data found for {ticker} with at least {days_to_expiry_min} days to expiry"
        )

    options_df = pd.concat(all_options)

    # Add moneyness column (strike / current price)
    options_df["moneyness"] = options_df["strike"] / current_price

    # Calculate moneyness categories
    options_df["moneyness_category"] = pd.cut(
        options_df["moneyness"],
        bins=[0, 0.8, 0.95, 1.05, 1.2, float("inf")],
        labels=["Deep ITM", "ITM", "ATM", "OTM", "Deep OTM"],
    )

    return options_df, current_price


def calculate_implied_volatility_surface(options_df, current_price):
    """
    Calculate and plot the volatility surface
    """
    # Filter for options with valid implied volatility
    valid_iv = options_df[
        (options_df["impliedVolatility"] > 0) & (options_df["impliedVolatility"] < 2)
    ]  # Filter out unrealistic IVs

    # Create a pivot table for the volatility surface
    # Use moneyness and days to expiry as axes, and implied volatility as values
    # Separate for calls and puts
    call_surface = valid_iv[valid_iv["optionType"] == "call"].pivot_table(
        values="impliedVolatility", index="moneyness", columns="dte", aggfunc="mean"
    )

    put_surface = valid_iv[valid_iv["optionType"] == "put"].pivot_table(
        values="impliedVolatility", index="moneyness", columns="dte", aggfunc="mean"
    )

    return call_surface, put_surface


def create_combined_plots(call_surface, put_surface, ticker):
    """
    Create a combined figure with all four plots
    """

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.5, 1])

    # 3D plot for call surface
    ax1 = fig.add_subplot(gs[0, 0], projection="3d")
    # Create mesh grid for 3D plot
    if not call_surface.empty and len(call_surface.columns) > 0:
        x = call_surface.columns  # Days to expiry
        y = call_surface.index  # Moneyness
        X, Y = np.meshgrid(x, y)
        Z = call_surface.values

        # Plot the surface
        surf1 = ax1.plot_surface(X, Y, Z, cmap="Blues", alpha=0.8, edgecolor="k")

        # Add labels and title
        ax1.set_xlabel("Days to Expiry")
        ax1.set_ylabel("Moneyness (Strike/Spot)")
        ax1.set_zlabel("Implied Volatility")
        ax1.set_title(f"{ticker} Call Options IV Surface")

        # Add colorbar
        fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)

    # 3D plot for put surface
    ax2 = fig.add_subplot(gs[0, 1], projection="3d")
    if not put_surface.empty and len(put_surface.columns) > 0:
        x = put_surface.columns  # Days to expiry
        y = put_surface.index  # Moneyness
        X, Y = np.meshgrid(x, y)
        Z = put_surface.values

        # Plot the surface
        surf2 = ax2.plot_surface(X, Y, Z, cmap="Reds", alpha=0.8, edgecolor="k")

        # Add labels and title
        ax2.set_xlabel("Days to Expiry")
        ax2.set_ylabel("Moneyness (Strike/Spot)")
        ax2.set_zlabel("Implied Volatility")
        ax2.set_title(f"{ticker} Put Options IV Surface")

        # Add colorbar
        fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)

    # 2D plot for call volatility smile
    ax3 = fig.add_subplot(gs[1, 0])
    if not call_surface.empty and len(call_surface.columns) > 0:
        # Choose a few representative expiration dates
        dte_values = sorted(call_surface.columns)
        if len(dte_values) > 4:
            indices = np.linspace(0, len(dte_values) - 1, 4).astype(int)
            dte_values = [dte_values[i] for i in indices]

        # Plot each volatility smile
        for dte in dte_values:
            if dte in call_surface.columns:
                smile = call_surface[dte].dropna()
                ax3.plot(smile.index, smile.values, marker="o", label=f"DTE: {dte}")

        # Add labels and legend
        ax3.set_xlabel("Moneyness (Strike/Spot)")
        ax3.set_ylabel("Implied Volatility")
        ax3.set_title("Call Options Volatility Smile")
        ax3.grid(True, alpha=0.3)
        ax3.legend()

    # 2D plot for put volatility smile
    ax4 = fig.add_subplot(gs[1, 1])
    if not put_surface.empty and len(put_surface.columns) > 0:
        # Choose a few representative expiration dates
        dte_values = sorted(put_surface.columns)
        if len(dte_values) > 4:
            indices = np.linspace(0, len(dte_values) - 1, 4).astype(int)
            dte_values = [dte_values[i] for i in indices]

        # Plot each volatility smile
        for dte in dte_values:
            if dte in put_surface.columns:
                smile = put_surface[dte].dropna()
                ax4.plot(smile.index, smile.values, marker="o", label=f"DTE: {dte}")

        # Add labels and legend
        ax4.set_xlabel("Moneyness (Strike/Spot)")
        ax4.set_ylabel("Implied Volatility")
        ax4.set_title("Put Options Volatility Smile")
        ax4.grid(True, alpha=0.3)
        ax4.legend()

    plt.tight_layout()
    return fig


def analyze_volatility_surface(ticker="SPY", days_to_expiry_min=10):
    """
    Main function to analyze and visualize the volatility surface
    """
    # Get option data
    print(f"Fetching options data for {ticker}...")
    options_df, current_price = get_option_data(ticker, days_to_expiry_min)

    # Summary
    print(f"\nOptions data summary:")
    print(f"Total options: {len(options_df)}")
    print(f"Expirations: {options_df['expiration'].nunique()}")
    print(
        f"Days to expiry range: {options_df['dte'].min()} to {options_df['dte'].max()} days"
    )
    print(
        f"Strike price range: ${options_df['strike'].min():.2f} to ${options_df['strike'].max():.2f}"
    )

    # Calculate volatility surfaces
    call_surface, put_surface = calculate_implied_volatility_surface(
        options_df, current_price
    )

    combined_fig = create_combined_plots(call_surface, put_surface, ticker)

    print("\nVolatility surface statistics:")
    if not call_surface.empty:
        print(
            f"Call options IV range: {call_surface.min().min():.2%} to {call_surface.max().max():.2%}"
        )
    if not put_surface.empty:
        print(
            f"Put options IV range: {put_surface.min().min():.2%} to {put_surface.max().max():.2%}"
        )

    # Check for volatility skew
    print("\nVolatility skew analysis:")

    # For near-term calls
    if not call_surface.empty and len(call_surface.columns) > 0:
        near_term_calls = call_surface[call_surface.columns[0]].dropna()
        if len(near_term_calls) > 1:
            # Calculate the absolute difference from 1.0 for each moneyness value
            diffs = abs(near_term_calls.index - 1.0)
            # Find the index of the minimum difference (closest to ATM)
            atm_index = near_term_calls.index[diffs.argmin()]

            # Find OTM and ITM indices
            otm_indices = [idx for idx in near_term_calls.index if idx > 1.0]
            itm_indices = [idx for idx in near_term_calls.index if idx < 1.0]

            otm_index = min(otm_indices) if otm_indices else None
            itm_index = max(itm_indices) if itm_indices else None

            print(f"Near-term call options:")
            print(f"  ATM IV (moneyness ≈ 1.00): {near_term_calls.loc[atm_index]:.2%}")
            if otm_index:
                print(
                    f"  OTM IV (moneyness = {otm_index:.2f}): {near_term_calls.loc[otm_index]:.2%}"
                )
            if itm_index:
                print(
                    f"  ITM IV (moneyness = {itm_index:.2f}): {near_term_calls.loc[itm_index]:.2%}"
                )

    # For near-term puts
    if not put_surface.empty and len(put_surface.columns) > 0:
        near_term_puts = put_surface[put_surface.columns[0]].dropna()
        if len(near_term_puts) > 1:
            # Calculate the absolute difference from 1.0 for each moneyness value
            diffs = abs(near_term_puts.index - 1.0)
            # Find the index of the minimum difference (closest to ATM)
            atm_index = near_term_puts.index[diffs.argmin()]

            # Find OTM and ITM indices
            otm_indices = [idx for idx in near_term_puts.index if idx < 1.0]
            itm_indices = [idx for idx in near_term_puts.index if idx > 1.0]

            otm_index = max(otm_indices) if otm_indices else None
            itm_index = min(itm_indices) if itm_indices else None

            print(f"\nNear-term put options:")
            print(f"  ATM IV (moneyness ≈ 1.00): {near_term_puts.loc[atm_index]:.2%}")
            if otm_index:
                print(
                    f"  OTM IV (moneyness = {otm_index:.2f}): {near_term_puts.loc[otm_index]:.2%}"
                )
            if itm_index:
                print(
                    f"  ITM IV (moneyness = {itm_index:.2f}): {near_term_puts.loc[itm_index]:.2%}"
                )

    return {
        "options_df": options_df,
        "call_surface": call_surface,
        "put_surface": put_surface,
        "combined_figure": combined_fig,
    }


if __name__ == "__main__":
    ticker = "SPY"
    min_dte = 0

    results = analyze_volatility_surface(ticker, min_dte)

    plt.show()
