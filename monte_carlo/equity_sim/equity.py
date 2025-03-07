import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from backtesting.test import SMA
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional
from enum import Enum
import streamlit.components.v1 as components
import tempfile
from pathlib import Path
import os


class SampleMethod(Enum):
    SHUFFLE = "Shuffle"
    BOOTSTRAP = "Bootstrap"


class SimpleMovingAverageStrategy(Strategy):

    fast_ma = 20
    slow_ma = 50

    def init(self):
        self.fast = self.I(SMA, self.data.Close, self.fast_ma)
        self.slow = self.I(SMA, self.data.Close, self.slow_ma)

    def next(self):
        if crossover(self.fast, self.slow):
            self.position.close()
            self.buy()
        elif crossover(self.slow, self.fast):
            self.position.close()
            self.sell()


# class RSIStrategy(Strategy):
#     """RSI-based strategy."""

#     rsi_period = 14
#     rsi_oversold = 30
#     rsi_overbought = 70

#     def init(self):
#         # Calculate RSI using backtesting.py's method
#         # First calculate price changes
#         close = self.data.Close

#         # Use native backtesting.py for calculations
#         diff = self.I(lambda x: x.diff(), close)

#         # Separate gains and losses
#         up = self.I(lambda x: x.clip(lower=0), diff)
#         down = self.I(lambda x: -x.clip(upper=0), diff)

#         # Calculate rolling averages
#         ma_up = self.I(lambda x: x.rolling(self.rsi_period).mean(), up)
#         ma_down = self.I(lambda x: x.rolling(self.rsi_period).mean(), down)

#         # Calculate RS and RSI
#         rs = self.I(lambda x, y: x / y, ma_up, ma_down)
#         self.rsi = self.I(lambda x: 100 - (100 / (1 + x)), rs)

#     def next(self):
#         if self.rsi[-1] < self.rsi_oversold and not self.position:
#             self.buy()
#         elif self.rsi[-1] > self.rsi_overbought and self.position:
#             self.sell()


# class BollingerBandsStrategy(Strategy):
#     """Bollinger Bands strategy."""

#     bb_period = 20
#     bb_std = 2

#     def init(self):
#         close = self.data.Close

#         self.sma = self.I(SMA, close, self.bb_period)
#         bb_std = self.I(lambda x: x.rolling(self.bb_period).std(), close)

#         self.upper = self.I(lambda x, y: x + self.bb_std * y, self.sma, bb_std)
#         self.lower = self.I(lambda x, y: x - self.bb_std * y, self.sma, bb_std)

#     def next(self):
#         if self.data.Close[-1] < self.lower[-1] and not self.position:
#             self.buy()
#         elif self.data.Close[-1] > self.upper[-1] and self.position:
#             self.sell()


def get_stock_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    try:
        data = yf.download(ticker, start=start_date, end=end_date)

        if data.empty:
            return pd.DataFrame()

        data.reset_index(inplace=True)
        data.columns = ["Date", "Close", "High", "Low", "Open", "Volume"]
        return data

    except Exception as e:
        st.error(f"Error downloading data: {str(e)}")
        return pd.DataFrame()


def run_backtest(
    data: pd.DataFrame,
    strategy_class: Strategy,
    strategy_params: Optional[Dict[str, Any]] = None,
    cash: float = 10000,
    commission: float = 0.002,
) -> Tuple[Backtest, Dict[str, Any]]:
    bt = Backtest(
        data, strategy_class, cash=cash, commission=commission, exclusive_orders=True
    )
    stats = bt.run(**(strategy_params or {}))
    return bt, stats


def extract_trades(stats: Dict[str, Any]) -> pd.DataFrame:
    trades = stats._trades
    return trades


def monte_carlo_simulation(
    trades: pd.DataFrame,
    n_simulations: int = 1000,
    sample_method: SampleMethod = SampleMethod.SHUFFLE,
    initial_cash: float = 10000,
) -> Tuple[List[pd.Series], pd.Series]:
    if len(trades) == 0:
        raise ValueError("No trades found in the backtest results")

    trade_returns = trades["ReturnPct"].values
    drawdowns = []
    final_balances = []
    max_wins = []
    max_losses = []
    max_win_streaks = []
    max_loss_streaks = []

    fig_equity, ax_equity = plt.subplots(figsize=(10, 5))

    for _ in range(n_simulations):
        num_trades = len(trades)

        if sample_method == SampleMethod.BOOTSTRAP:
            sampled_indices = np.random.choice(
                num_trades, size=num_trades, replace=True
            )
            sampled_returns = trade_returns[sampled_indices]
        else:
            sampled_returns = np.random.permutation(trade_returns)

        equity = initial_cash
        peak_equity = equity
        max_drawdown = 0

        current_drawdown = 0
        win_streak = 0
        loss_streak = 0
        max_win_streak = 0
        max_loss_streak = 0
        max_win = 0
        max_loss = 0

        equity_curve = [equity]

        for ret in sampled_returns:
            profit_loss = equity * ret
            equity += profit_loss

            # Calculate drawdown
            if equity > peak_equity:
                peak_equity = equity
                current_drawdown = 0
            else:
                current_drawdown = (peak_equity - equity) / peak_equity
                max_drawdown = max(max_drawdown, current_drawdown)

            if ret > 0:
                win_streak += 1
                loss_streak = 0
                max_win = max(max_win, ret)
            elif ret < 0:
                loss_streak += 1
                win_streak = 0
                max_loss = min(max_loss, ret)

            max_win_streak = max(max_win_streak, win_streak)
            max_loss_streak = max(max_loss_streak, loss_streak)

            equity_curve.append(equity)

            if equity <= 0:
                break

        final_balances.append(equity)
        drawdowns.append(max_drawdown)
        max_wins.append(max_win)
        max_losses.append(max_loss)
        max_win_streaks.append(max_win_streak)
        max_loss_streaks.append(max_loss_streak)

        drawdowns.append(max_drawdown)
        ax_equity.plot(equity_curve, alpha=0.1, color="blue")

    ax_equity.set_xlabel("Trade Number")
    ax_equity.set_ylabel("Equity ($)")
    ax_equity.set_title("Monte Carlo Simulation - Equity Curves")
    ax_equity.grid(True, alpha=0.3)
    st.pyplot(fig_equity)
    plt.close(fig_equity)

    # **Plot Monte Carlo Drawdown Distribution**
    fig_drawdown, ax_drawdown = plt.subplots(figsize=(10, 5))

    # Histogram of drawdowns
    counts, bins, patches = ax_drawdown.hist(
        drawdowns, bins=30, color="green", alpha=0.6, density=True
    )

    # Cumulative distribution line
    cumulative = np.cumsum(counts) / np.sum(counts)
    ax_drawdown.plot(
        bins[:-1],
        cumulative,
        color="blue",
        linestyle="--",
        label="Cumulative Distribution",
    )

    # 95% confidence level drawdown
    drawdown_95 = np.percentile(drawdowns, 95)
    ax_drawdown.axvline(
        drawdown_95,
        color="red",
        linestyle="dashed",
        label=f"95% Conf. Level ({drawdown_95:.2%})",
    )

    # Mark 95% drawdown point with red X
    ax_drawdown.scatter(
        drawdown_95,
        np.interp(drawdown_95, bins[:-1], cumulative),
        color="red",
        marker="x",
        s=100,
    )

    # Labels and legend
    ax_drawdown.set_xlabel("Max Drawdown (%)")
    ax_drawdown.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, pos=None: f"{x * 100:.0f}")
    )
    ax_drawdown.set_ylabel("Frequency")
    ax_drawdown.set_title("Monte Carlo Drawdown Distribution")
    ax_drawdown.legend()
    ax_drawdown.grid(True, alpha=0.3)

    st.pyplot(fig_drawdown)
    plt.close(fig_drawdown)

    return (
        final_balances,
        drawdowns,
        max_wins,
        max_losses,
        max_win_streaks,
        max_loss_streaks,
    )


def main():
    """Main function for Streamlit app."""

    st.set_page_config(
        page_title="Backtesting & Monte Carlo Analysis", page_icon="📈", layout="wide"
    )

    st.title("Trading Strategy Backtesting & Monte Carlo Analysis")

    # Create tabs for better organization
    tab_data, tab_backtest, tab_monte_carlo = st.tabs(
        ["Data", "Backtest", "Monte Carlo"]
    )

    # Sidebar for all parameters
    st.sidebar.header("Data Parameters")
    ticker = st.sidebar.text_input("Stock Ticker", value="SPY")

    # Date range selection with default values
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 3)  # 3 years by default

    start_date = st.sidebar.date_input("Start Date", value=start_date)
    end_date = st.sidebar.date_input("End Date", value=end_date)

    # Strategy selection
    st.sidebar.header("Strategy Parameters")
    strategy_option = st.sidebar.selectbox(
        # "Strategy Type", ["Moving Average Crossover", "RSI", "Bollinger Bands"]
        "Strategy Type",
        ["Moving Average Crossover"],
    )

    # Strategy-specific parameters
    if strategy_option == "Moving Average Crossover":
        strategy_class = SimpleMovingAverageStrategy
        fast_ma = st.sidebar.slider(
            "Fast MA Period", min_value=5, max_value=50, value=20
        )
        slow_ma = st.sidebar.slider(
            "Slow MA Period", min_value=20, max_value=200, value=50
        )

        strategy_params = {"fast_ma": fast_ma, "slow_ma": slow_ma}

    # elif strategy_option == "RSI":
    #     strategy_class = RSIStrategy
    #     rsi_period = st.sidebar.slider(
    #         "RSI Period", min_value=5, max_value=30, value=14
    #     )
    #     rsi_oversold = st.sidebar.slider(
    #         "RSI Oversold Level", min_value=10, max_value=40, value=30
    #     )
    #     rsi_overbought = st.sidebar.slider(
    #         "RSI Overbought Level", min_value=60, max_value=90, value=70
    #     )

    #     strategy_params = {
    #         "rsi_period": rsi_period,
    #         "rsi_oversold": rsi_oversold,
    #         "rsi_overbought": rsi_overbought,
    #     }

    # elif strategy_option == "Bollinger Bands":
    #     strategy_class = BollingerBandsStrategy
    #     bb_period = st.sidebar.slider(
    #         "Bollinger Bands Period", min_value=5, max_value=50, value=20
    #     )
    #     bb_std = st.sidebar.slider(
    #         "Standard Deviation Multiplier",
    #         min_value=1.0,
    #         max_value=4.0,
    #         value=2.0,
    #         step=0.1,
    #     )

    #     strategy_params = {"bb_period": bb_period, "bb_std": bb_std}

    # Backtest parameters
    st.sidebar.header("Backtest Parameters")
    initial_cash = st.sidebar.number_input(
        "Initial Cash", min_value=1000, max_value=1000000, value=10000, step=1000
    )
    commission = (
        st.sidebar.number_input(
            "Commission (%)", min_value=0.0, max_value=5.0, value=0.1, step=0.05
        )
        / 100
    )

    # Monte Carlo parameters
    st.sidebar.header("Monte Carlo Parameters")
    n_simulations = st.sidebar.slider(
        "Number of Simulations", min_value=10, max_value=1000, value=100
    )
    sample_method_str = st.sidebar.selectbox(
        "Sampling Method", ["Shuffle", "Bootstrap"]
    )
    sample_method = SampleMethod(sample_method_str)

    with tab_data:
        data_load_state = st.text("Loading data...")
        try:
            data = get_stock_data(
                ticker, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
            )
            if data.empty:
                st.error(f"No data available for {ticker} in the specified date range.")
                return

            data_load_state.text(
                f"Data loaded successfully! Found {len(data)} data points."
            )

            # Price chart
            st.subheader("Price and Volume Chart")
            fig, (ax_price, ax_volume) = plt.subplots(
                2,
                1,
                figsize=(12, 8),
                gridspec_kw={"height_ratios": [3, 1]},
                sharex=True,
            )

            # Price chart
            ax_price.plot(data["Date"], data["Close"], label="Close Price")
            ax_price.set_title(f"{ticker} Price Chart")
            ax_price.set_ylabel("Price")
            ax_price.grid(True, alpha=0.3)

            # Volume chart
            ax_volume.bar(data["Date"], data["Volume"], color="blue", alpha=0.5)
            ax_volume.set_title(f"{ticker} Volume Chart")
            ax_volume.set_xlabel("Date")
            ax_volume.set_ylabel("Volume")
            ax_volume.grid(True, alpha=0.3)

            fig.autofmt_xdate()
            st.pyplot(fig)
            plt.close((fig))

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            data_load_state.text(f"Error loading data: {str(e)}")
            return

    # Store data in session state for use across tabs
    if "data" not in st.session_state:
        st.session_state.data = data if "data" in locals() and not data.empty else None

    # Backtest tab
    with tab_backtest:
        if st.session_state.data is None:
            st.warning("Please load data in the Data tab first.")
        else:
            if st.button("Run Backtest"):
                with st.spinner("Running backtest..."):
                    try:
                        bt, stats = run_backtest(
                            st.session_state.data,
                            strategy_class,
                            strategy_params,
                            cash=initial_cash,
                            commission=commission,
                        )

                        st.session_state.bt = bt
                        st.session_state.stats = stats
                        st.session_state.trades = extract_trades(stats)

                        with tempfile.NamedTemporaryFile(
                            suffix=".html", delete=False
                        ) as tmp:
                            temp_path = tmp.name  # This is a string, not a PosixPath

                            # Generate the plot and save it to the temporary file
                            bt.plot(filename=temp_path, open_browser=False)

                            # Display the plot in Streamlit
                            with open(temp_path, "r", encoding="utf-8") as f:
                                html_data = f.read()

                            # Render the HTML in Streamlit
                            components.html(html_data, height=700)

                        # Clean up the temporary file
                        os.unlink(temp_path)

                        col1, col2 = st.columns([1, 3])

                        with col1:
                            st.subheader("Stats")
                            st.write(st.session_state.stats)

                        with col2:
                            st.subheader("Trades")
                            st.dataframe(extract_trades(st.session_state.stats))

                    except Exception as e:
                        st.error(f"Backtest error: {str(e)}")
    with tab_monte_carlo:
        if (
            "trades" not in st.session_state
            or st.session_state.trades is None
            or len(st.session_state.trades) == 0
        ):
            st.warning("Please run a backtest with at least one trade first.")
        else:
            if st.button("Run Monte Carlo Simulation"):
                with st.spinner(
                    f"Running {n_simulations} simulations using {sample_method.value} sampling..."
                ):
                    try:
                        # Run Monte Carlo simulation
                        (
                            final_balances,
                            drawdowns,
                            max_wins,
                            max_losses,
                            max_win_streaks,
                            max_loss_streaks,
                        ) = monte_carlo_simulation(
                            st.session_state.trades,
                            n_simulations=n_simulations,
                            sample_method=sample_method,
                            initial_cash=initial_cash,
                        )

                        # Calculate statistics
                        results = {
                            "Metric": [
                                "Best Drawdown",
                                "Worst Drawdown",
                                "Average Drawdown",
                                "Average Max Win",
                                "Max Consecutive Wins",
                                "Average Consecutive Wins",
                                "Average Max Loss",
                                "Max Consecutive Losses",
                                "Average Consecutive Losses",
                            ],
                            "Value": [
                                min(drawdowns),  # Best Drawdown (least negative)
                                max(drawdowns),  # Worst Drawdown (most negative)
                                sum(drawdowns) / len(drawdowns),  # Average Drawdown
                                sum(max_wins) / len(max_wins),  # Average Max Win
                                max(max_win_streaks),  # Max Consecutive Wins
                                sum(max_win_streaks)
                                / len(max_win_streaks),  # Average Consecutive Wins
                                sum(max_losses) / len(max_losses),  # Average Max Loss
                                max(max_loss_streaks),  # Max Consecutive Losses
                                sum(max_loss_streaks)
                                / len(max_loss_streaks),  # Average Consecutive Losses
                            ],
                        }

                        # Convert to DataFrame
                        results_df = pd.DataFrame(results)

                        col1, col2 = st.columns(2)

                        with col1:
                            # Display Results
                            st.subheader("Monte Carlo Results")
                            st.dataframe(results_df)

                        # Expected shortfall (conditional value at risk) at 5%
                        returns = [
                            (b - initial_cash) / initial_cash * 100
                            for b in final_balances
                        ]
                        worst_returns = sorted(returns)[: int(len(returns) * 0.05)]
                        cvar_5 = (
                            np.mean(worst_returns) if worst_returns else np.min(returns)
                        )

                        with col2:
                            st.subheader("Risk Metrics")
                            risk_metrics = {
                                "Metric": [
                                    "Expected Shortfall (CVaR) at 5%",
                                    "Probability of 50% Drawdown",
                                    "Probability of Negative Return",
                                    "Return-to-Risk Ratio",
                                    # "Monte Carlo Sharpe Ratio",
                                ],
                                "Value": [
                                    f"{cvar_5:.2f}%",
                                    f"{sum(1 for d in drawdowns if d >= 50) / len(drawdowns) * 100:.2f}%",
                                    f"{sum(1 for r in returns if r < 0) / len(returns) * 100:.2f}%",
                                    f"{np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 'N/A'}",
                                    # f"{np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 'N/A'}",
                                ],
                            }

                            st.table(pd.DataFrame(risk_metrics))

                        # Calculate and display confidence intervals
                        st.subheader("Confidence Intervals")

                        # Create confidence interval table
                        ci_data = {
                            "Metric": ["Final Equity", "Return %", "Max Drawdown %"],
                            "5th Percentile": [
                                f"${np.percentile(final_balances, 5):.2f}",
                                f"{np.percentile([(b - initial_cash) / initial_cash * 100 for b in final_balances], 5):.2f}%",
                                f"{np.percentile(drawdowns, 5):.2f}%",
                            ],
                            "25th Percentile": [
                                f"${np.percentile(final_balances, 25):.2f}",
                                f"{np.percentile([(b - initial_cash) / initial_cash * 100 for b in final_balances], 25):.2f}%",
                                f"{np.percentile(drawdowns, 25):.2f}%",
                            ],
                            "50th Percentile (Median)": [
                                f"${np.percentile(final_balances, 50):.2f}",
                                f"{np.percentile([(b - initial_cash) / initial_cash * 100 for b in final_balances], 50):.2f}%",
                                f"{np.percentile(drawdowns, 50):.2f}%",
                            ],
                            "75th Percentile": [
                                f"${np.percentile(final_balances, 75):.2f}",
                                f"{np.percentile([(b - initial_cash) / initial_cash * 100 for b in final_balances], 75):.2f}%",
                                f"{np.percentile(drawdowns, 75):.2f}%",
                            ],
                            "95th Percentile": [
                                f"${np.percentile(final_balances, 95):.2f}",
                                f"{np.percentile([(b - initial_cash) / initial_cash * 100 for b in final_balances], 95):.2f}%",
                                f"{np.percentile(drawdowns, 95):.2f}%",
                            ],
                        }

                        st.table(pd.DataFrame(ci_data))

                    except Exception as e:
                        st.error(f"Monte Carlo simulation error: {str(e)}")


if __name__ == "__main__":
    main()
