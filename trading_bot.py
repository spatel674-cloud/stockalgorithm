"""
Simple Stock Trading Algorithm
------------------------------

Educational backtesting / paper-trading system.

Strategy:
    1. Calculate a fast and slow moving average.
    2. Calculate RSI.
    3. BUY when:
         - Fast MA crosses above Slow MA
         - RSI is below the overbought threshold
    4. SELL when:
         - Fast MA crosses below Slow MA
    5. Otherwise HOLD.

IMPORTANT:
    This program does NOT place real trades.
    It is intended for learning, experimentation, and backtesting.

Install:
    pip install yfinance pandas numpy matplotlib

Run:
    python trading_bot.py
"""

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

TICKER = "NVDA"

START_DATE = "2020-01-01"
END_DATE = None

INITIAL_CASH = 10_000.00

# Strategy parameters
FAST_MA = 20
SLOW_MA = 50
RSI_PERIOD = 14
RSI_MAX_FOR_BUY = 70

# Simulated transaction cost.
# 0.001 = 0.1%
TRANSACTION_COST = 0.001

# Fraction of available cash used on a buy.
POSITION_SIZE = 1.0


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class Trade:
    date: pd.Timestamp
    action: str
    price: float
    shares: float
    cash_after: float
    portfolio_value: float


# ============================================================
# DATA DOWNLOAD
# ============================================================

def download_data(ticker: str, start: str, end: str | None = None) -> pd.DataFrame:
    """
    Download historical market data using Yahoo Finance.
    """

    print(f"Downloading data for {ticker}...")

    if end:
        data = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
        )
    else:
        data = yf.download(
            ticker,
            start=start,
            auto_adjust=True,
            progress=False,
        )

    if data.empty:
        raise RuntimeError(
            f"No data was downloaded for {ticker}. "
            "Check the ticker symbol and internet connection."
        )

    # yfinance can sometimes return MultiIndex columns.
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    required_columns = ["Open", "High", "Low", "Close", "Volume"]

    for column in required_columns:
        if column not in data.columns:
            raise RuntimeError(f"Missing required column: {column}")

    data = data[required_columns].copy()
    data.dropna(inplace=True)

    return data


# ============================================================
# INDICATORS
# ============================================================

def calculate_rsi(
    prices: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI).

    RSI ranges approximately from 0 to 100.
    """

    delta = prices.diff()

    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    average_gain = gains.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False,
    ).mean()

    average_loss = losses.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False,
    ).mean()

    # Avoid division by zero.
    average_loss = average_loss.replace(0, np.nan)

    relative_strength = average_gain / average_loss

    rsi = 100 - (100 / (1 + relative_strength))

    return rsi


def add_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators to the price dataframe.
    """

    data = data.copy()

    data["FastMA"] = (
        data["Close"]
        .rolling(window=FAST_MA)
        .mean()
    )

    data["SlowMA"] = (
        data["Close"]
        .rolling(window=SLOW_MA)
        .mean()
    )

    data["RSI"] = calculate_rsi(
        data["Close"],
        RSI_PERIOD,
    )

    return data


# ============================================================
# TRADING SIGNALS
# ============================================================

def generate_signals(data: pd.DataFrame) -> pd.DataFrame:
    """
    Generate BUY / SELL / HOLD signals.

    BUY:
        Previous fast MA <= previous slow MA
        AND
        Current fast MA > current slow MA
        AND
        RSI < RSI_MAX_FOR_BUY

    SELL:
        Previous fast MA >= previous slow MA
        AND
        Current fast MA < current slow MA
    """

    data = data.copy()

    data["Signal"] = "HOLD"

    previous_fast = data["FastMA"].shift(1)
    previous_slow = data["SlowMA"].shift(1)

    bullish_cross = (
        (previous_fast <= previous_slow)
        & (data["FastMA"] > data["SlowMA"])
    )

    bearish_cross = (
        (previous_fast >= previous_slow)
        & (data["FastMA"] < data["SlowMA"])
    )

    buy_signal = (
        bullish_cross
        & (data["RSI"] < RSI_MAX_FOR_BUY)
    )

    data.loc[buy_signal, "Signal"] = "BUY"
    data.loc[bearish_cross, "Signal"] = "SELL"

    return data


# ============================================================
# PORTFOLIO
# ============================================================

class Portfolio:
    """
    Simple long-only paper-trading portfolio.
    """

    def __init__(self, initial_cash: float):
        self.cash = initial_cash
        self.shares = 0.0

        self.trades: list[Trade] = []
        self.equity_curve: list[float] = []

    def value(self, price: float) -> float:
        """
        Current portfolio value.
        """

        return self.cash + (self.shares * price)

    def buy(
        self,
        date: pd.Timestamp,
        price: float,
    ):
        """
        Buy using the configured percentage of available cash.
        """

        if self.cash <= 0:
            return

        cash_to_use = self.cash * POSITION_SIZE

        # Account for transaction costs.
        effective_price = price * (1 + TRANSACTION_COST)

        shares = cash_to_use / effective_price

        if shares <= 0:
            return

        total_cost = shares * effective_price

        self.cash -= total_cost
        self.shares += shares

        portfolio_value = self.value(price)

        self.trades.append(
            Trade(
                date=date,
                action="BUY",
                price=price,
                shares=shares,
                cash_after=self.cash,
                portfolio_value=portfolio_value,
            )
        )

    def sell(
        self,
        date: pd.Timestamp,
        price: float,
    ):
        """
        Sell the entire position.
        """

        if self.shares <= 0:
            return

        effective_price = price * (1 - TRANSACTION_COST)

        proceeds = self.shares * effective_price

        sold_shares = self.shares

        self.cash += proceeds
        self.shares = 0.0

        portfolio_value = self.value(price)

        self.trades.append(
            Trade(
                date=date,
                action="SELL",
                price=price,
                shares=sold_shares,
                cash_after=self.cash,
                portfolio_value=portfolio_value,
            )
        )


# ============================================================
# BACKTEST
# ============================================================

def run_backtest(data: pd.DataFrame) -> tuple[Portfolio, pd.DataFrame]:
    """
    Run the strategy through historical data.
    """

    portfolio = Portfolio(INITIAL_CASH)

    portfolio_values = []

    for date, row in data.iterrows():

        price = float(row["Close"])
        signal = row["Signal"]

        if signal == "BUY":
            portfolio.buy(
                date=date,
                price=price,
            )

        elif signal == "SELL":
            portfolio.sell(
                date=date,
                price=price,
            )

        current_value = portfolio.value(price)

        portfolio_values.append(current_value)

    data = data.copy()
    data["PortfolioValue"] = portfolio_values

    portfolio.equity_curve = portfolio_values

    return portfolio, data


# ============================================================
# PERFORMANCE METRICS
# ============================================================

def calculate_metrics(
    data: pd.DataFrame,
    portfolio: Portfolio,
) -> dict:
    """
    Calculate basic performance statistics.
    """

    final_price = float(data["Close"].iloc[-1])

    final_value = portfolio.value(final_price)

    total_return = (
        final_value / INITIAL_CASH
    ) - 1

    # Buy-and-hold benchmark.
    first_price = float(data["Close"].iloc[0])

    buy_hold_return = (
        final_price / first_price
    ) - 1

    # Daily returns.
    daily_returns = (
        data["PortfolioValue"]
        .pct_change()
        .dropna()
    )

    if len(daily_returns) > 1:
        annualized_return = (
            (1 + total_return)
            ** (252 / len(daily_returns))
        ) - 1

        annualized_volatility = (
            daily_returns.std() * math.sqrt(252)
        )

        if annualized_volatility > 0:
            sharpe_ratio = (
                annualized_return
                / annualized_volatility
            )
        else:
            sharpe_ratio = 0.0
    else:
        annualized_return = 0.0
        annualized_volatility = 0.0
        sharpe_ratio = 0.0

    # Maximum drawdown.
    running_max = (
        data["PortfolioValue"]
        .cummax()
    )

    drawdown = (
        data["PortfolioValue"] / running_max
    ) - 1

    maximum_drawdown = float(drawdown.min())

    winning_trades = 0
    losing_trades = 0

    completed_trades = []

    buy_trade = None

    for trade in portfolio.trades:

        if trade.action == "BUY":
            buy_trade = trade

        elif trade.action == "SELL" and buy_trade is not None:

            profit = (
                trade.price - buy_trade.price
            ) * trade.shares

            completed_trades.append(profit)

            if profit > 0:
                winning_trades += 1
            elif profit < 0:
                losing_trades += 1

            buy_trade = None

    total_completed_trades = (
        winning_trades + losing_trades
    )

    if total_completed_trades > 0:
        win_rate = (
            winning_trades
            / total_completed_trades
        )
    else:
        win_rate = 0.0

    return {
        "initial_value": INITIAL_CASH,
        "final_value": final_value,
        "strategy_return": total_return,
        "buy_hold_return": buy_hold_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": sharpe_ratio,
        "maximum_drawdown": maximum_drawdown,
        "number_of_trades": len(portfolio.trades),
        "completed_trades": total_completed_trades,
        "winning_trades": winning_trades,
        "losing_trades": losing_trades,
        "win_rate": win_rate,
    }


# ============================================================
# PRINT RESULTS
# ============================================================

def print_results(
    ticker: str,
    metrics: dict,
    portfolio: Portfolio,
):
    """
    Display backtest results.
    """

    print()
    print("=" * 60)
    print(f"BACKTEST RESULTS: {ticker}")
    print("=" * 60)

    print(
        f"Initial portfolio: "
        f"${metrics['initial_value']:,.2f}"
    )

    print(
        f"Final portfolio:   "
        f"${metrics['final_value']:,.2f}"
    )

    print(
        f"Strategy return:   "
        f"{metrics['strategy_return']:.2%}"
    )

    print(
        f"Buy & hold return: "
        f"{metrics['buy_hold_return']:.2%}"
    )

    print(
        f"Annualized return: "
        f"{metrics['annualized_return']:.2%}"
    )

    print(
        f"Annualized risk:   "
        f"{metrics['annualized_volatility']:.2%}"
    )

    print(
        f"Sharpe ratio:      "
        f"{metrics['sharpe_ratio']:.2f}"
    )

    print(
        f"Max drawdown:      "
        f"{metrics['maximum_drawdown']:.2%}"
    )

    print(
        f"Total signals:     "
        f"{metrics['number_of_trades']}"
    )

    print(
        f"Completed trades:  "
        f"{metrics['completed_trades']}"
    )

    print(
        f"Winning trades:    "
        f"{metrics['winning_trades']}"
    )

    print(
        f"Losing trades:     "
        f"{metrics['losing_trades']}"
    )

    print(
        f"Win rate:          "
        f"{metrics['win_rate']:.2%}"
    )

    print("=" * 60)

    print()
    print("TRADE HISTORY")
    print("-" * 60)

    if not portfolio.trades:
        print("No trades generated.")
        return

    for trade in portfolio.trades:

        print(
            f"{trade.date.date()} | "
            f"{trade.action:4} | "
            f"Price: ${trade.price:,.2f} | "
            f"Shares: {trade.shares:.4f}"
        )


# ============================================================
# CHART
# ============================================================

def plot_results(
    ticker: str,
    data: pd.DataFrame,
):
    """
    Display price, moving averages, trading signals,
    and portfolio value.
    """

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(14, 10),
        sharex=True,
    )

    # --------------------------------------------------------
    # Price chart
    # --------------------------------------------------------

    axes[0].plot(
        data.index,
        data["Close"],
        label="Close",
    )

    axes[0].plot(
        data.index,
        data["FastMA"],
        label=f"{FAST_MA}-Day MA",
    )

    axes[0].plot(
        data.index,
        data["SlowMA"],
        label=f"{SLOW_MA}-Day MA",
    )

    buys = data[data["Signal"] == "BUY"]
    sells = data[data["Signal"] == "SELL"]

    axes[0].scatter(
        buys.index,
        buys["Close"],
        marker="^",
        s=80,
        label="BUY",
    )

    axes[0].scatter(
        sells.index,
        sells["Close"],
        marker="v",
        s=80,
        label="SELL",
    )

    axes[0].set_title(
        f"{ticker} Trading Strategy"
    )

    axes[0].set_ylabel("Price")

    axes[0].legend()
    axes[0].grid(True)

    # --------------------------------------------------------
    # Portfolio chart
    # --------------------------------------------------------

    axes[1].plot(
        data.index,
        data["PortfolioValue"],
        label="Strategy",
    )

    # Buy-and-hold portfolio.
    buy_hold = (
        INITIAL_CASH
        * data["Close"]
        / data["Close"].iloc[0]
    )

    axes[1].plot(
        data.index,
        buy_hold,
        label="Buy & Hold",
    )

    axes[1].set_title(
        "Portfolio Performance"
    )

    axes[1].set_ylabel("Portfolio Value ($)")
    axes[1].set_xlabel("Date")

    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()

    plt.show()


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("Simple Stock Trading Algorithm")
    print("=" * 60)
    print(f"Ticker: {TICKER}")
    print(f"Initial cash: ${INITIAL_CASH:,.2f}")
    print()

    # 1. Download historical data.
    data = download_data(
        ticker=TICKER,
        start=START_DATE,
        end=END_DATE,
    )

    # 2. Calculate indicators.
    data = add_indicators(data)

    # 3. Generate trading signals.
    data = generate_signals(data)

    # Remove rows where indicators aren't ready.
    data = data.dropna().copy()

    if data.empty:
        raise RuntimeError(
            "Not enough historical data for the selected "
            "indicator periods."
        )

    # 4. Backtest.
    portfolio, data = run_backtest(data)

    # 5. Calculate statistics.
    metrics = calculate_metrics(
        data=data,
        portfolio=portfolio,
    )

    # 6. Print results.
    print_results(
        ticker=TICKER,
        metrics=metrics,
        portfolio=portfolio,
    )

    # 7. Show chart.
    plot_results(
        ticker=TICKER,
        data=data,
    )


if __name__ == "__main__":
    main()
