"""
GOOGL Monte Carlo Simulation
============================
This simulation uses geometric Brownian motion (GBM) to model thousands of
possible price paths for GOOGL over a one-year horizon. The model takes the
current price, the expected annual return (drift), and the annualized
volatility as inputs, then generates random price paths to map out the range
of possible outcomes.

The output gives you probability distributions of:
- Where GOOGL might end up in 12 months
- The probability of hitting various drawdown levels
- The probability of breaching specific exit triggers
- The expected value and risk profile of a $10,000 position

Run with:  python src/monte_carlo.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render charts to file without needing a display.

import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# INPUT PARAMETERS - the assumptions driving everything
# ============================================================
# Current price of GOOGL (as of late April 2026, post-Q1 earnings).
S0 = 345.00

# Expected annual return (drift) - the most important and most uncertain input.
# 12% combines ~4.3% earnings yield + ~14% expected EPS growth - some multiple
# compression. This is what professional investors call the "fundamental
# expected return".
MU = 0.12

# Annualized volatility - GOOGL's historical 3-year volatility.
SIGMA = 0.30

# Time horizon - 1 year of trading days.
T = 1.0
TRADING_DAYS = 252
DT = T / TRADING_DAYS

# 10,000 simulations is the standard for serious work (accuracy vs. speed).
NUM_SIMULATIONS = 10_000

# Position size assumption - a $10,000 investment.
POSITION_SIZE_DOLLARS = 10_000

# Output location for the chart.
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
SEED = 42


def simulate_price_paths(s0, mu, sigma, dt, trading_days, num_simulations):
    """Generate GBM price paths.

    Returns an array of shape (num_simulations, trading_days + 1); every path
    starts at s0. The +1 column is the starting price.
    """
    price_paths = np.zeros((num_simulations, trading_days + 1))
    price_paths[:, 0] = s0

    # GBM step: P_t = P_{t-1} * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z).
    # The -0.5*sigma^2 term is the Ito correction (log vs. arithmetic returns).
    for t in range(1, trading_days + 1):
        z = np.random.standard_normal(num_simulations)
        drift_term = (mu - 0.5 * sigma**2) * dt
        diffusion_term = sigma * np.sqrt(dt) * z
        price_paths[:, t] = price_paths[:, t - 1] * np.exp(drift_term + diffusion_term)

    return price_paths


def compute_max_drawdowns(price_paths):
    """Return the worst peak-to-trough drawdown for each simulated path.

    Vectorised across all paths: running_max tracks the highest price seen so
    far along each path, and the drawdown is how far below that peak the price
    fell. (The original computed this with a Python loop over every path.)
    """
    running_max = np.maximum.accumulate(price_paths, axis=1)
    drawdowns = (price_paths - running_max) / running_max
    return drawdowns.min(axis=1)


def print_statistics(s0, final_prices, final_returns, max_drawdowns):
    """Print the full text summary of simulation outcomes."""
    print("=" * 65)
    print("GOOGL MONTE CARLO SIMULATION RESULTS")
    print(f"Current Price: ${s0:.2f}")
    print("Time Horizon: 1 year")
    print(f"Number of Simulations: {NUM_SIMULATIONS:,}")
    print(f"Assumed Annual Return: {MU*100:.1f}%")
    print(f"Assumed Annual Volatility: {SIGMA*100:.1f}%")
    print("=" * 65)

    print("\nPRICE OUTCOMES IN 1 YEAR:")
    print(f"  Mean Final Price:    ${np.mean(final_prices):.2f}")
    print(f"  Median Final Price:  ${np.median(final_prices):.2f}")
    print(f"  5th percentile:      ${np.percentile(final_prices, 5):.2f}  (very bad outcome)")
    print(f"  25th percentile:     ${np.percentile(final_prices, 25):.2f}  (below average)")
    print(f"  75th percentile:     ${np.percentile(final_prices, 75):.2f}  (above average)")
    print(f"  95th percentile:     ${np.percentile(final_prices, 95):.2f}  (very good outcome)")

    print("\nRETURN OUTCOMES IN 1 YEAR:")
    print(f"  Mean Return:         {np.mean(final_returns)*100:+.1f}%")
    print(f"  Median Return:       {np.median(final_returns)*100:+.1f}%")
    print(f"  5th percentile:      {np.percentile(final_returns, 5)*100:+.1f}%")
    print(f"  95th percentile:     {np.percentile(final_returns, 95)*100:+.1f}%")

    print("\nPROBABILITY OF KEY OUTCOMES:")
    print(f"  Probability of profit:           {(final_returns > 0).mean()*100:.1f}%")
    print(f"  Probability of >20% gain:        {(final_returns > 0.20).mean()*100:.1f}%")
    print(f"  Probability of >50% gain:        {(final_returns > 0.50).mean()*100:.1f}%")
    print(f"  Probability of any loss:         {(final_returns < 0).mean()*100:.1f}%")
    print(f"  Probability of >20% loss:        {(final_returns < -0.20).mean()*100:.1f}%")
    print(f"  Probability of >30% loss:        {(final_returns < -0.30).mean()*100:.1f}%")

    print("\nMAXIMUM DRAWDOWN DURING THE YEAR:")
    print("  (Even if final return is positive, how far did it drop along the way?)")
    print(f"  Mean Max Drawdown:               {np.mean(max_drawdowns)*100:.1f}%")
    print(f"  Median Max Drawdown:             {np.median(max_drawdowns)*100:.1f}%")
    print(f"  Probability of >20% drawdown:    {(max_drawdowns < -0.20).mean()*100:.1f}%")
    print(f"  Probability of >30% drawdown:    {(max_drawdowns < -0.30).mean()*100:.1f}%")
    print(f"  Probability of >40% drawdown:    {(max_drawdowns < -0.40).mean()*100:.1f}%")

    print("\nDOLLAR OUTCOMES FOR A $10,000 POSITION:")
    position_outcomes = POSITION_SIZE_DOLLARS * (1 + final_returns)
    print(f"  Mean Final Value:    ${np.mean(position_outcomes):,.0f}")
    print(f"  Median Final Value:  ${np.median(position_outcomes):,.0f}")
    print(f"  5% Worst Case:       ${np.percentile(position_outcomes, 5):,.0f}")
    print(f"  5% Best Case:        ${np.percentile(position_outcomes, 95):,.0f}")
    print(f"  Worst Single Sim:    ${position_outcomes.min():,.0f}")
    print(f"  Best Single Sim:     ${position_outcomes.max():,.0f}")


def plot_results(price_paths, final_prices, final_returns, max_drawdowns, s0):
    """Build the 2x2 panel of charts and save it to outputs/."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: a sample of price paths to visualise the range of outcomes.
    ax1 = axes[0, 0]
    sample_indices = np.random.choice(len(price_paths), 100, replace=False)
    for idx in sample_indices:
        ax1.plot(price_paths[idx, :], alpha=0.1, color="steelblue", linewidth=0.5)
    ax1.plot(np.percentile(price_paths, 50, axis=0), color="black", linewidth=2, label="Median path")
    ax1.plot(np.percentile(price_paths, 5, axis=0), color="red", linewidth=1.5, linestyle="--", label="5th percentile (bad)")
    ax1.plot(np.percentile(price_paths, 95, axis=0), color="green", linewidth=1.5, linestyle="--", label="95th percentile (good)")
    ax1.axhline(y=s0, color="gray", linestyle=":", label=f"Starting price ${s0}")
    ax1.set_title("Sample Price Paths Over 1 Year", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Trading Days")
    ax1.set_ylabel("Price ($)")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Plot 2: distribution of final prices.
    ax2 = axes[0, 1]
    ax2.hist(final_prices, bins=60, color="steelblue", alpha=0.7, edgecolor="black", linewidth=0.5)
    ax2.axvline(s0, color="gray", linestyle=":", linewidth=2, label=f"Starting price ${s0}")
    ax2.axvline(np.mean(final_prices), color="black", linewidth=2, label=f"Mean ${np.mean(final_prices):.0f}")
    ax2.axvline(np.percentile(final_prices, 5), color="red", linestyle="--", linewidth=1.5, label=f"5th %ile ${np.percentile(final_prices, 5):.0f}")
    ax2.axvline(np.percentile(final_prices, 95), color="green", linestyle="--", linewidth=1.5, label=f"95th %ile ${np.percentile(final_prices, 95):.0f}")
    ax2.set_title("Distribution of Final Prices in 1 Year", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Final Price ($)")
    ax2.set_ylabel("Frequency")
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Plot 3: distribution of returns.
    ax3 = axes[1, 0]
    ax3.hist(final_returns * 100, bins=60, color="steelblue", alpha=0.7, edgecolor="black", linewidth=0.5)
    ax3.axvline(0, color="gray", linestyle=":", linewidth=2, label="Break-even")
    ax3.axvline(np.mean(final_returns) * 100, color="black", linewidth=2, label=f"Mean {np.mean(final_returns)*100:+.1f}%")
    ax3.axvline(-20, color="orange", linestyle="--", linewidth=1.5, label="-20% threshold")
    ax3.axvline(-30, color="red", linestyle="--", linewidth=1.5, label="-30% threshold")
    ax3.set_title("Distribution of 1-Year Returns", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Return (%)")
    ax3.set_ylabel("Frequency")
    ax3.legend(loc="upper right", fontsize=9)
    ax3.grid(True, alpha=0.3)

    # Plot 4: distribution of maximum drawdowns.
    ax4 = axes[1, 1]
    ax4.hist(max_drawdowns * 100, bins=60, color="salmon", alpha=0.7, edgecolor="black", linewidth=0.5)
    ax4.axvline(np.mean(max_drawdowns) * 100, color="black", linewidth=2, label=f"Mean {np.mean(max_drawdowns)*100:.1f}%")
    ax4.axvline(-20, color="orange", linestyle="--", linewidth=1.5, label="-20% drawdown")
    ax4.axvline(-30, color="red", linestyle="--", linewidth=1.5, label="-30% drawdown")
    ax4.set_title("Distribution of Max Drawdowns Within Year", fontsize=12, fontweight="bold")
    ax4.set_xlabel("Maximum Drawdown (%)")
    ax4.set_ylabel("Frequency")
    ax4.legend(loc="upper left", fontsize=9)
    ax4.grid(True, alpha=0.3)

    plt.suptitle("GOOGL Monte Carlo Simulation: 10,000 Paths Over 1 Year", fontsize=14, fontweight="bold", y=1.00)
    plt.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "googl_monte_carlo.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    # Seed for reproducibility - anyone running this gets the same results.
    np.random.seed(SEED)

    price_paths = simulate_price_paths(S0, MU, SIGMA, DT, TRADING_DAYS, NUM_SIMULATIONS)
    final_prices = price_paths[:, -1]
    final_returns = (final_prices - S0) / S0
    max_drawdowns = compute_max_drawdowns(price_paths)

    print_statistics(S0, final_prices, final_returns, max_drawdowns)
    out_path = plot_results(price_paths, final_prices, final_returns, max_drawdowns, S0)
    print(f"\nVisualization saved to {out_path}")


if __name__ == "__main__":
    main()
