import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp
from scipy.stats import norm
import yfinance as yf

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Portfolio VaR Simulator",
    page_icon="📈",
    layout="wide",
)

# ── Global plot style ─────────────────────────────────────────────────────────
sns.set_style("white")
plt.rcParams["figure.autolayout"] = True
np.set_printoptions(precision=4, suppress=True)

# ── Helper ────────────────────────────────────────────────────────────────────
def simulate_var(pv, er, vol, T, iterations):
    """Geometric Brownian Motion Monte Carlo VaR."""
    end = pv * np.exp(
        (er - 0.5 * vol**2) * T
        + vol * np.sqrt(T) * np.random.standard_normal(iterations)
    )
    return end - pv


def simulate_paths(pv, er, vol, t, num_paths=100):
    """Generate GBM price paths."""
    num_steps = max(int(t * 252), 1)
    dt = t / num_steps
    paths = np.zeros((num_steps + 1, num_paths))
    paths[0] = pv
    for i in range(num_paths):
        for j in range(1, num_steps + 1):
            drift = (er - 0.5 * vol**2) * dt
            diffusion = vol * np.sqrt(dt) * np.random.standard_normal()
            paths[j, i] = paths[j - 1, i] * np.exp(drift + diffusion)
    return paths


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR – inputs
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("⚙️ Portfolio Setup")

    stock_input = st.text_input(
        "Stock Tickers (comma-separated)",
        value="AAPL,TSLA,MSFT",
        help="e.g. AAPL,TSLA,MSFT",
    )

    portfolio_value = st.number_input(
        "Initial Portfolio Value ($)",
        min_value=1_000,
        max_value=1_000_000_000,
        value=1_000_000,
        step=10_000,
        format="%d",
    )

    trading_days = st.slider(
        "Time Horizon (Trading Days)",
        min_value=1,
        max_value=252,
        value=21,
    )

    iterations = st.select_slider(
        "Monte Carlo Iterations",
        options=[10_000, 25_000, 50_000, 100_000],
        value=50_000,
    )

    st.divider()
    st.subheader("Manual Overrides (optional)")
    st.caption("Leave at 0 to use values calculated from historical data.")

    manual_er = st.number_input(
        "Expected Annual Return (%)",
        min_value=0.0,
        max_value=200.0,
        value=15.0,
        step=0.5,
        format="%.2f",
    )
    manual_vol = st.number_input(
        "Annual Volatility (%)",
        min_value=0.0,
        max_value=200.0,
        value=27.49,
        step=0.5,
        format="%.2f",
    )

    run = st.button("▶ Run Simulation", type="primary", use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN PANEL
# ══════════════════════════════════════════════════════════════════════════════
st.title("📊 Portfolio Value-at-Risk (VaR) Simulator")
st.caption(
    "Monte Carlo simulation using Geometric Brownian Motion to estimate portfolio risk."
)

if not run:
    st.info("Configure your portfolio in the sidebar and click **▶ Run Simulation**.")
    st.stop()

# ── Parse tickers ─────────────────────────────────────────────────────────────
tickers_list = [t.strip().upper() for t in stock_input.split(",") if t.strip()]
if not tickers_list:
    st.error("Please enter at least one valid ticker symbol.")
    st.stop()

weights = np.array([1 / len(tickers_list)] * len(tickers_list))

# ── Fetch historical data ─────────────────────────────────────────────────────
with st.spinner("Fetching historical data from Yahoo Finance…"):
    end_date = pd.Timestamp.now()
    start_date = end_date - pd.DateOffset(years=1)
    try:
        hist = yf.download(
            tickers_list,
            start=start_date,
            end=end_date,
            auto_adjust=True,
            progress=False,
        )["Close"]
    except Exception as e:
        st.error(f"Failed to download data: {e}")
        st.stop()

if hist.empty:
    st.error("No data returned. Check your ticker symbols and try again.")
    st.stop()

# Handle single-ticker edge case (yfinance returns a Series)
if isinstance(hist, pd.Series):
    hist = hist.to_frame(name=tickers_list[0])

# Drop tickers with no data
valid_tickers = [t for t in tickers_list if t in hist.columns and hist[t].notna().any()]
if not valid_tickers:
    st.error("None of the tickers returned valid data.")
    st.stop()

if len(valid_tickers) < len(tickers_list):
    missing = set(tickers_list) - set(valid_tickers)
    st.warning(f"No data found for: {', '.join(missing)}. Continuing with {valid_tickers}.")
    tickers_list = valid_tickers
    weights = np.array([1 / len(tickers_list)] * len(tickers_list))

hist = hist[tickers_list].dropna(how="all")

# ── Calculate portfolio statistics ───────────────────────────────────────────
log_returns = np.log(hist / hist.shift(1)).dropna()
portfolio_daily_returns = (log_returns * weights).sum(axis=1)

portfolio_daily_er = portfolio_daily_returns.mean()
portfolio_daily_vol = portfolio_daily_returns.std()

er_calculated = portfolio_daily_er * 252
vol_calculated = portfolio_daily_vol * np.sqrt(252)

er = manual_er / 100 if manual_er > 0 else er_calculated
vol = manual_vol / 100 if manual_vol > 0 else vol_calculated
er_source = "Manual Override" if manual_er > 0 else "Calculated"
vol_source = "Manual Override" if manual_vol > 0 else "Calculated"

pv = portfolio_value
t_years = trading_days / 252

# ── Run simulation ────────────────────────────────────────────────────────────
with st.spinner("Running Monte Carlo simulation…"):
    at_risk = simulate_var(pv, er, vol, t_years, iterations)

var_1 = np.percentile(at_risk, 1)
var_5 = np.percentile(at_risk, 5)
var_10 = np.percentile(at_risk, 10)

# ══════════════════════════════════════════════════════════════════════════════
# RESULTS – KPI cards
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("📋 Portfolio Summary")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Initial Value", f"${pv:,.0f}")
col2.metric(f"Expected Return ({er_source})", f"{er*100:.2f}%")
col3.metric(f"Volatility ({vol_source})", f"{vol*100:.2f}%")
col4.metric("Time Horizon", f"{trading_days} days")

st.divider()

# ── VaR metrics ───────────────────────────────────────────────────────────────
st.subheader("🎯 Value at Risk")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Mean P&L", f"${at_risk.mean():,.0f}")
c2.metric("Std Dev", f"${at_risk.std():,.0f}")
c3.metric("1% VaR", f"${var_1:,.0f}", delta_color="inverse")
c4.metric("5% VaR", f"${var_5:,.0f}", delta_color="inverse")
c5.metric("10% VaR", f"${var_10:,.0f}", delta_color="inverse")

st.caption(
    "VaR values represent the maximum loss NOT exceeded at each confidence level "
    "over the specified horizon."
)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# CHARTS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs(
    ["📉 P&L Distribution", "📈 CDF", "🌊 Full Distribution", "🛤️ Price Paths"]
)

# ── Tab 1: Histogram with VaR lines ──────────────────────────────────────────
with tab1:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(at_risk, bins=100, kde=True, color="steelblue", edgecolor="white", ax=ax)
    ax.axvline(var_1, color="red", linestyle="--", label=f"1% VaR: ${var_1:,.0f}")
    ax.axvline(var_5, color="orange", linestyle="--", label=f"5% VaR: ${var_5:,.0f}")
    ax.axvline(var_10, color="green", linestyle="--", label=f"10% VaR: ${var_10:,.0f}")
    ax.set_title(f"Distribution of Simulated Portfolio Value Changes — {', '.join(tickers_list)}")
    ax.set_xlabel("Change in Portfolio Value ($)")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    st.pyplot(fig)
    plt.close(fig)

# ── Tab 2: CDF ────────────────────────────────────────────────────────────────
with tab2:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.ecdfplot(at_risk, color="royalblue", linewidth=2, ax=ax)
    for level, color, label in [
        (0.01, "red", "1% VaR"),
        (0.05, "orange", "5% VaR"),
        (0.10, "green", "10% VaR"),
    ]:
        val = np.percentile(at_risk, level * 100)
        ax.axhline(y=level, color=color, linestyle="--", label=label)
        ax.axvline(x=val, color=color, linestyle="--")
    ax.set_title(f"CDF of Simulated Portfolio Value Changes — {', '.join(tickers_list)}")
    ax.set_xlabel("Change in Portfolio Value ($)")
    ax.set_ylabel("Cumulative Probability")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    st.pyplot(fig)
    plt.close(fig)

# ── Tab 3: Full distribution ──────────────────────────────────────────────────
with tab3:
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.histplot(at_risk, bins=100, kde=True, color="mediumpurple", edgecolor="white", ax=ax)
    ax.axvline(x=0, color="gray", linestyle=":", label="No Change")
    ax.set_title(f"Full Distribution of Simulated Portfolio Profit/Loss — {', '.join(tickers_list)}")
    ax.set_xlabel("Change in Portfolio Value ($)")
    ax.set_ylabel("Frequency")
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

# ── Tab 4: Monte Carlo price paths ────────────────────────────────────────────
with tab4:
    num_paths = st.slider("Number of paths to plot", 10, 300, 100, step=10)
    with st.spinner("Simulating price paths…"):
        paths = simulate_paths(pv, er, vol, t_years, num_paths=num_paths)
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(paths, alpha=0.3, linewidth=0.7)
    ax.set_title(f"Monte Carlo Simulated Price Paths — {', '.join(tickers_list)}")
    ax.set_xlabel("Time Steps (Days)")
    ax.set_ylabel("Portfolio Value ($)")
    ax.grid(True, linestyle="--", alpha=0.5)
    st.pyplot(fig)
    plt.close(fig)

st.divider()

# ── Raw summary table ─────────────────────────────────────────────────────────
with st.expander("📄 Full Risk Statistics Table"):
    summary = pd.DataFrame(
        {
            "Statistic": [
                "Mean Change in Portfolio Value",
                "Standard Deviation of Change",
                "1% Value at Risk (VaR)",
                "5% Value at Risk (VaR)",
                "10% Value at Risk (VaR)",
            ],
            "Value ($)": [
                at_risk.mean(),
                at_risk.std(),
                var_1,
                var_5,
                var_10,
            ],
        }
    )
    summary["Value ($)"] = summary["Value ($)"].map("${:,.2f}".format)
    st.dataframe(summary, use_container_width=True, hide_index=True)
