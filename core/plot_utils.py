import os

import matplotlib.pyplot as plt


def plot_equity_curve(x_dates, equity, bh_equity, strategy_name, bh_label="Buy & Hold"):
    # Central figures directory
    base_figures_dir = os.path.join(os.getcwd(), "figures")
    os.makedirs(base_figures_dir, exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(x_dates, equity, label=strategy_name)
    plt.plot(x_dates, bh_equity, label=bh_label, linestyle="--")
    plt.title(f"{strategy_name} vs {bh_label}")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value ($)")
    plt.legend()
    plt.grid(True)
    fname = f"{strategy_name.replace(' ', '_').lower()}.png"
    plt.savefig(os.path.join(base_figures_dir, fname))
