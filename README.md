# Backtests Project

## Overview
This project provides a modular framework for backtesting various trading strategies on historical market data. Each strategy is implemented as a separate module, and all strategies share common utilities for data handling, metrics, and plotting.

## Project Structure
```
Backtests/
  strategies/
    ma200/
    overnight_swing/
    sell_may-oct/
    spy_tlt_rotation/
    turn_of_month/
  core/
    backtest_utils.py      # General backtest helpers (download, metrics, etc.)
    plot_utils.py          # Plotting styles and helpers
    strategy_base.py       # Base class/interface for strategies
  main.py                 # Entry point: select and run strategies
  requirements.txt
  README.md
```

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run a specific strategy or all strategies:
   ```bash
   python main.py --strategy ma200
   python main.py --all
   ```
   Use `--help` for all options.

## Adding a New Strategy
- Create a new folder in `strategies/` and add your strategy script.
- Implement a class that inherits from `StrategyBase` and implements `run` and `summarize` methods.
- Register your strategy in `main.py`.

## Results & Plots
- After running, results and equity curve plots are saved in each strategy's `figures/` directory.
- The output includes summary statistics (CAGR, drawdown, etc.) and comparison to Buy & Hold.

## Dependencies
- See `requirements.txt` for all required packages (e.g., yfinance, pandas, numpy, matplotlib).

## License
MIT License 