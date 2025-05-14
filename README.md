# Backtests Project

## Overview
This project provides a modular framework for backtesting various trading strategies on historical market data. Each strategy is implemented as a separate module, and all strategies share common utilities for data handling, metrics, and plotting.

## Project Structure
```
Backtests/
  strategies/             # Strategy implementations
    ma200.py              # 200-day moving average strategy
    monday_reversal.py    # Monday market reversal strategy
    overnight_swing.py    # Overnight holding strategy
    sellmay.py            # Sell in May and go away strategy
    spy_tlt_rotation.py   # SPY/TLT rotation strategy
    turn_of_month.py      # Turn of the month strategy
    berkshire.py          # Berkshire Hathaway vs SPY comparison
  core/
    backtest_utils.py     # Utility functions for data handling, caching, metrics
    plot_utils.py         # Plotting functions for equity curves and stats
    strategy_base.py      # Base strategy interface class
  figures/                # Output figures directory
  main.py                 # Entry point for running strategies
  requirements.txt        # Project dependencies
  README.md               # This documentation
```

## Key Features
- Modular strategy implementation with a common interface
- Efficient data caching to avoid redundant downloads
- Standardized benchmark comparison across all strategies
- Consistent metrics and visualization

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run a specific strategy or all strategies:
   ```bash
   # Run a single strategy
   python main.py --strategy ma200
   
   # Run all registered strategies
   python main.py --all
   
   # Get help on available options
   python main.py --help
   ```

## Adding a New Strategy
1. Create a new Python file in the `strategies/` directory
2. Implement a class that inherits from `StrategyBase` with:
   - `run(self, data, initial_investment, **kwargs)` - Runs the strategy and returns results
   - `summarize(self, results, data=None)` - Prints summary statistics and creates plots

3. Register your strategy in `main.py`:
   ```python
   STRATEGIES = {
       # Add your strategy to this dictionary:
       "your_strategy_key": (
           "strategies.your_module", "YourStrategyClass", 
           "SYMBOL", years_back, initial_investment
       ),
       # ...existing strategies...
   }
   ```

## Core Utilities
- **Data Management**: Automatic caching of downloaded data
- **Benchmark Standardization**: Consistent benchmark calculations across strategies
- **Performance Metrics**: CAGR, max drawdown, trade statistics
- **Visualization**: Equity curve plots with benchmark comparison

## Results Output
Each strategy generates:
- Summary statistics in the console (CAGR, drawdown, trade metrics, etc.)
- Comparison to Buy & Hold benchmark (SPY by default)
- Equity curve plots saved in the `figures/` directory

## Dependencies
- pandas, numpy: Data manipulation
- matplotlib, seaborn: Visualization
- yfinance: Market data retrieval
- See `requirements.txt` for full details

## License
MIT License 