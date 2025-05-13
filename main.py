import argparse
import importlib
import os
from datetime import datetime, timedelta
from core.backtest_utils import download_data
from core.strategy_base import StrategyBase
import pandas as pd

# Register strategies here: {name: (module_path, class_name, symbol, years_back, initial_investment)}
STRATEGIES = {
    "ma200": ("strategies.ma200", "MA200Strategy", "SPY", 20, 10000),
    "monday_reversal": ("strategies.monday_reversal", "MondayReversalStrategy", "SPY", 20, 10000),
    "overnight_swing": ("strategies.overnight_swing", "OvernightSwingStrategy", "SPY", 20, 10000),
    "sell_may_oct": ("strategies.sellmay", "SellMayStrategy", ["SPY"], 20, 10000),
    "spy_tlt_rotation": ("strategies.spy_tlt_rotation", "SpyTltRotationStrategy", ["SPY", "TLT"], 20, 10000),
    "turn_of_month": ("strategies.turn_of_month", "TurnOfMonthStrategy", "SPY", 20, 10000),
    #"buy_hold": ("core.buy_hold", "BuyHoldStrategy", "SPY", 20, 10000),
}

def run_strategy(strategy_key):
    module_path, class_name, symbol, years_back, initial_investment = STRATEGIES[strategy_key]
    module = importlib.import_module(module_path)
    strategy_class = getattr(module, class_name)
    # Date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * (years_back + 1))
    # Download data
    if strategy_key == "sell_may_oct":
        # Download data for all stocks and pass as dict
        data = {s: download_data(s, start_date, end_date) for s in symbol}
    elif isinstance(symbol, list):
        data = download_data(symbol, start_date, end_date)
        # For multi-index DataFrame, convert to dict of DataFrames
        if isinstance(data, pd.DataFrame) and isinstance(data.columns, pd.MultiIndex):
            data = {s: data[s].dropna() for s in symbol}
    else:
        data = download_data(symbol, start_date, end_date)
    # Run strategy
    strategy = strategy_class()
    results = strategy.run(data, initial_investment, symbol=symbol)
    strategy.summarize(results, data)
    return results

def main():
    parser = argparse.ArgumentParser(description="Run backtest strategies.")
    parser.add_argument('--strategy', type=str, choices=STRATEGIES.keys(), help='Strategy to run')
    parser.add_argument('--all', action='store_true', help='Run all strategies')
    args = parser.parse_args()

    if args.all:
        for key in STRATEGIES:
            print(f"\nRunning {key}...")
            run_strategy(key)
    elif args.strategy:
        run_strategy(args.strategy)
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 