import argparse
import os
from core.backtest_utils import (
    get_cached_data, calculate_date_range, load_strategy,
    download_data
)
import pandas as pd
import numpy as np

# Global benchmark data cache - will store downloaded data to ensure consistency
BENCHMARK_DATA = {}
# Store standard benchmark equity values for consistent comparison
STANDARD_BENCHMARK = {}

# Register strategies here: {name: (module_path, class_name, symbol, years_back, initial_investment)}
STRATEGIES = {
    "ma200": ("strategies.ma200", "MA200Strategy", "SPY", 20, 10000),
    "monday_reversal": ("strategies.monday_reversal", "MondayReversalStrategy", "SPY", 20, 10000),
    "overnight_swing": ("strategies.overnight_swing", "OvernightSwingStrategy", "SPY", 20, 10000),
    "sell_may_oct": ("strategies.sellmay", "SellMayStrategy", ["SPY"], 20, 10000),
    "spy_tlt_rotation": ("strategies.spy_tlt_rotation", "SpyTltRotationStrategy", ["SPY", "TLT"], 20, 10000),
    "turn_of_month": ("strategies.turn_of_month", "TurnOfMonthStrategy", "SPY", 20, 10000),
    "berkshire": ("strategies.berkshire", "BerkshireStrategy", "BRK-B", 4, 10000),
}

def create_standard_benchmark(benchmark_data, initial_investment):
    """
    Create a standardized benchmark equity curve.
    This ensures all strategies compare to the exact same benchmark values.
    
    Parameters:
    - benchmark_data: DataFrame with benchmark price data
    - initial_investment: Initial investment amount
    
    Returns:
    - Dictionary with standard benchmark equity values indexed by date
    """
    global STANDARD_BENCHMARK
    
    if not STANDARD_BENCHMARK:
        close_col = 'Close' if 'Close' in benchmark_data.columns else benchmark_data.columns[0]
        start_price = benchmark_data[close_col].iloc[0]
        standard_equity = {}
        
        for date in benchmark_data.index:
            price = benchmark_data.loc[date, close_col]
            equity = initial_investment * (price / start_price)
            standard_equity[date] = equity
        
        STANDARD_BENCHMARK = standard_equity
    
    return STANDARD_BENCHMARK

def prepare_benchmark(start_date, end_date, initial_investment):
    """
    Prepare the standard benchmark data for comparison.
    
    Parameters:
    - start_date: Start date for the benchmark
    - end_date: End date for the benchmark
    - initial_investment: Initial investment amount
    
    Returns:
    - Tuple of (benchmark_data, standard_benchmark)
    """
    benchmark_symbol = 'SPY'  # Standard benchmark for all strategies
    
    # Get cached benchmark data
    benchmark_data = get_cached_data(benchmark_symbol, start_date, end_date, BENCHMARK_DATA)
    
    # Create or get standardized benchmark equity values
    standard_benchmark = create_standard_benchmark(benchmark_data, initial_investment)
    
    return benchmark_data, standard_benchmark

def run_strategy(strategy_key):
    """
    Run a backtest strategy based on the strategy key.
    
    Parameters:
    - strategy_key: Key identifying the strategy in the STRATEGIES dictionary
    
    Returns:
    - Dictionary containing backtest results
    """
    try:
        print(f"\n=== Running strategy: {strategy_key} ===")
        
        # Load strategy configuration
        strategy_class, symbol, years_back, initial_investment = load_strategy(strategy_key, STRATEGIES)
        
        # Calculate date range
        start_date, end_date = calculate_date_range(years_back)
        
        # Record time period for reference
        time_period = {
            'start_date': start_date,
            'end_date': end_date,
            'years_back': years_back
        }
        
        # Prepare benchmark data and standardized benchmark
        benchmark_data, standard_benchmark = prepare_benchmark(start_date, end_date, initial_investment)
        
        # Get strategy-specific data
        data = get_cached_data(symbol, start_date, end_date, BENCHMARK_DATA)
        
        # Create strategy instance and run backtest
        strategy = strategy_class()
        results = strategy.run(data, initial_investment, symbol=symbol)
        
        # Add metadata to results
        results.update({
            'time_period': time_period,
            'benchmark_symbol': 'SPY',
            'benchmark_data': benchmark_data,
            'standard_benchmark': standard_benchmark
        })
        
        # Summarize results
        strategy.summarize(results, data)
        
        return results
    
    except Exception as e:
        print(f"Error running strategy '{strategy_key}': {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def main():
    parser = argparse.ArgumentParser(description="Run backtest strategies.")
    parser.add_argument('--strategy', type=str, choices=STRATEGIES.keys(), help='Strategy to run')
    parser.add_argument('--all', action='store_true', help='Run all strategies')
    args = parser.parse_args()

    if args.all:
        for key in STRATEGIES:
            run_strategy(key)
    elif args.strategy:
        run_strategy(args.strategy)
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 