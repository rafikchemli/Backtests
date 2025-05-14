import os
import numpy as np
import pandas as pd
from core.strategy_base import StrategyBase
from core.backtest_utils import structure_equity_curve, calculate_cagr, calculate_max_drawdown, print_results, download_data
from core.plot_utils import plot_equity_curve

class SpyTltRotationStrategy(StrategyBase):
    def run(self, data, initial_investment, **kwargs):
        symbols = list(data.keys()) if isinstance(data, dict) else ["SPY", "TLT"]
        close_col = 'Close' if 'Close' in data[symbols[0]].columns else data[symbols[0]].columns[0]
        idx = data[symbols[0]].index
        
        # Start with first date for proper benchmark alignment
        first_date = data[symbols[0]].index[0]
        
        months = pd.Series(idx).dt.to_period('M').unique()
        portfolio = initial_investment
        portfolio_curve = [portfolio]
        trade_dates = [first_date]  # Include first date for alignment
        
        for i in range(1, len(months)):
            prev_month = months[i-1]
            this_month = months[i]
            prev_month_data = {sym: data[sym].loc[data[sym].index.to_period('M') == prev_month] for sym in symbols}
            this_month_data = {sym: data[sym].loc[data[sym].index.to_period('M') == this_month] for sym in symbols}
            if any(len(df) == 0 for df in prev_month_data.values()) or any(len(df) == 0 for df in this_month_data.values()):
                continue
            perf = {}
            for sym in symbols:
                start = prev_month_data[sym][close_col].iloc[0]
                end = prev_month_data[sym][close_col].iloc[-1]
                perf[sym] = (end - start) / start
            best = max(perf, key=perf.get)
            entry_date = this_month_data[best].index[0]
            entry_open = this_month_data[best]['Open'].iloc[0] if 'Open' in this_month_data[best].columns else this_month_data[best][close_col].iloc[0]
            exit_date = this_month_data[best].index[-1]
            exit_close = this_month_data[best][close_col].iloc[-1]
            trade_return = (exit_close - entry_open) / entry_open
            portfolio = portfolio * (1 + trade_return)
            portfolio_curve.append(portfolio)
            trade_dates.append(exit_date)
            
        return {
            'trade_dates': trade_dates,
            'portfolio_curve': portfolio_curve,
            'strategy_name': 'SPY-TLT Rotation',
            'initial_investment': initial_investment,
            'symbol': 'SPY'  # Use standard symbol for benchmark
        }

    def summarize(self, results, data=None):
        # Use SPY as standard benchmark
        symbol = 'SPY'
        
        # Get benchmark data (SPY) if available
        benchmark_data = results.get('benchmark_data', None)
        
        # If no benchmark data provided, get it on demand (fallback)
        if benchmark_data is None:
            start_date = results['trade_dates'][0]
            end_date = results['trade_dates'][-1]
            benchmark_data = download_data(symbol, start_date, end_date)
        
        # Structure equity curves with benchmark data
        x_dates, equity, bh_equity = structure_equity_curve(
            results['trade_dates'], 
            results['portfolio_curve'], 
            results['initial_investment'], 
            data[symbol] if isinstance(data, dict) and symbol in data else None,
            bh_symbol=symbol,
            benchmark_data=benchmark_data
        )
        
        # Calculate metrics
        returns = np.diff(equity) / equity[:-1]
        n_trades = len(equity) - 1
        avg_gain = np.mean(returns) * 100 if len(returns) > 0 else 0
        
        # Use standardized print function
        print_results(results['strategy_name'], equity, bh_equity, n_trades, avg_gain, "month")
        
        # Plot the equity curve
        plot_equity_curve(x_dates, equity, bh_equity, results['strategy_name'], bh_label="Buy & Hold (SPY)")

if __name__ == "__main__":
    from core.backtest_utils import download_data
    symbols = ["SPY", "TLT"]
    years_back = 20
    end_date = pd.Timestamp.now()
    start_date = end_date - pd.Timedelta(days=365 * (years_back + 1))
    data = download_data(symbols, start_date, end_date)
    
    # Download benchmark data separately for consistency
    benchmark_data = download_data('SPY', start_date, end_date)
    
    if isinstance(data, pd.DataFrame) and isinstance(data.columns, pd.MultiIndex):
        data = {sym: data[sym].dropna() for sym in symbols}
        
    strategy = SpyTltRotationStrategy()
    results = strategy.run(data, 10000)
    
    # Add benchmark data to results
    results['benchmark_data'] = benchmark_data
    
    strategy.summarize(results, data) 