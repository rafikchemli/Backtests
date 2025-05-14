import os
import numpy as np
import pandas as pd
from core.strategy_base import StrategyBase
from core.backtest_utils import structure_equity_curve, print_results, download_data
from core.plot_utils import plot_equity_curve

class BerkshireStrategy(StrategyBase):
    def run(self, data, initial_investment, **kwargs):
        close_col = 'Close' if 'Close' in data.columns else data.columns[0]
        data = data.sort_index()
        equity = [initial_investment]
        trade_dates = [data.index[0]]
        start_price = data[close_col].iloc[0]
        
        for d in data.index[1:]:
            price = data[close_col].loc[d]
            equity.append(initial_investment * (price / start_price))
            trade_dates.append(d)
            
        return {
            'trade_dates': trade_dates,
            'portfolio_curve': equity,
            'strategy_name': 'Berkshire Hathaway',
            'initial_investment': initial_investment,
            'symbol': kwargs.get('symbol', 'BRK-B')
        }

    def summarize(self, results, data=None):
        # Get benchmark data (SPY) - use the provided benchmark if available
        benchmark_data = results.get('benchmark_data', None)
        
        # Get standardized benchmark if available
        standard_benchmark = results.get('standard_benchmark', None)
        
        if benchmark_data is None:
            # If no benchmark data provided, get it on demand (fallback)
            start_date = results['trade_dates'][0]
            end_date = results['trade_dates'][-1]
            benchmark_data = download_data('SPY', start_date, end_date)
        
        # Calculate BRK-B equity curve
        x_dates, equity, bh_equity = structure_equity_curve(
            results['trade_dates'], 
            results['portfolio_curve'], 
            results['initial_investment'], 
            data, 
            bh_symbol='SPY',  # Always use SPY as benchmark
            benchmark_data=benchmark_data  # Pass the consistent benchmark data
        )
        
        # Calculate returns and metrics
        returns = np.diff(equity) / equity[:-1]
        # For buy and hold, we only have 1 trade (the initial buy)
        n_trades = 1
        # Calculate total return rather than average daily return
        total_gain = ((equity[-1] / equity[0]) - 1) * 100
        
        # Use the standard print_results function with standard benchmark
        print_results("Berkshire Hathaway vs S&P 500", equity, bh_equity, n_trades, total_gain, "buy and hold", standard_benchmark)
        
        # Plot the equity curve
        plot_equity_curve(x_dates, equity, bh_equity, results['strategy_name'], bh_label="S&P 500")

if __name__ == "__main__":
    from core.backtest_utils import download_data
    symbol = "BRK-B"
    years_back = 20
    end_date = pd.Timestamp.now()
    start_date = end_date - pd.Timedelta(days=365 * (years_back + 1))
    data = download_data(symbol, start_date, end_date)
    strategy = BerkshireStrategy()
    results = strategy.run(data, 10000)
    strategy.summarize(results, data) 