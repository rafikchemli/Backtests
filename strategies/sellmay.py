# Install required libraries
# !pip install yfinance pandas matplotlib seaborn

# Import libraries
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import numpy as np
import os
from core.strategy_base import StrategyBase
from core.backtest_utils import structure_equity_curve, calculate_cagr, calculate_max_drawdown, print_results, download_data
from core.plot_utils import plot_equity_curve

# Set parameters
years_back = 20  # Number of years to analyze
end_date = datetime.now()  # Current date
start_date = end_date - timedelta(days=365 * (years_back + 1))  # Extra year for Oct-Apr
initial_investment = 10000  # Starting portfolio value in USD

# List of stocks to analyze
stocks = ["SPY", "TSLA", "PLTR", "COST", "CP", "SHOP"]

class SellMayStrategy(StrategyBase):
    def run(self, data, initial_investment, **kwargs):
        """
        Implements the "Sell in May and Go Away" strategy:
        - Invested from October to April
        - Out of market from May to September
        """
        # Handle single symbol or multiple symbols
        if isinstance(data, dict):
            symbol = kwargs.get('symbol', ['SPY'])[0]
            data = data[symbol]
        else:
            symbol = kwargs.get('symbol', 'SPY')
            
        close_col = 'Close' if 'Close' in data.columns else data.columns[0]
        data = data.sort_index()
        
        # Initialize portfolio tracking
        portfolio = initial_investment
        portfolio_curve = [portfolio]
        trade_dates = [data.index[0]]
        last_price = data[close_col].iloc[0]
        
        # Track if we're in the market
        in_market = False
        
        # Process each day
        for i in range(1, len(data)):
            date = data.index[i]
            price = data[close_col].iloc[i]
            
            # Check if we should be in the market (October to April)
            # In market: October (10) through April (4)
            # Out of market: May (5) through September (9)
            month = date.month
            should_be_in_market = month >= 10 or month <= 4
            
            # Update portfolio based on market position
            if should_be_in_market:
                if not in_market:
                    in_market = True
                # When in market, apply market returns
                portfolio = portfolio * (price / last_price)
            else:
                # When out of market, portfolio stays the same (cash position)
                in_market = False
                
            last_price = price
            portfolio_curve.append(portfolio)
            trade_dates.append(date)
            
        return {
            'trade_dates': trade_dates,
            'portfolio_curve': portfolio_curve,
            'strategy_name': 'Sell in May',
            'initial_investment': initial_investment,
            'symbol': symbol
        }

    def summarize(self, results, data=None):
        """
        Print summary statistics and plot equity curve
        """
        # Use SPY as standard benchmark
        symbol = 'SPY'  # Standard benchmark
        
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
            data, 
            bh_symbol=symbol,
            benchmark_data=benchmark_data
        )
        
        # Calculate trade statistics
        returns = np.diff(equity) / equity[:-1]
        non_zero_returns = returns[returns != 0]
        n_trades = len(non_zero_returns) if len(non_zero_returns) > 0 else 0
        avg_gain = np.mean(non_zero_returns) * 100 if len(non_zero_returns) > 0 else 0
        
        # Use standardized print function
        print_results(results['strategy_name'], equity, bh_equity, n_trades, avg_gain, "period")
        
        # Plot the equity curve
        plot_equity_curve(x_dates, equity, bh_equity, results['strategy_name'])

if __name__ == "__main__":
    from core.backtest_utils import download_data
    
    symbol = "SPY"
    years_back = 20
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * (years_back + 1))
    data = download_data(symbol, start_date, end_date)
    
    # Download benchmark data for consistency
    benchmark_data = download_data('SPY', start_date, end_date)
    
    strategy = SellMayStrategy()
    results = strategy.run(data, 10000)
    
    # Add benchmark data to results
    results['benchmark_data'] = benchmark_data
    
    strategy.summarize(results, data)