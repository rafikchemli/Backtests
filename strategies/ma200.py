import yfinance as yf
import pandas as pd
import numpy as np
from core.strategy_base import StrategyBase
from core.backtest_utils import structure_equity_curve, print_results, download_data
from core.plot_utils import plot_equity_curve

class MA200Strategy(StrategyBase):
    def run(self, data, initial_investment, **kwargs):
        # Track trades for reporting
        trades = []
        
        close_col = 'Close' if 'Close' in data.columns else data.columns[0]
        data = data.sort_index()
        data['MA200'] = data[close_col].rolling(200).mean()
        
        in_market = False
        portfolio = initial_investment
        portfolio_curve = [portfolio]
        trade_dates = [data.index[0]]
        position_size = 0
        buy_trades = 0
        sell_trades = 0
        
        for i in range(1, len(data)):
            curr_date = data.index[i]
            price = data[close_col].iloc[i]
            ma200 = data['MA200'].iloc[i]
            
            # Skip calculation if MA not available yet
            if np.isnan(ma200):
                portfolio_curve.append(portfolio)
                trade_dates.append(curr_date)
                continue
            
            # Trading logic: Buy when price > MA200, Sell when price <= MA200
            if price > ma200:
                # Signal to be in the market
                if not in_market:
                    # Enter the market
                    entry_price = price
                    position_size = portfolio / entry_price
                    in_market = True
                    trades.append((entry_price, None, curr_date, "BUY"))
                    buy_trades += 1
                    
                # Update portfolio value
                portfolio = position_size * price
            else:
                # Signal to be out of the market
                if in_market:
                    # Exit the market
                    exit_price = price
                    portfolio = position_size * exit_price
                    
                    # Update the last buy with this exit price
                    for j in range(len(trades)-1, -1, -1):
                        if trades[j][3] == "BUY" and trades[j][1] is None:
                            trades[j] = (trades[j][0], exit_price, trades[j][2], "BUY-SELL")
                            break
                    
                    position_size = 0
                    in_market = False
                    sell_trades += 1
            
            # Store portfolio value and date
            portfolio_curve.append(portfolio)
            trade_dates.append(curr_date)
        
        # If still in market at the end, close the position
        if in_market:
            exit_price = data[close_col].iloc[-1]
            portfolio = position_size * exit_price
            # Update the last buy with this exit price
            for j in range(len(trades)-1, -1, -1):
                if trades[j][3] == "BUY" and trades[j][1] is None:
                    trades[j] = (trades[j][0], exit_price, trades[j][2], "BUY-SELL")
                    break
            sell_trades += 1
            portfolio_curve[-1] = portfolio
            
        # Calculate returns for completed trades
        returns = []
        for entry, exit, _, trade_type in trades:
            if entry is not None and exit is not None:
                returns.append((exit - entry) / entry)
        
        return {
            'trades': trades,
            'returns': returns,
            'trade_dates': trade_dates,
            'portfolio_curve': portfolio_curve,
            'strategy_name': '200-day MA',
            'initial_investment': initial_investment,
            'symbol': kwargs.get('symbol', 'SPY')
        }

    def summarize(self, results, data=None):
        # Use SPY as standard benchmark
        symbol = 'SPY'  # Standard benchmark
        
        # Get benchmark data (SPY) if available
        benchmark_data = results.get('benchmark_data', None)
        
        # Get standardized benchmark if available
        standard_benchmark = results.get('standard_benchmark', None)
        
        # If no benchmark data provided, get it on demand (fallback)
        if benchmark_data is None:
            start_date = results['trade_dates'][0]
            end_date = results['trade_dates'][-1]
            benchmark_data = download_data(symbol, start_date, end_date)
        
        x_dates, equity, bh_equity = structure_equity_curve(
            results['trade_dates'], 
            results['portfolio_curve'], 
            results['initial_investment'], 
            data, 
            bh_symbol=symbol,
            benchmark_data=benchmark_data
        )
        
        returns = np.diff(equity) / equity[:-1]
        n_trades = len([t for t in results['trades'] if t[3] in ["BUY", "BUY-SELL"]])
        avg_gain = np.mean(results['returns']) * 100 if len(results['returns']) > 0 else 0
        
        # Use standardized print function with standard benchmark
        print_results(results['strategy_name'], equity, bh_equity, n_trades, avg_gain, "trade", standard_benchmark)
        
        # Plot the equity curve
        plot_equity_curve(x_dates, equity, bh_equity, results['strategy_name'])

if __name__ == "__main__":
    from core.backtest_utils import download_data
    symbol = "SPY"
    years_back = 20
    end_date = pd.Timestamp.now()
    start_date = end_date - pd.Timedelta(days=365 * (years_back + 1))
    data = download_data(symbol, start_date, end_date)
    
    # Download benchmark data for consistency
    benchmark_data = download_data('SPY', start_date, end_date)
    
    strategy = MA200Strategy()
    results = strategy.run(data, 10000)
    
    # Add benchmark data to results
    results['benchmark_data'] = benchmark_data
    
    strategy.summarize(results, data) 