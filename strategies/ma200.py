import yfinance as yf
import pandas as pd
import numpy as np
from core.strategy_base import StrategyBase
from core.backtest_utils import structure_equity_curve, calculate_cagr, calculate_max_drawdown, print_results
from core.plot_utils import plot_equity_curve

class MA200Strategy(StrategyBase):
    def run(self, data, initial_investment, **kwargs):
        close_col = 'Close' if 'Close' in data.columns else data.columns[0]
        data = data.sort_index()
        data['MA200'] = data[close_col].rolling(200).mean()
        in_market = False
        portfolio = initial_investment
        portfolio_curve = [portfolio]
        trade_dates = [data.index[0]]
        last_price = data[close_col].iloc[0]
        for i in range(1, len(data)):
            price = data[close_col].iloc[i]
            ma200 = data['MA200'].iloc[i]
            if np.isnan(ma200):
                portfolio_curve.append(portfolio)
                trade_dates.append(data.index[i])
                continue
            if price > ma200:
                if not in_market:
                    entry_price = price
                    in_market = True
                portfolio = portfolio * (price / last_price)
            else:
                in_market = False
            last_price = price
            portfolio_curve.append(portfolio)
            trade_dates.append(data.index[i])
        return {
            'trade_dates': trade_dates,
            'portfolio_curve': portfolio_curve,
            'strategy_name': '200-day MA',
            'initial_investment': initial_investment,
            'symbol': kwargs.get('symbol', 'SPY')
        }

    def summarize(self, results, data=None):
        symbol = results.get('symbol', 'SPY')
        x_dates, equity, bh_equity = structure_equity_curve(
            results['trade_dates'], results['portfolio_curve'], results['initial_investment'], data, bh_symbol=symbol)
        returns = np.diff(equity) / equity[:-1]
        n_trades = np.sum(np.diff((equity > 0).astype(int)) != 0)
        avg_gain = np.mean(returns) * 100 if len(returns) > 0 else 0
        
        # Use standardized print function
        print_results(results['strategy_name'], equity, bh_equity, n_trades, avg_gain, "period")
        
        # Plot the equity curve
        plot_equity_curve(x_dates, equity, bh_equity, results['strategy_name'])

if __name__ == "__main__":
    from core.backtest_utils import download_data
    symbol = "SPY"
    years_back = 30
    end_date = pd.Timestamp.now()
    start_date = end_date - pd.Timedelta(days=365 * (years_back + 1))
    data = download_data(symbol, start_date, end_date)
    strategy = MA200Strategy()
    results = strategy.run(data, 10000)
    strategy.summarize(results, data) 