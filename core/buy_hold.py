import os
import numpy as np
from core.strategy_base import StrategyBase
from core.backtest_utils import structure_equity_curve, calculate_cagr, calculate_max_drawdown
from core.plot_utils import plot_equity_curve

class BuyHoldStrategy(StrategyBase):
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
            'strategy_name': 'Buy & Hold',
            'initial_investment': initial_investment
        }

    def summarize(self, results, data=None):
        x_dates, equity, _ = structure_equity_curve(
            results['trade_dates'], results['portfolio_curve'], results['initial_investment'], data, bh_symbol=None)
        returns = np.diff(equity) / equity[:-1]
        n_trades = len(equity) - 1
        avg_gain = np.mean(returns) * 100 if len(returns) > 0 else 0
        cagr = calculate_cagr(equity)
        max_dd = calculate_max_drawdown(equity)
        print(f"\n===== {results['strategy_name']} =====")
        print(f"Periods: {n_trades}")
        print(f"Average gain per period: {avg_gain:.2f}%")
        print(f"CAGR (approx): {cagr:.2f}%")
        print(f"Max drawdown: {max_dd:.2f}%")
        print(f"Final equity: ${equity[-1]:.2f}")
        plot_equity_curve(x_dates, equity, equity, results['strategy_name'], bh_label="Buy & Hold") 