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
from core.backtest_utils import structure_equity_curve, calculate_cagr, calculate_max_drawdown, print_results
from core.plot_utils import plot_equity_curve

# Set parameters
years_back = 20  # Number of years to analyze
end_date = datetime.now()  # Current date
start_date = end_date - timedelta(days=365 * (years_back + 1))  # Extra year for Oct-Apr
initial_investment = 10000  # Starting portfolio value in USD

# List of stocks to analyze
stocks = ["SPY", "TSLA", "PLTR", "COST", "CP", "SHOP"]

class SellMayStrategy(StrategyBase):
    def run(self, data_dict, initial_investment, **kwargs):
        # If a single dataframe is passed, treat it as SPY data
        if not isinstance(data_dict, dict):
            symbols = ["SPY"]
            data_dict = {"SPY": data_dict}
        else:
            symbols = list(data_dict.keys())
            # If SPY is available, use it as primary symbol
            primary_symbol = "SPY" if "SPY" in symbols else symbols[0]
        
        # Get results for the primary symbol
        primary_symbol = "SPY" if "SPY" in data_dict else list(data_dict.keys())[0]
        res = self.analyze_stock(primary_symbol, data_dict[primary_symbol], initial_investment)
        
        if not res:
            return {
                'trade_dates': [],
                'portfolio_curve': [initial_investment],
                'strategy_name': 'Sell in May',
                'initial_investment': initial_investment,
                'summary': {}
            }
        
        # Convert yearly results to a continuous equity curve
        trade_dates = res["yearly_results_df"]["Year"]
        sell_may_curve = res["yearly_results_df"]["Sell-in-May Value ($)"]
        buy_hold_curve = res["yearly_results_df"]["Buy-Hold Value ($)"]
        
        return {
            'trade_dates': trade_dates,
            'portfolio_curve': sell_may_curve,
            'buy_hold_curve': buy_hold_curve,
            'strategy_name': 'Sell in May',
            'initial_investment': initial_investment,
            'symbol': primary_symbol,
            'summary': {primary_symbol: res}
        }

    def analyze_stock(self, symbol, data, initial_investment):
        if "Adj Close" in data.columns:
            price_col = "Adj Close"
        elif "Close" in data.columns:
            price_col = "Close"
        else:
            return None
        data = data[[price_col]].rename(columns={price_col: "Price"})
        actual_start_date = data.index.min()
        years_back = 20
        end_date = data.index.max()
        may_sep_returns = []
        oct_apr_returns = []
        valid_years = []
        def get_closest_date(date_str):
            date = pd.to_datetime(date_str)
            original_date = date
            while date not in data.index:
                date += pd.Timedelta(days=1)
                if date > end_date:
                    date = original_date
                    while date not in data.index:
                        date -= pd.Timedelta(days=1)
                        if date < actual_start_date:
                            return None
                    return date
            return date
        for year in range(end_date.year - years_back, end_date.year):
            if pd.to_datetime(f"{year}-05-01") < actual_start_date:
                continue
            may_start = get_closest_date(f"{year}-05-01")
            oct_end = get_closest_date(f"{year}-10-01")
            if may_start is None or oct_end is None:
                continue
            try:
                may_start_price = data.loc[may_start]["Price"]
                oct_end_price = data.loc[oct_end]["Price"]
                may_sep_return = (oct_end_price - may_start_price) / may_start_price * 100
            except KeyError:
                continue
            oct_start = oct_end
            may_end = get_closest_date(f"{year + 1}-05-01")
            if oct_start is None or may_end is None:
                continue
            try:
                oct_start_price = data.loc[oct_start]["Price"]
                may_end_price = data.loc[may_end]["Price"]
                oct_apr_return = (may_end_price - oct_start_price) / oct_start_price * 100
            except KeyError:
                continue
            valid_years.append(year)
            may_sep_returns.append(may_sep_return)
            oct_apr_returns.append(oct_apr_return)
        if not valid_years or len(valid_years) < 2:
            return None
        returns_df = pd.DataFrame({
            "Year": valid_years,
            "May-Sep (%)": may_sep_returns,
            "Oct-Apr (%)": oct_apr_returns
        })
        returns_df["Buy-Hold (%)"] = returns_df["May-Sep (%)"] + returns_df["Oct-Apr (%)"]
        returns_df["Sell-in-May (%)"] = returns_df["Oct-Apr (%)"]
        sell_may_portfolio = initial_investment
        buy_hold_portfolio = initial_investment
        yearly_results = []
        for _, row in returns_df.iterrows():
            buy_hold_year_end = buy_hold_portfolio * (1 + row["Buy-Hold (%)"] / 100)
            sell_may_year_end = sell_may_portfolio * (1 + row["Sell-in-May (%)"] / 100)
            buy_hold_yearly_return = (buy_hold_year_end - buy_hold_portfolio) / buy_hold_portfolio * 100
            sell_may_yearly_return = (sell_may_year_end - sell_may_portfolio) / sell_may_portfolio * 100
            yearly_results.append({
                "Year": row["Year"],
                "Buy-Hold Value ($)": buy_hold_year_end,
                "Sell-in-May Value ($)": sell_may_year_end,
                "Buy-Hold Return (%)": buy_hold_yearly_return,
                "Sell-in-May Return (%)": sell_may_yearly_return
            })
            buy_hold_portfolio = buy_hold_year_end
            sell_may_portfolio = sell_may_year_end
        yearly_results_df = pd.DataFrame(yearly_results)
        total_years = len(valid_years)
        buy_hold_annual_return = (buy_hold_portfolio / initial_investment) ** (1 / total_years) - 1
        sell_may_annual_return = (sell_may_portfolio / initial_investment) ** (1 / total_years) - 1
        return {
            "symbol": symbol,
            "yearly_results_df": yearly_results_df,
            "buy_hold_final": buy_hold_portfolio,
            "sell_may_final": sell_may_portfolio,
            "buy_hold_annual": buy_hold_annual_return * 100,
            "sell_may_annual": sell_may_annual_return * 100,
            "years": f"{min(valid_years)}-{max(valid_years)}",
            "total_years": total_years
        }

    def summarize(self, results, data=None):
        if 'summary' not in results or not results['summary']:
            print("\n===== Sell in May Strategy =====")
            print("No valid backtest results found.")
            return
            
        # Only plot one equity curve for the strategy
        x_dates = results['trade_dates']
        equity = results['portfolio_curve']
        bh_equity = results['buy_hold_curve']
        
        # Get the primary symbol's results
        symbol = results['symbol']
        res = results['summary'][symbol]
        
        # Print results
        n_trades = len(x_dates) - 1
        avg_gain = (res['sell_may_annual'] / res['total_years']) if 'total_years' in res and res['total_years'] > 0 else 0
        
        # For yearly data, we need to manually calculate CAGR
        equity_arr = np.array(equity)
        bh_equity_arr = np.array(bh_equity)
        
        # Calculate annual returns ourselves since we have yearly data
        annual_return = res['sell_may_annual']
        bh_annual_return = res['buy_hold_annual']
        
        # Calculate max drawdown
        max_dd = calculate_max_drawdown(equity_arr)
        bh_max_dd = calculate_max_drawdown(bh_equity_arr)
        
        # Print standardized results
        print(f"\n===== {results['strategy_name']} =====")
        print(f"Trades: {n_trades}")
        print(f"Average gain per year: {avg_gain:.2f}%")
        print(f"CAGR (approx): {annual_return:.2f}%")
        print(f"Max drawdown: {max_dd:.2f}%")
        print(f"Final equity: ${equity_arr[-1]:.2f}")
        print(f"Buy & Hold CAGR: {bh_annual_return:.2f}%")
        print(f"Buy & Hold Max Drawdown: {bh_max_dd:.2f}%")
        print(f"Buy & Hold Final Equity: ${bh_equity_arr[-1]:.2f}")
        
        # Provide additional information for Sell in May strategy
        print(f"\n----- {symbol} ADDITIONAL DETAILS -----")
        print(f"Time Period: {res['years']} ({res['total_years']} years)")
        print(f"Strategy Annual Return: {res['sell_may_annual']:.2f}%")
        print(f"Buy & Hold Annual Return: {res['buy_hold_annual']:.2f}%")
        winner = "Buy & Hold" if res["buy_hold_annual"] > res["sell_may_annual"] else "Sell in May"
        print(f"Winner: {winner}")
        
        # Plot the equity curve
        plot_equity_curve(x_dates, equity, bh_equity, results['strategy_name'], bh_label="Buy & Hold")

if __name__ == "__main__":
    from core.backtest_utils import download_data
    symbol = "SPY"
    years_back = 20
    end_date = pd.Timestamp.now()
    start_date = end_date - pd.Timedelta(days=365 * (years_back + 1))
    data = download_data(symbol, start_date, end_date)
    strategy = SellMayStrategy()
    results = strategy.run(data, 10000)
    strategy.summarize(results, data)