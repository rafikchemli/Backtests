from abc import ABC, abstractmethod

class StrategyBase(ABC):
    @abstractmethod
    def run(self, data, initial_investment, **kwargs):
        """
        Run the strategy logic. Should return a dict with at least:
            - 'trade_dates': list of dates
            - 'portfolio_curve': list of portfolio values
            - 'strategy_name': str
            - any other relevant metrics/results
        """
        pass

    @abstractmethod
    def summarize(self, results, data=None):
        """
        Print and/or return summary statistics and results for the strategy.
        """
        pass 