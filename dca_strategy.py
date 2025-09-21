import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

class DynamicDCAStrategy:
    """
    Dynamic Dollar Cost Averaging Strategy
    
    Formula:
    - 偏離度 = (價格 - MA(D)) / MA(D)
    - 單位數 n = 偏離度 / x
    - 定投率 = 1 - n × y
    - 投入金額 = 基礎金額 × 定投率
    """
    
    def __init__(self, ma_period: int = 200, x: float = 0.03, y: float = 0.3, base_amount: float = 1000):
        """
        Initialize DCA strategy
        
        Args:
            ma_period: Moving average period (default 200 for MA200)
            x: Deviation threshold (default 3.0% = 0.03)
            y: Adjustment factor (default 0.3)
            base_amount: Base investment amount in USD (default 1000)
        """
        self.ma_period = ma_period
        self.x = x
        self.y = y
        self.base_amount = base_amount
        
    def calculate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate DCA signals based on price and moving average
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with additional columns for strategy signals
        """
        result = df.copy()
        
        result[f'ma_{self.ma_period}'] = result['close'].rolling(window=self.ma_period).mean()
        
        result['deviation'] = (result['close'] - result[f'ma_{self.ma_period}']) / result[f'ma_{self.ma_period}']
        
        result['unit_n'] = result['deviation'] / self.x
        
        result['investment_rate'] = 1 - result['unit_n'] * self.y
        
        result['investment_amount'] = self.base_amount * result['investment_rate']
        
        result['action'] = np.where(result['investment_amount'] > 0, 'buy', 'sell')
        result['action'] = np.where(result['investment_amount'] == 0, 'hold', result['action'])
        
        result['abs_investment_amount'] = result['investment_amount'].abs()
        
        return result
    
    def backtest(self, df: pd.DataFrame, initial_cash: float = 10000) -> Dict:
        """
        Backtest the DCA strategy
        
        Args:
            df: DataFrame with OHLCV data
            initial_cash: Initial cash amount
            
        Returns:
            Dictionary with backtest results
        """
        signals_df = self.calculate_signals(df)
        
        cash = initial_cash
        btc_holdings = 0
        portfolio_values = []
        trades = []
        
        for idx, row in signals_df.iterrows():
            if pd.isna(row[f'ma_{self.ma_period}']):
                portfolio_value = cash + btc_holdings * row['close']
                portfolio_values.append({
                    'timestamp': idx,
                    'cash': cash,
                    'btc_holdings': btc_holdings,
                    'btc_value': btc_holdings * row['close'],
                    'portfolio_value': portfolio_value,
                    'price': row['close']
                })
                continue
                
            investment_amount = row['investment_amount']
            price = row['close']
            
            if investment_amount > 0:  # Buy
                if cash >= investment_amount:
                    btc_bought = investment_amount / price
                    cash -= investment_amount
                    btc_holdings += btc_bought
                    
                    trades.append({
                        'timestamp': idx,
                        'action': 'buy',
                        'amount_usd': investment_amount,
                        'amount_btc': btc_bought,
                        'price': price
                    })
                    
            elif investment_amount < 0:  # Sell
                sell_amount_usd = abs(investment_amount)
                btc_to_sell = sell_amount_usd / price
                
                if btc_holdings >= btc_to_sell:
                    cash += sell_amount_usd
                    btc_holdings -= btc_to_sell
                    
                    trades.append({
                        'timestamp': idx,
                        'action': 'sell',
                        'amount_usd': sell_amount_usd,
                        'amount_btc': btc_to_sell,
                        'price': price
                    })
            
            portfolio_value = cash + btc_holdings * price
            portfolio_values.append({
                'timestamp': idx,
                'cash': cash,
                'btc_holdings': btc_holdings,
                'btc_value': btc_holdings * price,
                'portfolio_value': portfolio_value,
                'price': price
            })
        
        portfolio_df = pd.DataFrame(portfolio_values)
        portfolio_df.set_index('timestamp', inplace=True)
        
        trades_df = pd.DataFrame(trades)
        if not trades_df.empty:
            trades_df.set_index('timestamp', inplace=True)
        
        performance = self._calculate_performance(portfolio_df, initial_cash)
        
        return {
            'portfolio': portfolio_df,
            'trades': trades_df,
            'signals': signals_df,
            'performance': performance,
            'parameters': {
                'ma_period': self.ma_period,
                'x': self.x,
                'y': self.y,
                'base_amount': self.base_amount
            }
        }
    
    def _calculate_performance(self, portfolio_df: pd.DataFrame, initial_cash: float) -> Dict:
        """Calculate performance metrics"""
        if portfolio_df.empty:
            return {}
            
        final_value = portfolio_df['portfolio_value'].iloc[-1]
        total_return = (final_value - initial_cash) / initial_cash
        
        portfolio_df['returns'] = portfolio_df['portfolio_value'].pct_change()
        returns = portfolio_df['returns'].dropna()
        
        if len(returns) > 1 and returns.std() > 0:
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(len(returns) * 365.25 / len(portfolio_df))
        else:
            sharpe_ratio = 0
        
        rolling_max = portfolio_df['portfolio_value'].expanding().max()
        drawdown = (portfolio_df['portfolio_value'] - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        if len(returns) > 1:
            volatility = returns.std() * np.sqrt(len(returns) * 365.25 / len(portfolio_df))
        else:
            volatility = 0
        
        start_price = portfolio_df['price'].iloc[0]
        end_price = portfolio_df['price'].iloc[-1]
        buy_hold_return = (end_price - start_price) / start_price
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'volatility': volatility,
            'final_value': final_value,
            'buy_hold_return': buy_hold_return,
            'excess_return': total_return - buy_hold_return
        }

def main():
    """Test the DCA strategy"""
    from data_fetcher import OKXDataFetcher
    
    fetcher = OKXDataFetcher()
    df = fetcher.load_data("BTC_USDT_4H.csv")
    
    if df.empty:
        print("No data found. Fetching new data...")
        df = fetcher.fetch_historical_data(days=180)
        if not df.empty:
            fetcher.save_data(df, "BTC_USDT_4H.csv")
    
    if df.empty:
        print("No data available for testing")
        return
    
    strategy = DynamicDCAStrategy(ma_period=200, x=0.03, y=0.3, base_amount=1000)
    results = strategy.backtest(df)
    
    print("Strategy Performance:")
    for key, value in results['performance'].items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    
    print(f"\nTotal trades: {len(results['trades'])}")
    if not results['trades'].empty:
        print("Recent trades:")
        print(results['trades'].tail())

if __name__ == "__main__":
    main()
