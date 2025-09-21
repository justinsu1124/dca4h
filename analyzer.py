import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class PerformanceAnalyzer:
    """Analyze and visualize DCA strategy performance"""
    
    def __init__(self, results_dir: str = "results"):
        """
        Initialize analyzer
        
        Args:
            results_dir: Directory to save analysis results
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def plot_portfolio_performance(self, backtest_results: Dict, save_path: Optional[str] = None):
        """
        Plot portfolio value over time
        
        Args:
            backtest_results: Results from DCA strategy backtest
            save_path: Path to save the plot
        """
        portfolio_df = backtest_results['portfolio']
        performance = backtest_results['performance']
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
        
        ax1_twin = ax1.twinx()
        
        ax1.plot(portfolio_df.index, portfolio_df['portfolio_value'], 
                label='Portfolio Value', color='blue', linewidth=2)
        ax1_twin.plot(portfolio_df.index, portfolio_df['price'], 
                     label='BTC Price', color='orange', alpha=0.7)
        
        ax1.set_ylabel('Portfolio Value ($)', color='blue')
        ax1_twin.set_ylabel('BTC Price ($)', color='orange')
        ax1.set_title('Portfolio Performance vs BTC Price')
        ax1.grid(True, alpha=0.3)
        
        metrics_text = f"Total Return: {performance['total_return']:.2%}\n"
        metrics_text += f"Sharpe Ratio: {performance['sharpe_ratio']:.3f}\n"
        metrics_text += f"Max Drawdown: {performance['max_drawdown']:.2%}\n"
        metrics_text += f"vs Buy & Hold: {performance['excess_return']:.2%}"
        
        ax1.text(0.02, 0.98, metrics_text, transform=ax1.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax2.plot(portfolio_df.index, portfolio_df['cash'], label='Cash', color='green')
        ax2.plot(portfolio_df.index, portfolio_df['btc_value'], label='BTC Value', color='orange')
        ax2.set_ylabel('Value ($)')
        ax2.set_title('Portfolio Composition')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        rolling_max = portfolio_df['portfolio_value'].expanding().max()
        drawdown = (portfolio_df['portfolio_value'] - rolling_max) / rolling_max * 100
        
        ax3.fill_between(portfolio_df.index, drawdown, 0, alpha=0.3, color='red')
        ax3.plot(portfolio_df.index, drawdown, color='red', linewidth=1)
        ax3.set_ylabel('Drawdown (%)')
        ax3.set_title('Portfolio Drawdown')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Portfolio performance plot saved to {save_path}")
        
        plt.show()
    
    def plot_trading_activity(self, backtest_results: Dict, save_path: Optional[str] = None):
        """
        Plot trading activity and signals
        
        Args:
            backtest_results: Results from DCA strategy backtest
            save_path: Path to save the plot
        """
        signals_df = backtest_results['signals']
        trades_df = backtest_results['trades']
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
        
        ax1.plot(signals_df.index, signals_df['close'], label='BTC Price', color='blue', alpha=0.7)
        ax1.plot(signals_df.index, signals_df['ma_200'], label='MA200', color='red', linewidth=2)
        
        if not trades_df.empty:
            buy_trades = trades_df[trades_df['action'] == 'buy']
            sell_trades = trades_df[trades_df['action'] == 'sell']
            
            if not buy_trades.empty:
                ax1.scatter(buy_trades.index, buy_trades['price'], 
                           color='green', marker='^', s=50, label='Buy', alpha=0.7)
            if not sell_trades.empty:
                ax1.scatter(sell_trades.index, sell_trades['price'], 
                           color='red', marker='v', s=50, label='Sell', alpha=0.7)
        
        ax1.set_ylabel('Price ($)')
        ax1.set_title('Price, Moving Average, and Trading Signals')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(signals_df.index, signals_df['deviation'] * 100, 
                label='Deviation (%)', color='purple')
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.set_ylabel('Deviation (%)')
        ax2.set_title('Price Deviation from MA200')
        ax2.grid(True, alpha=0.3)
        
        colors = ['green' if x > 0 else 'red' for x in signals_df['investment_amount']]
        ax3.bar(signals_df.index, signals_df['investment_amount'], 
               color=colors, alpha=0.6, width=pd.Timedelta(hours=2))
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax3.set_ylabel('Investment Amount ($)')
        ax3.set_title('Investment Amounts (Positive=Buy, Negative=Sell)')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Trading activity plot saved to {save_path}")
        
        plt.show()
    
    def plot_parameter_optimization(self, optimization_results: pd.DataFrame, save_path: Optional[str] = None):
        """
        Plot parameter optimization results
        
        Args:
            optimization_results: DataFrame with optimization results
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        scatter_params = {
            'alpha': 0.6,
            'c': optimization_results['sharpe_ratio'],
            'cmap': 'viridis',
            's': 50
        }
        
        im1 = axes[0, 0].scatter(optimization_results['x'], optimization_results['y'], **scatter_params)
        axes[0, 0].set_xlabel('x (Deviation Threshold)')
        axes[0, 0].set_ylabel('y (Adjustment Factor)')
        axes[0, 0].set_title('Parameter Space (colored by Sharpe Ratio)')
        plt.colorbar(im1, ax=axes[0, 0])
        
        im2 = axes[0, 1].scatter(optimization_results['x'], optimization_results['base_amount'], **scatter_params)
        axes[0, 1].set_xlabel('x (Deviation Threshold)')
        axes[0, 1].set_ylabel('Base Amount ($)')
        axes[0, 1].set_title('x vs Base Amount')
        plt.colorbar(im2, ax=axes[0, 1])
        
        axes[1, 0].hist(optimization_results['sharpe_ratio'], bins=30, alpha=0.7, color='blue')
        axes[1, 0].set_xlabel('Sharpe Ratio')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Sharpe Ratio Distribution')
        axes[1, 0].grid(True, alpha=0.3)
        
        top_10 = optimization_results.head(10)
        y_pos = np.arange(len(top_10))
        
        axes[1, 1].barh(y_pos, top_10['sharpe_ratio'], alpha=0.7)
        axes[1, 1].set_yticks(y_pos)
        axes[1, 1].set_yticklabels([f"x={row['x']:.3f}, y={row['y']:.2f}" for _, row in top_10.iterrows()])
        axes[1, 1].set_xlabel('Sharpe Ratio')
        axes[1, 1].set_title('Top 10 Parameter Combinations')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Parameter optimization plot saved to {save_path}")
        
        plt.show()
    
    def generate_performance_report(self, backtest_results: Dict, optimization_results: Optional[pd.DataFrame] = None) -> str:
        """
        Generate a comprehensive performance report
        
        Args:
            backtest_results: Results from DCA strategy backtest
            optimization_results: Optional optimization results
            
        Returns:
            Formatted report string
        """
        performance = backtest_results['performance']
        parameters = backtest_results['parameters']
        portfolio_df = backtest_results['portfolio']
        trades_df = backtest_results['trades']
        
        report = "=" * 60 + "\n"
        report += "DYNAMIC DCA STRATEGY PERFORMANCE REPORT\n"
        report += "=" * 60 + "\n\n"
        
        report += "STRATEGY PARAMETERS:\n"
        report += f"  Moving Average Period: {parameters['ma_period']}\n"
        report += f"  Deviation Threshold (x): {parameters['x']:.3f} ({parameters['x']*100:.1f}%)\n"
        report += f"  Adjustment Factor (y): {parameters['y']:.3f}\n"
        report += f"  Base Amount: ${parameters['base_amount']:.2f}\n\n"
        
        report += "PERFORMANCE METRICS:\n"
        report += f"  Total Return: {performance['total_return']:.2%}\n"
        report += f"  Sharpe Ratio: {performance['sharpe_ratio']:.3f}\n"
        report += f"  Maximum Drawdown: {performance['max_drawdown']:.2%}\n"
        report += f"  Volatility (Annualized): {performance['volatility']:.2%}\n"
        report += f"  Final Portfolio Value: ${performance['final_value']:.2f}\n\n"
        
        report += "BENCHMARK COMPARISON:\n"
        report += f"  Buy & Hold Return: {performance['buy_hold_return']:.2%}\n"
        report += f"  Excess Return: {performance['excess_return']:.2%}\n\n"
        
        if not trades_df.empty:
            buy_trades = trades_df[trades_df['action'] == 'buy']
            sell_trades = trades_df[trades_df['action'] == 'sell']
            
            report += "TRADING STATISTICS:\n"
            report += f"  Total Trades: {len(trades_df)}\n"
            report += f"  Buy Trades: {len(buy_trades)}\n"
            report += f"  Sell Trades: {len(sell_trades)}\n"
            
            if not buy_trades.empty:
                report += f"  Average Buy Amount: ${buy_trades['amount_usd'].mean():.2f}\n"
            if not sell_trades.empty:
                report += f"  Average Sell Amount: ${sell_trades['amount_usd'].mean():.2f}\n"
            
            report += f"  Trading Period: {portfolio_df.index.min().strftime('%Y-%m-%d')} to {portfolio_df.index.max().strftime('%Y-%m-%d')}\n\n"
        
        if optimization_results is not None:
            report += "OPTIMIZATION SUMMARY:\n"
            report += f"  Total Combinations Tested: {len(optimization_results)}\n"
            report += f"  Best Sharpe Ratio: {optimization_results['sharpe_ratio'].max():.3f}\n"
            report += f"  Worst Sharpe Ratio: {optimization_results['sharpe_ratio'].min():.3f}\n"
            report += f"  Average Sharpe Ratio: {optimization_results['sharpe_ratio'].mean():.3f}\n\n"
        
        return report
    
    def save_results(self, backtest_results: Dict, optimization_results: Optional[pd.DataFrame] = None):
        """
        Save all analysis results to files
        
        Args:
            backtest_results: Results from DCA strategy backtest
            optimization_results: Optional optimization results
        """
        report = self.generate_performance_report(backtest_results, optimization_results)
        report_path = self.results_dir / "performance_report.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        logger.info(f"Performance report saved to {report_path}")
        
        portfolio_path = self.results_dir / "portfolio_history.csv"
        backtest_results['portfolio'].to_csv(portfolio_path)
        logger.info(f"Portfolio history saved to {portfolio_path}")
        
        if not backtest_results['trades'].empty:
            trades_path = self.results_dir / "trades_history.csv"
            backtest_results['trades'].to_csv(trades_path)
            logger.info(f"Trades history saved to {trades_path}")
        
        if optimization_results is not None:
            opt_path = self.results_dir / "optimization_results.csv"
            optimization_results.to_csv(opt_path, index=False)
            logger.info(f"Optimization results saved to {opt_path}")
        
        self.plot_portfolio_performance(backtest_results, self.results_dir / "portfolio_performance.png")
        self.plot_trading_activity(backtest_results, self.results_dir / "trading_activity.png")
        
        if optimization_results is not None:
            self.plot_parameter_optimization(optimization_results, self.results_dir / "parameter_optimization.png")

def main():
    """Test the analyzer"""
    from data_fetcher import OKXDataFetcher
    from dca_strategy import DynamicDCAStrategy
    
    fetcher = OKXDataFetcher()
    df = fetcher.load_data("BTC_USDT_4H.csv")
    
    if df.empty:
        print("No data found. Please run data_fetcher.py first")
        return
    
    strategy = DynamicDCAStrategy(ma_period=200, x=0.03, y=0.3, base_amount=1000)
    results = strategy.backtest(df)
    
    analyzer = PerformanceAnalyzer()
    
    report = analyzer.generate_performance_report(results)
    print(report)
    
    analyzer.plot_portfolio_performance(results)
    analyzer.plot_trading_activity(results)

if __name__ == "__main__":
    main()
