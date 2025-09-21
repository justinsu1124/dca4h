import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
from typing import Dict, List, Tuple
from dca_strategy import DynamicDCAStrategy
from optimizer import ParameterOptimizer
from analyzer import PerformanceAnalyzer

logger = logging.getLogger(__name__)

class MultiYearAnalyzer:
    """Analyze DCA strategy performance across multiple years"""
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize multi-year analyzer
        
        Args:
            data: Historical price data spanning multiple years
        """
        self.data = data
        self.yearly_data = {}
        self.yearly_results = {}
        self.combined_results = None
        
        self._split_data_by_year()
        
    def _split_data_by_year(self):
        """Split data by calendar years"""
        if self.data.empty:
            return
            
        years = sorted(self.data.index.year.unique())
        logger.info(f"Splitting data by years: {years}")
        
        for year in years:
            year_data = self.data[self.data.index.year == year]
            if len(year_data) > 0:
                self.yearly_data[year] = year_data
                logger.info(f"Year {year}: {len(year_data)} candles from {year_data.index.min()} to {year_data.index.max()}")
    
    def analyze_individual_years(self, x: float = 0.03, y: float = 0.3, base_amount: float = 1000, 
                               initial_cash: float = 10000) -> Dict:
        """
        Analyze strategy performance for each individual year
        
        Args:
            x: Deviation threshold
            y: Adjustment factor
            base_amount: Base investment amount
            initial_cash: Initial cash for backtesting
            
        Returns:
            Dictionary with results for each year
        """
        logger.info(f"Analyzing individual years with parameters: x={x}, y={y}, base_amount={base_amount}")
        
        for year, year_data in self.yearly_data.items():
            if len(year_data) < 200:  # Need enough data for MA200
                logger.warning(f"Insufficient data for year {year}: {len(year_data)} candles")
                continue
                
            strategy = DynamicDCAStrategy(ma_period=200, x=x, y=y, base_amount=base_amount)
            results = strategy.backtest(year_data, initial_cash)
            
            self.yearly_results[year] = results
            
            performance = results['performance']
            logger.info(f"Year {year} performance: Return={performance.get('total_return', 0):.2%}, "
                       f"Sharpe={performance.get('sharpe_ratio', 0):.3f}, "
                       f"Drawdown={performance.get('max_drawdown', 0):.2%}")
        
        return self.yearly_results
    
    def optimize_combined_parameters(self, optimization_method: str = 'grid') -> Dict:
        """
        Optimize parameters using the entire multi-year dataset
        
        Args:
            optimization_method: 'grid' or 'differential_evolution'
            
        Returns:
            Optimization results with best parameters
        """
        logger.info(f"Optimizing parameters using {optimization_method} method on combined dataset")
        
        optimizer = ParameterOptimizer(self.data, initial_cash=10000)
        
        if optimization_method == 'grid':
            results_df = optimizer.grid_search(
                x_range=(0.02, 0.06, 10),
                y_range=(0.3, 0.8, 10), 
                base_amount_range=(500, 2000, 4)
            )
            
            best_params = results_df.iloc[0]
            
            strategy = DynamicDCAStrategy(
                ma_period=200,
                x=best_params['x'],
                y=best_params['y'], 
                base_amount=best_params['base_amount']
            )
            backtest_results = strategy.backtest(self.data, 10000)
            
            self.combined_results = {
                'best_params': {
                    'x': best_params['x'],
                    'y': best_params['y'],
                    'base_amount': best_params['base_amount']
                },
                'best_score': -best_params['score'],
                'backtest_results': backtest_results,
                'optimization_history': results_df
            }
            
        else:  # differential_evolution
            self.combined_results = optimizer.differential_evolution_optimize()
        
        best_performance = self.combined_results['backtest_results']['performance']
        logger.info(f"Combined optimization results: "
                   f"x={self.combined_results['best_params']['x']:.4f}, "
                   f"y={self.combined_results['best_params']['y']:.4f}, "
                   f"base_amount={self.combined_results['best_params']['base_amount']:.0f}")
        logger.info(f"Combined performance: Return={best_performance.get('total_return', 0):.2%}, "
                   f"Sharpe={best_performance.get('sharpe_ratio', 0):.3f}")
        
        return self.combined_results
    
    def compare_yearly_performance(self) -> pd.DataFrame:
        """
        Create comparison table of yearly performance metrics
        
        Returns:
            DataFrame with performance comparison
        """
        if not self.yearly_results:
            return pd.DataFrame()
        
        comparison_data = []
        
        for year, results in self.yearly_results.items():
            performance = results['performance']
            trades = results['trades']
            
            year_data = {
                'Year': year,
                'Total_Return': performance.get('total_return', 0),
                'Sharpe_Ratio': performance.get('sharpe_ratio', 0),
                'Max_Drawdown': performance.get('max_drawdown', 0),
                'Volatility': performance.get('volatility', 0),
                'Buy_Hold_Return': performance.get('buy_hold_return', 0),
                'Excess_Return': performance.get('excess_return', 0),
                'Total_Trades': len(trades),
                'Buy_Trades': len(trades[trades['action'] == 'buy']) if not trades.empty else 0,
                'Sell_Trades': len(trades[trades['action'] == 'sell']) if not trades.empty else 0,
                'Final_Value': performance.get('final_value', 0)
            }
            comparison_data.append(year_data)
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Year')
        
        return comparison_df
    
    def generate_multi_year_report(self, output_dir: str = 'results') -> str:
        """
        Generate comprehensive multi-year analysis report
        
        Args:
            output_dir: Directory to save the report
            
        Returns:
            Path to the generated report file
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        report_path = os.path.join(output_dir, 'multi_year_analysis_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("MULTI-YEAR DYNAMIC DCA STRATEGY ANALYSIS REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write("DATASET OVERVIEW:\n")
            f.write("-"*40 + "\n")
            f.write(f"Total Period: {self.data.index.min()} to {self.data.index.max()}\n")
            f.write(f"Total Candles: {len(self.data)}\n")
            f.write(f"Years Analyzed: {sorted(self.yearly_data.keys())}\n")
            f.write(f"Price Range: ${self.data['close'].min():.2f} to ${self.data['close'].max():.2f}\n\n")
            
            if self.combined_results:
                f.write("COMBINED OPTIMIZATION RESULTS:\n")
                f.write("-"*40 + "\n")
                best_params = self.combined_results['best_params']
                f.write(f"Optimal Parameters:\n")
                f.write(f"  Deviation Threshold (x): {best_params['x']:.4f} ({best_params['x']*100:.2f}%)\n")
                f.write(f"  Adjustment Factor (y): {best_params['y']:.3f}\n")
                f.write(f"  Base Amount: ${best_params['base_amount']:.2f}\n\n")
                
                combined_performance = self.combined_results['backtest_results']['performance']
                f.write(f"Combined Performance Metrics:\n")
                f.write(f"  Total Return: {combined_performance.get('total_return', 0):.2%}\n")
                f.write(f"  Sharpe Ratio: {combined_performance.get('sharpe_ratio', 0):.4f}\n")
                f.write(f"  Maximum Drawdown: {combined_performance.get('max_drawdown', 0):.2%}\n")
                f.write(f"  Volatility: {combined_performance.get('volatility', 0):.2%}\n")
                f.write(f"  Buy & Hold Return: {combined_performance.get('buy_hold_return', 0):.2%}\n")
                f.write(f"  Excess Return: {combined_performance.get('excess_return', 0):.2%}\n\n")
            
            if self.yearly_results:
                f.write("YEAR-BY-YEAR PERFORMANCE COMPARISON:\n")
                f.write("-"*40 + "\n")
                comparison_df = self.compare_yearly_performance()
                
                f.write(f"{'Year':<6} {'Return':<8} {'Sharpe':<8} {'Drawdown':<10} {'Trades':<8} {'Excess':<8}\n")
                f.write("-" * 60 + "\n")
                
                for _, row in comparison_df.iterrows():
                    f.write(f"{row['Year']:<6} {row['Total_Return']:<8.2%} {row['Sharpe_Ratio']:<8.3f} "
                           f"{row['Max_Drawdown']:<10.2%} {row['Total_Trades']:<8} {row['Excess_Return']:<8.2%}\n")
                
                f.write("\n")
                
                f.write("SUMMARY STATISTICS:\n")
                f.write("-"*20 + "\n")
                f.write(f"Average Annual Return: {comparison_df['Total_Return'].mean():.2%}\n")
                f.write(f"Average Sharpe Ratio: {comparison_df['Sharpe_Ratio'].mean():.3f}\n")
                f.write(f"Best Year: {comparison_df.loc[comparison_df['Total_Return'].idxmax(), 'Year']} "
                       f"({comparison_df['Total_Return'].max():.2%})\n")
                f.write(f"Worst Year: {comparison_df.loc[comparison_df['Total_Return'].idxmin(), 'Year']} "
                       f"({comparison_df['Total_Return'].min():.2%})\n")
                f.write(f"Total Trades Across All Years: {comparison_df['Total_Trades'].sum()}\n")
        
        logger.info(f"Multi-year analysis report saved to {report_path}")
        return report_path
    
    def create_multi_year_visualizations(self, output_dir: str = 'results'):
        """Create visualizations for multi-year analysis"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        if self.yearly_results:
            comparison_df = self.compare_yearly_performance()
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Multi-Year DCA Strategy Performance Analysis', fontsize=16)
            
            axes[0, 0].bar(comparison_df['Year'], comparison_df['Total_Return'] * 100)
            axes[0, 0].set_title('Annual Returns (%)')
            axes[0, 0].set_ylabel('Return (%)')
            axes[0, 0].grid(True, alpha=0.3)
            
            axes[0, 1].bar(comparison_df['Year'], comparison_df['Sharpe_Ratio'])
            axes[0, 1].set_title('Annual Sharpe Ratios')
            axes[0, 1].set_ylabel('Sharpe Ratio')
            axes[0, 1].grid(True, alpha=0.3)
            
            axes[1, 0].bar(comparison_df['Year'], comparison_df['Max_Drawdown'] * 100)
            axes[1, 0].set_title('Maximum Drawdowns (%)')
            axes[1, 0].set_ylabel('Drawdown (%)')
            axes[1, 0].grid(True, alpha=0.3)
            
            axes[1, 1].bar(comparison_df['Year'], comparison_df['Total_Trades'])
            axes[1, 1].set_title('Trading Activity (Number of Trades)')
            axes[1, 1].set_ylabel('Number of Trades')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            yearly_comparison_path = os.path.join(output_dir, 'yearly_performance_comparison.png')
            plt.savefig(yearly_comparison_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Yearly comparison chart saved to {yearly_comparison_path}")
        
        if self.combined_results:
            portfolio_df = self.combined_results['backtest_results']['portfolio']
            
            fig, axes = plt.subplots(2, 1, figsize=(15, 10))
            fig.suptitle('Multi-Year Portfolio Performance', fontsize=16)
            
            axes[0].plot(portfolio_df.index, portfolio_df['portfolio_value'], linewidth=2, label='Strategy')
            
            initial_value = portfolio_df['portfolio_value'].iloc[0]
            price_ratio = portfolio_df['price'] / portfolio_df['price'].iloc[0]
            buy_hold_values = initial_value * price_ratio
            axes[0].plot(portfolio_df.index, buy_hold_values, linewidth=2, alpha=0.7, label='Buy & Hold')
            
            axes[0].set_title('Portfolio Value Over Time')
            axes[0].set_ylabel('Portfolio Value ($)')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            rolling_max = portfolio_df['portfolio_value'].expanding().max()
            drawdown = (portfolio_df['portfolio_value'] - rolling_max) / rolling_max * 100
            axes[1].fill_between(portfolio_df.index, drawdown, 0, alpha=0.3, color='red')
            axes[1].plot(portfolio_df.index, drawdown, color='red', linewidth=1)
            axes[1].set_title('Portfolio Drawdown Over Time')
            axes[1].set_ylabel('Drawdown (%)')
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            portfolio_path = os.path.join(output_dir, 'multi_year_portfolio_performance.png')
            plt.savefig(portfolio_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Multi-year portfolio chart saved to {portfolio_path}")

def main():
    """Test the multi-year analyzer"""
    from data_fetcher import OKXDataFetcher
    
    fetcher = OKXDataFetcher()
    
    multi_year_df = fetcher.load_data("BTC_USDT_4H_multi_year.csv")
    
    if multi_year_df.empty:
        multi_year_df = fetcher.load_data("BTC_USDT_4H.csv")
    
    if multi_year_df.empty:
        print("No data available for testing")
        return
    
    analyzer = MultiYearAnalyzer(multi_year_df)
    
    print(f"Loaded data with {len(multi_year_df)} candles")
    print(f"Years available: {sorted(analyzer.yearly_data.keys())}")
    
    yearly_results = analyzer.analyze_individual_years(x=0.03, y=0.3, base_amount=1000)
    
    combined_results = analyzer.optimize_combined_parameters('grid')
    
    comparison_df = analyzer.compare_yearly_performance()
    print("\nYear-by-year comparison:")
    print(comparison_df)
    
    report_path = analyzer.generate_multi_year_report()
    print(f"\nReport generated: {report_path}")
    
    analyzer.create_multi_year_visualizations()
    print("Visualizations created")

if __name__ == "__main__":
    main()
