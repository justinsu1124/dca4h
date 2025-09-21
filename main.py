import argparse
import logging
import sys
from pathlib import Path
from data_fetcher import OKXDataFetcher
from dca_strategy import DynamicDCAStrategy
from optimizer import ParameterOptimizer
from analyzer import PerformanceAnalyzer
from multi_year_analyzer import MultiYearAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dca_backtest.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def fetch_data(days: int = 180):
    """Fetch Bitcoin 4H data from OKX"""
    logger.info(f"Fetching {days} days of Bitcoin 4H data from OKX...")
    
    fetcher = OKXDataFetcher()
    df = fetcher.fetch_historical_data(days=days)
    
    if not df.empty:
        fetcher.save_data(df, "BTC_USDT_4H.csv")
        logger.info(f"Successfully fetched {len(df)} candles")
        print(f"Data range: {df.index.min()} to {df.index.max()}")
        print(f"Price range: ${df['close'].min():.2f} to ${df['close'].max():.2f}")
    else:
        logger.error("Failed to fetch data")
        return False
    
    return True

def run_backtest(x: float = 0.03, y: float = 0.3, base_amount: float = 1000, 
                ma_period: int = 200, initial_cash: float = 10000):
    """Run backtest with specified parameters"""
    logger.info(f"Running backtest with parameters: x={x}, y={y}, base_amount={base_amount}")
    
    fetcher = OKXDataFetcher()
    df = fetcher.load_data("BTC_USDT_4H.csv")
    
    if df.empty:
        logger.error("No data found. Please run 'fetch' command first")
        return None
    
    strategy = DynamicDCAStrategy(ma_period=ma_period, x=x, y=y, base_amount=base_amount)
    results = strategy.backtest(df, initial_cash)
    
    analyzer = PerformanceAnalyzer()
    
    report = analyzer.generate_performance_report(results)
    print(report)
    
    analyzer.save_results(results)
    
    return results

def optimize_parameters(method: str = "grid", initial_cash: float = 10000):
    """Optimize strategy parameters"""
    logger.info(f"Starting parameter optimization using {method} method...")
    
    fetcher = OKXDataFetcher()
    df = fetcher.load_data("BTC_USDT_4H.csv")
    
    if df.empty:
        logger.error("No data found. Please run 'fetch' command first")
        return None
    
    optimizer = ParameterOptimizer(df, initial_cash)
    
    if method == "grid":
        results_df = optimizer.grid_search(
            x_range=(0.01, 0.08, 8),      # 1% to 8% deviation threshold
            y_range=(0.1, 0.8, 8),        # 0.1 to 0.8 adjustment factor
            base_amount_range=(500, 2000, 4)  # $500 to $2000 base amount
        )
        
        print("\nTop 10 parameter combinations:")
        print(results_df.head(10).to_string(index=False))
        
        sensitivity = optimizer.analyze_parameter_sensitivity(results_df)
        print("\nParameter sensitivity analysis (top 10% performers):")
        for param in ['x', 'y', 'base_amount']:
            stats = sensitivity[f'{param}_stats']
            print(f"{param}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, "
                  f"range=[{stats['min']:.4f}, {stats['max']:.4f}]")
        
        best_params = results_df.iloc[0]
        print(f"\nRunning backtest with best parameters:")
        print(f"x={best_params['x']:.4f}, y={best_params['y']:.4f}, base_amount={best_params['base_amount']:.2f}")
        
        strategy = DynamicDCAStrategy(
            ma_period=200, 
            x=best_params['x'], 
            y=best_params['y'], 
            base_amount=best_params['base_amount']
        )
        best_results = strategy.backtest(df, initial_cash)
        
        analyzer = PerformanceAnalyzer()
        report = analyzer.generate_performance_report(best_results, results_df)
        print("\n" + report)
        
        analyzer.save_results(best_results, results_df)
        
        return best_results, results_df
        
    elif method == "differential":
        opt_result = optimizer.differential_evolution_optimize()
        
        best_params = opt_result['best_params']
        print(f"\nBest parameters found:")
        print(f"x={best_params['x']:.4f}, y={best_params['y']:.4f}, base_amount={best_params['base_amount']:.2f}")
        
        best_results = opt_result['backtest_results']
        optimization_history = opt_result['optimization_history']
        
        analyzer = PerformanceAnalyzer()
        report = analyzer.generate_performance_report(best_results, optimization_history)
        print("\n" + report)
        
        analyzer.save_results(best_results, optimization_history)
        
        return best_results, optimization_history
    
    else:
        logger.error(f"Unknown optimization method: {method}")

def fetch_multi_year_data():
    """Fetch multi-year historical data using pagination or alternative sources"""
    logger.info("Attempting to fetch multi-year historical data...")
    
    try:
        from test_historical_limits import fetch_multi_year_data as fetch_func
        
        multi_year_df = fetch_func([2022, 2023, 2024, 2025])
        
        if not multi_year_df.empty and len(multi_year_df) > 2000:  # Check if we got substantial data
            fetcher = OKXDataFetcher()
            fetcher.save_data(multi_year_df, "BTC_USDT_4H_multi_year.csv")
            logger.info("Multi-year data fetched from OKX and saved successfully!")
            
            print(f"Successfully fetched {len(multi_year_df)} candles from OKX")
            print(f"Date range: {multi_year_df.index.min()} to {multi_year_df.index.max()}")
            print(f"Price range: ${multi_year_df['close'].min():.2f} to ${multi_year_df['close'].max():.2f}")
            return True
        else:
            logger.warning("OKX API limited to recent data. Creating synthetic multi-year data for demonstration...")
            print("OKX API is limited to ~240 days of recent data.")
            print("Creating synthetic Bitcoin 4H data for 2022-2025 demonstration...")
            
            from alternative_data_sources import AlternativeDataFetcher
            alt_fetcher = AlternativeDataFetcher()
            synthetic_df = alt_fetcher.create_synthetic_multi_year_data(2022, 2025)
            
            if not synthetic_df.empty:
                fetcher = OKXDataFetcher()
                fetcher.save_data(synthetic_df, "BTC_USDT_4H_multi_year.csv")
                logger.info("Synthetic multi-year data created and saved!")
                
                print(f"Created synthetic dataset with {len(synthetic_df)} candles")
                print(f"Date range: {synthetic_df.index.min()} to {synthetic_df.index.max()}")
                print(f"Price range: ${synthetic_df['close'].min():.2f} to ${synthetic_df['close'].max():.2f}")
                print("\nNote: This is synthetic data for demonstration purposes.")
                print("For production use, consider alternative data sources like:")
                print("- CoinGecko API (daily data)")
                print("- Binance API (if available)")
                print("- Yahoo Finance")
                print("- Paid data providers")
                return True
            else:
                logger.error("Failed to create synthetic data")
                return False
            
    except Exception as e:
        logger.error(f"Error fetching multi-year data: {e}")
        print(f"Error fetching multi-year data: {e}")
        return False

def run_multi_year_analysis(years: list, optimize: bool = True, method: str = 'grid'):
    """Run multi-year analysis"""
    logger.info(f"Starting multi-year analysis for years: {years}")
    
    fetcher = OKXDataFetcher()
    
    df = fetcher.load_data("BTC_USDT_4H_multi_year.csv")
    if df.empty:
        df = fetcher.load_data("BTC_USDT_4H.csv")
    
    if df.empty:
        print("No data found. Please run 'python main.py fetch --multi-year' first")
        return False
    
    analyzer = MultiYearAnalyzer(df)
    
    print(f"Loaded data with {len(df)} candles")
    print(f"Years available: {sorted(analyzer.yearly_data.keys())}")
    
    print("\nAnalyzing individual years...")
    yearly_results = analyzer.analyze_individual_years(x=0.03, y=0.3, base_amount=1000)
    
    if optimize:
        print(f"\nOptimizing parameters using {method} method...")
        combined_results = analyzer.optimize_combined_parameters(method)
        
        best_params = combined_results['best_params']
        print(f"\nRe-analyzing individual years with optimal parameters...")
        yearly_results = analyzer.analyze_individual_years(
            x=best_params['x'],
            y=best_params['y'], 
            base_amount=best_params['base_amount']
        )
    
    comparison_df = analyzer.compare_yearly_performance()
    print("\nYear-by-year performance comparison:")
    print(comparison_df.to_string(index=False))
    
    report_path = analyzer.generate_multi_year_report()
    print(f"\nDetailed report saved to: {report_path}")
    
    analyzer.create_multi_year_visualizations()
    print("Multi-year visualizations created in results/ directory")
    
    if optimize and 'combined_results' in locals():
        import os
        os.makedirs('results', exist_ok=True)
        
        optimization_df = combined_results['optimization_history']
        optimization_df.to_csv('results/multi_year_optimization_results.csv', index=False)
        
        portfolio_df = combined_results['backtest_results']['portfolio']
        trades_df = combined_results['backtest_results']['trades']
        
        portfolio_df.to_csv('results/multi_year_portfolio_history.csv')
        if not trades_df.empty:
            trades_df.to_csv('results/multi_year_trades_history.csv')
        
        print("Optimization results and trading data saved to results/ directory")
    
    return True

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Dynamic DCA Strategy Backtesting System")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    fetch_parser = subparsers.add_parser('fetch', help='Fetch Bitcoin 4H data from OKX')
    fetch_parser.add_argument('--days', type=int, default=180, help='Number of days to fetch (default: 180)')
    fetch_parser.add_argument('--multi-year', action='store_true', help='Attempt to fetch multi-year data (2022-2025)')
    
    backtest_parser = subparsers.add_parser('backtest', help='Run backtest with specified parameters')
    backtest_parser.add_argument('--x', type=float, default=0.03, help='Deviation threshold (default: 0.03)')
    backtest_parser.add_argument('--y', type=float, default=0.3, help='Adjustment factor (default: 0.3)')
    backtest_parser.add_argument('--base-amount', type=float, default=1000, help='Base amount in USD (default: 1000)')
    backtest_parser.add_argument('--ma-period', type=int, default=200, help='Moving average period (default: 200)')
    backtest_parser.add_argument('--initial-cash', type=float, default=10000, help='Initial cash (default: 10000)')
    
    optimize_parser = subparsers.add_parser('optimize', help='Optimize strategy parameters')
    optimize_parser.add_argument('--method', choices=['grid', 'differential'], default='grid', 
                                help='Optimization method (default: grid)')
    optimize_parser.add_argument('--initial-cash', type=float, default=10000, help='Initial cash (default: 10000)')
    
    multi_year_parser = subparsers.add_parser('multi-year', help='Run multi-year analysis (2022-2025)')
    multi_year_parser.add_argument('--years', nargs='+', type=int, default=[2022, 2023, 2024, 2025], help='Years to analyze')
    multi_year_parser.add_argument('--optimize', action='store_true', help='Run parameter optimization on combined data')
    multi_year_parser.add_argument('--method', choices=['grid', 'differential'], default='grid', help='Optimization method')
    
    example_parser = subparsers.add_parser('example', help='Run example from user description')
    
    args = parser.parse_args()
    
    if args.command == 'fetch':
        if args.multi_year:
            success = fetch_multi_year_data()
        else:
            success = fetch_data(args.days)
        if not success:
            sys.exit(1)
            
    elif args.command == 'backtest':
        results = run_backtest(
            x=args.x, 
            y=args.y, 
            base_amount=args.base_amount,
            ma_period=args.ma_period,
            initial_cash=args.initial_cash
        )
        if results is None:
            sys.exit(1)
            
    elif args.command == 'optimize':
        results = optimize_parameters(method=args.method, initial_cash=args.initial_cash)
        if results is None:
            sys.exit(1)
            
    elif args.command == 'multi-year':
        success = run_multi_year_analysis(args.years, args.optimize, args.method)
        if not success:
            sys.exit(1)
            
    elif args.command == 'example':
        print("Running examples from user description...")
        
        print("\n" + "="*50)
        print("EXAMPLE 1: Price 6% below MA200")
        print("Parameters: x=3.0%, y=0.3, base=1000")
        print("Expected: deviation=-6%, n=-2, rate=1.6, amount=1600 (buy)")
        print("="*50)
        
        print("\n" + "="*50)
        print("EXAMPLE 2: Price 15% above MA200")
        print("Parameters: x=3.0%, y=0.3, base=1000")
        print("Expected: deviation=15%, n=5, rate=-0.5, amount=-500 (sell)")
        print("="*50)
        
        results = run_backtest(x=0.03, y=0.3, base_amount=1000)
        if results is None:
            sys.exit(1)
            
    else:
        parser.print_help()
        print("\nExample usage:")
        print("  python main.py fetch --days 180")
        print("  python main.py fetch --multi-year")
        print("  python main.py backtest --x 0.03 --y 0.3 --base-amount 1000")
        print("  python main.py optimize --method grid")
        print("  python main.py multi-year --optimize --method grid")
        print("  python main.py example")

if __name__ == "__main__":
    main()
