import argparse
import logging
import sys
from pathlib import Path
from data_fetcher import OKXDataFetcher
from dca_strategy import DynamicDCAStrategy
from optimizer import ParameterOptimizer
from analyzer import PerformanceAnalyzer

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
        return None

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Dynamic DCA Strategy Backtesting System")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    fetch_parser = subparsers.add_parser('fetch', help='Fetch Bitcoin 4H data from OKX')
    fetch_parser.add_argument('--days', type=int, default=180, help='Number of days to fetch (default: 180)')
    
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
    
    example_parser = subparsers.add_parser('example', help='Run example from user description')
    
    args = parser.parse_args()
    
    if args.command == 'fetch':
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
        print("  python main.py backtest --x 0.03 --y 0.3 --base-amount 1000")
        print("  python main.py optimize --method grid")
        print("  python main.py example")

if __name__ == "__main__":
    main()
