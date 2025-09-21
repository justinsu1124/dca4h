import pandas as pd
import numpy as np
from scipy.optimize import differential_evolution
from itertools import product
import logging
from typing import Dict, List, Tuple, Optional
from dca_strategy import DynamicDCAStrategy

logger = logging.getLogger(__name__)

class ParameterOptimizer:
    """Optimize DCA strategy parameters to maximize Sharpe ratio"""
    
    def __init__(self, data: pd.DataFrame, initial_cash: float = 10000):
        """
        Initialize optimizer
        
        Args:
            data: Historical price data
            initial_cash: Initial cash for backtesting
        """
        self.data = data
        self.initial_cash = initial_cash
        self.optimization_results = []
        
    def objective_function(self, params: List[float]) -> float:
        """
        Objective function to minimize (negative Sharpe ratio with drawdown penalty)
        
        Args:
            params: [x, y, base_amount] parameters
            
        Returns:
            Negative score to minimize
        """
        x, y, base_amount = params
        
        try:
            strategy = DynamicDCAStrategy(ma_period=200, x=x, y=y, base_amount=base_amount)
            results = strategy.backtest(self.data, self.initial_cash)
            
            performance = results['performance']
            
            if not performance:
                return 1000  # High penalty for failed backtests
            
            sharpe_ratio = performance.get('sharpe_ratio', 0)
            max_drawdown = performance.get('max_drawdown', 0)
            total_return = performance.get('total_return', 0)
            
            drawdown_penalty = 0
            if max_drawdown < -0.5:
                drawdown_penalty = abs(max_drawdown) * 10
            
            return_penalty = 0
            if total_return < 0:
                return_penalty = abs(total_return) * 5
            
            score = -sharpe_ratio + drawdown_penalty + return_penalty
            
            self.optimization_results.append({
                'x': x,
                'y': y,
                'base_amount': base_amount,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'total_return': total_return,
                'score': score
            })
            
            return score
            
        except Exception as e:
            logger.error(f"Error in objective function: {e}")
            return 1000
    
    def grid_search(self, x_range: Tuple[float, float, int] = (0.01, 0.10, 10),
                   y_range: Tuple[float, float, int] = (0.1, 1.0, 10),
                   base_amount_range: Tuple[float, float, int] = (500, 2000, 4)) -> pd.DataFrame:
        """
        Grid search optimization
        
        Args:
            x_range: (min, max, steps) for x parameter
            y_range: (min, max, steps) for y parameter  
            base_amount_range: (min, max, steps) for base_amount parameter
            
        Returns:
            DataFrame with optimization results
        """
        logger.info("Starting grid search optimization...")
        
        x_values = np.linspace(x_range[0], x_range[1], x_range[2])
        y_values = np.linspace(y_range[0], y_range[1], y_range[2])
        base_amounts = np.linspace(base_amount_range[0], base_amount_range[1], base_amount_range[2])
        
        total_combinations = len(x_values) * len(y_values) * len(base_amounts)
        logger.info(f"Testing {total_combinations} parameter combinations")
        
        self.optimization_results = []
        
        for i, (x, y, base_amount) in enumerate(product(x_values, y_values, base_amounts)):
            if (i + 1) % 50 == 0:
                logger.info(f"Progress: {i + 1}/{total_combinations}")
                
            self.objective_function([x, y, base_amount])
        
        results_df = pd.DataFrame(self.optimization_results)
        results_df = results_df.sort_values('sharpe_ratio', ascending=False)
        
        logger.info("Grid search completed")
        return results_df
    
    def differential_evolution_optimize(self, bounds: List[Tuple[float, float]] = None) -> Dict:
        """
        Use differential evolution for optimization
        
        Args:
            bounds: Parameter bounds [(x_min, x_max), (y_min, y_max), (base_min, base_max)]
            
        Returns:
            Optimization result dictionary
        """
        if bounds is None:
            bounds = [(0.01, 0.10), (0.1, 1.0), (500, 2000)]
        
        logger.info("Starting differential evolution optimization...")
        
        self.optimization_results = []
        
        result = differential_evolution(
            self.objective_function,
            bounds,
            maxiter=100,
            popsize=15,
            seed=42
        )
        
        best_params = result.x
        best_score = result.fun
        
        logger.info(f"Optimization completed. Best score: {best_score:.4f}")
        
        strategy = DynamicDCAStrategy(
            ma_period=200, 
            x=best_params[0], 
            y=best_params[1], 
            base_amount=best_params[2]
        )
        best_results = strategy.backtest(self.data, self.initial_cash)
        
        return {
            'best_params': {
                'x': best_params[0],
                'y': best_params[1], 
                'base_amount': best_params[2]
            },
            'best_score': best_score,
            'backtest_results': best_results,
            'optimization_history': pd.DataFrame(self.optimization_results)
        }
    
    def analyze_parameter_sensitivity(self, results_df: pd.DataFrame) -> Dict:
        """
        Analyze parameter sensitivity from optimization results
        
        Args:
            results_df: DataFrame with optimization results
            
        Returns:
            Dictionary with sensitivity analysis
        """
        analysis = {}
        
        top_10_pct = int(len(results_df) * 0.1)
        top_results = results_df.head(top_10_pct)
        
        for param in ['x', 'y', 'base_amount']:
            analysis[f'{param}_stats'] = {
                'mean': top_results[param].mean(),
                'std': top_results[param].std(),
                'min': top_results[param].min(),
                'max': top_results[param].max(),
                'median': top_results[param].median()
            }
        
        corr_matrix = results_df[['x', 'y', 'base_amount', 'sharpe_ratio', 'max_drawdown', 'total_return']].corr()
        analysis['correlations'] = corr_matrix
        
        return analysis

def main():
    """Test the optimizer"""
    from data_fetcher import OKXDataFetcher
    
    fetcher = OKXDataFetcher()
    df = fetcher.load_data("BTC_USDT_4H.csv")
    
    if df.empty:
        print("No data found. Please run data_fetcher.py first")
        return
    
    test_data = df.tail(1000)  # Last 1000 candles
    
    optimizer = ParameterOptimizer(test_data)
    
    print("Running grid search optimization...")
    grid_results = optimizer.grid_search(
        x_range=(0.02, 0.06, 5),
        y_range=(0.2, 0.6, 5), 
        base_amount_range=(800, 1200, 3)
    )
    
    print("\nTop 5 parameter combinations:")
    print(grid_results.head())
    
    sensitivity = optimizer.analyze_parameter_sensitivity(grid_results)
    print("\nParameter sensitivity analysis:")
    for param in ['x', 'y', 'base_amount']:
        stats = sensitivity[f'{param}_stats']
        print(f"{param}: mean={stats['mean']:.4f}, std={stats['std']:.4f}")

if __name__ == "__main__":
    main()
