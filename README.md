# Dynamic DCA Strategy Backtesting System

A comprehensive backtesting system for a dynamic Dollar Cost Averaging (DCA) strategy using Bitcoin 4-hour data from OKX API.

## Strategy Overview

The dynamic DCA strategy adjusts investment amounts based on price deviation from a moving average:

### Mathematical Formula
```
偏離度 = (價格 - MA(D)) / MA(D)
單位數 n = 偏離度 / x
定投率 = 1 - n × y  
投入金額 = 基礎金額 × 定投率
```

### Strategy Logic
- **Price below MA**: Negative deviation → Investment rate > 1 → Buy more
- **Price near MA**: Zero deviation → Investment rate ≈ 1 → Normal investment
- **Price above MA**: Positive deviation → Investment rate < 0 → Sell

### Parameters
- **x**: Deviation threshold (e.g., 0.03 = 3%)
- **y**: Adjustment factor (e.g., 0.3)
- **base_amount**: Base investment amount in USD
- **MA period**: Moving average period (default: 200 for 4H data)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/justinsu1124/dca4h.git
cd dca4h
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Fetch Data
Fetch Bitcoin 4-hour data from OKX API:
```bash
python main.py fetch --days 180
```

### 2. Run Backtest
Test the strategy with specific parameters:
```bash
python main.py backtest --x 0.03 --y 0.3 --base-amount 1000
```

### 3. Optimize Parameters
Find optimal parameters using grid search:
```bash
python main.py optimize --method grid
```

Or use differential evolution:
```bash
python main.py optimize --method differential
```

### 4. Run Examples
Test the examples from the strategy description:
```bash
python main.py example
```

## Command Line Options

### Fetch Command
- `--days`: Number of days to fetch (default: 180)

### Backtest Command
- `--x`: Deviation threshold (default: 0.03)
- `--y`: Adjustment factor (default: 0.3)
- `--base-amount`: Base investment amount (default: 1000)
- `--ma-period`: Moving average period (default: 200)
- `--initial-cash`: Initial cash amount (default: 10000)

### Optimize Command
- `--method`: Optimization method ('grid' or 'differential', default: 'grid')
- `--initial-cash`: Initial cash amount (default: 10000)

## Output Files

Results are saved to the `results/` directory:
- `performance_report.txt`: Detailed performance analysis
- `portfolio_history.csv`: Portfolio value over time
- `trades_history.csv`: All trades executed
- `optimization_results.csv`: Parameter optimization results
- `portfolio_performance.png`: Portfolio performance charts
- `trading_activity.png`: Trading signals and activity
- `parameter_optimization.png`: Parameter optimization visualization

## Strategy Examples

### Example 1: Price 6% Below MA
- Deviation: -6%
- Unit number (n): -6% / 3% = -2
- Investment rate: 1 - (-2 × 0.3) = 1.6
- Investment amount: 1000 × 1.6 = $1600 (buy more)

### Example 2: Price 15% Above MA
- Deviation: 15%
- Unit number (n): 15% / 3% = 5
- Investment rate: 1 - (5 × 0.3) = -0.5
- Investment amount: 1000 × (-0.5) = -$500 (sell)

## Performance Metrics

The system calculates comprehensive performance metrics:
- **Total Return**: Overall portfolio return
- **Sharpe Ratio**: Risk-adjusted return
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Volatility**: Annualized portfolio volatility
- **Excess Return**: Return vs buy-and-hold strategy

## Architecture

### Core Components
- `data_fetcher.py`: OKX API integration for Bitcoin data
- `dca_strategy.py`: Dynamic DCA strategy implementation
- `optimizer.py`: Parameter optimization using grid search and differential evolution
- `analyzer.py`: Performance analysis and visualization
- `main.py`: Command-line interface

### Data Flow
1. **Data Fetching**: Retrieve 4H Bitcoin data from OKX API
2. **Strategy Execution**: Apply DCA formula to generate trading signals
3. **Backtesting**: Simulate portfolio performance over historical data
4. **Optimization**: Find optimal parameters using various methods
5. **Analysis**: Generate comprehensive performance reports and visualizations

## API Rate Limits

The OKX API has rate limits of 40 requests per 2 seconds. The data fetcher automatically handles rate limiting to ensure compliance.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Disclaimer

This software is for educational and research purposes only. Past performance does not guarantee future results. Always do your own research before making investment decisions.
