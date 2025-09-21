from data_fetcher import OKXDataFetcher
from datetime import datetime, timedelta
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_historical_data_limits():
    """Test how far back OKX API can fetch historical data"""
    fetcher = OKXDataFetcher()
    
    test_dates = [
        datetime(2024, 1, 1),  # 2024 start
        datetime(2023, 1, 1),  # 2023 start  
        datetime(2022, 1, 1),  # 2022 start
        datetime(2021, 1, 1),  # 2021 start (to test limits)
    ]
    
    for test_date in test_dates:
        before_ts = str(int(test_date.timestamp() * 1000))
        
        print(f"\nTesting data availability before {test_date.strftime('%Y-%m-%d')}...")
        logger.info(f"Testing before timestamp: {before_ts}")
        
        df = fetcher.fetch_candles(
            inst_id="BTC-USDT",
            bar="4H", 
            before=before_ts,
            limit=100  # Small sample to test availability
        )
        
        if not df.empty:
            earliest = df.index.min()
            latest = df.index.max()
            print(f"✓ Data available: {len(df)} candles from {earliest} to {latest}")
            print(f"  Price range: ${df['close'].min():.2f} to ${df['close'].max():.2f}")
        else:
            print(f"✗ No data available before {test_date}")

def fetch_multi_year_data(years=[2022, 2023, 2024, 2025]):
    """Attempt to fetch multi-year data using pagination"""
    fetcher = OKXDataFetcher()
    all_data = []
    
    start_year = min(years)
    end_year = max(years)
    
    total_days = (datetime(end_year + 1, 1, 1) - datetime(start_year, 1, 1)).days
    total_candles_needed = total_days * 6  # 6 candles per day for 4H
    requests_needed = (total_candles_needed + 299) // 300
    
    print(f"Attempting to fetch {len(years)} years of data ({start_year}-{end_year})")
    print(f"Estimated candles needed: {total_candles_needed}")
    print(f"Estimated requests needed: {requests_needed}")
    
    after = None
    max_requests = 50  # Safety limit
    
    for i in range(min(requests_needed, max_requests)):
        df = fetcher.fetch_candles(
            inst_id="BTC-USDT",
            bar="4H",
            after=after,
            limit=300
        )
        
        if df.empty:
            print(f"No more data available after {i} requests")
            break
            
        all_data.append(df)
        
        earliest_date = df.index.min()
        if earliest_date.year <= start_year:
            print(f"Reached target year {start_year} at request {i+1}")
            break
            
        after = str(int(df.index.min().timestamp() * 1000))
        
        if (i + 1) % 10 == 0:
            print(f"Progress: {i + 1} requests completed, earliest date: {earliest_date}")
    
    if not all_data:
        return pd.DataFrame()
        
    combined_df = pd.concat(all_data, axis=0)
    combined_df = combined_df.drop_duplicates().sort_index()
    
    start_date = datetime(start_year, 1, 1)
    end_date = datetime(end_year + 1, 1, 1)
    combined_df = combined_df[(combined_df.index >= start_date) & (combined_df.index < end_date)]
    
    print(f"\nFinal multi-year dataset:")
    print(f"Total candles: {len(combined_df)}")
    print(f"Date range: {combined_df.index.min()} to {combined_df.index.max()}")
    print(f"Price range: ${combined_df['close'].min():.2f} to ${combined_df['close'].max():.2f}")
    
    return combined_df

if __name__ == "__main__":
    print("Testing OKX API historical data limits...")
    test_historical_data_limits()
    
    print("\n" + "="*60)
    print("Attempting to fetch multi-year data...")
    multi_year_df = fetch_multi_year_data([2022, 2023, 2024, 2025])
    
    if not multi_year_df.empty:
        fetcher = OKXDataFetcher()
        fetcher.save_data(multi_year_df, "BTC_USDT_4H_multi_year.csv")
        print("Multi-year data saved successfully!")
    else:
        print("Failed to fetch multi-year data")
