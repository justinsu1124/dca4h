import requests
import pandas as pd
import numpy as np
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OKXDataFetcher:
    """OKX API data fetcher for Bitcoin 4-hour candlestick data"""
    
    def __init__(self):
        self.base_url = "https://www.okx.com"
        self.session = requests.Session()
        self.rate_limit_delay = 0.05  # 40 requests per 2 seconds = 0.05s between requests
        self.last_request_time = 0
        
        self.data_dir = Path('data')
        self.data_dir.mkdir(exist_ok=True)
        
    def _wait_for_rate_limit(self):
        """Wait to respect rate limits"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - time_since_last)
        self.last_request_time = time.time()
    
    def fetch_candles(self, inst_id: str = "BTC-USDT", bar: str = "4H", 
                     after: Optional[str] = None, before: Optional[str] = None, 
                     limit: int = 300) -> pd.DataFrame:
        """
        Fetch candlestick data from OKX API
        
        Args:
            inst_id: Instrument ID (e.g., BTC-USDT)
            bar: Bar size (4H for 4-hour)
            after: Pagination - return records earlier than this timestamp
            before: Pagination - return records newer than this timestamp  
            limit: Number of results (max 300)
        
        Returns:
            DataFrame with OHLCV data
        """
        self._wait_for_rate_limit()
        
        url = f"{self.base_url}/api/v5/market/candles"
        params = {
            'instId': inst_id,
            'bar': bar,
            'limit': str(limit)
        }
        
        if after:
            params['after'] = after
        if before:
            params['before'] = before
            
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            if data['code'] != '0':
                raise Exception(f"OKX API error: {data['msg']}")
                
            candles = data['data']
            if not candles:
                return pd.DataFrame()
                
            df = pd.DataFrame(candles, columns=['ts', 'open', 'high', 'low', 'close', 'volume', 'volCcy', 'volCcyQuote', 'confirm'])
            
            df['ts'] = pd.to_datetime(df['ts'].astype(int), unit='ms')
            df.set_index('ts', inplace=True)
            
            price_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in price_cols:
                df[col] = df[col].astype(float)
                
            df = df.sort_index()
            
            logger.info(f"Fetched {len(df)} candles for {inst_id}")
            return df[price_cols]  # Return only OHLCV columns
            
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            return pd.DataFrame()
    
    def fetch_historical_data(self, inst_id: str = "BTC-USDT", bar: str = "4H", 
                            days: int = 180) -> pd.DataFrame:
        """
        Fetch historical data for specified number of days
        
        Args:
            inst_id: Instrument ID
            bar: Bar size
            days: Number of days to fetch
            
        Returns:
            DataFrame with historical OHLCV data
        """
        all_data = []
        
        candles_per_day = 24 // 4  # 6 candles per day for 4H
        total_candles_needed = days * candles_per_day
        requests_needed = (total_candles_needed + 299) // 300  # Round up
        
        logger.info(f"Fetching {days} days of {bar} data for {inst_id} ({requests_needed} requests)")
        
        after = None
        for i in range(requests_needed):
            df = self.fetch_candles(inst_id=inst_id, bar=bar, after=after, limit=300)
            
            if df.empty:
                break
                
            all_data.append(df)
            
            after = str(int(df.index.min().timestamp() * 1000))
            
            if (i + 1) % 5 == 0:
                logger.info(f"Completed {i + 1}/{requests_needed} requests")
        
        if not all_data:
            return pd.DataFrame()
            
        combined_df = pd.concat(all_data, axis=0)
        combined_df = combined_df.drop_duplicates().sort_index()
        
        end_time = combined_df.index.max()
        start_time = end_time - timedelta(days=days)
        combined_df = combined_df[combined_df.index >= start_time]
        
        logger.info(f"Final dataset: {len(combined_df)} candles from {combined_df.index.min()} to {combined_df.index.max()}")
        return combined_df
    
    def save_data(self, df: pd.DataFrame, filename: str):
        """Save DataFrame to CSV"""
        filepath = self.data_dir / filename
        df.to_csv(filepath)
        logger.info(f"Saved data to {filepath}")
    
    def load_data(self, filename: str) -> pd.DataFrame:
        """Load DataFrame from CSV"""
        filepath = self.data_dir / filename
        if filepath.exists():
            df = pd.read_csv(filepath, index_col=0, parse_dates=True)
            logger.info(f"Loaded data from {filepath}")
            return df
        else:
            logger.warning(f"File {filepath} not found")
            return pd.DataFrame()

def main():
    """Test the data fetcher"""
    fetcher = OKXDataFetcher()
    
    df = fetcher.fetch_historical_data(days=30)
    
    if not df.empty:
        print(f"Data shape: {df.shape}")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        print(f"Price range: ${df['close'].min():.2f} to ${df['close'].max():.2f}")
        print("\nFirst 5 rows:")
        print(df.head())
        
        fetcher.save_data(df, "BTC_USDT_4H.csv")
    else:
        print("No data fetched")

if __name__ == "__main__":
    main()
