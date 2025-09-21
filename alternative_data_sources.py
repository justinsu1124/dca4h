"""
Alternative data sources for Bitcoin historical data when OKX API is limited.
This module provides fallback options for fetching multi-year Bitcoin 4H data.
"""

import pandas as pd
import numpy as np
import requests
import logging
from datetime import datetime, timedelta
from typing import Optional
import time

logger = logging.getLogger(__name__)

class AlternativeDataFetcher:
    """Fetch Bitcoin data from alternative sources when OKX is limited"""
    
    def __init__(self):
        self.session = requests.Session()
        
    def fetch_from_coingecko(self, days: int = 1460) -> pd.DataFrame:
        """
        Fetch Bitcoin data from CoinGecko API (free tier allows historical data)
        Note: CoinGecko provides daily data, we'll need to simulate 4H intervals
        """
        logger.info(f"Attempting to fetch {days} days of Bitcoin data from CoinGecko...")
        
        url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
        params = {
            'vs_currency': 'usd',
            'days': str(days),
            'interval': 'daily'
        }
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            prices = data['prices']
            
            df = pd.DataFrame(prices, columns=['timestamp', 'close'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            df_4h = self._simulate_4h_from_daily(df)
            
            logger.info(f"Fetched {len(df_4h)} simulated 4H candles from CoinGecko")
            return df_4h
            
        except Exception as e:
            logger.error(f"Error fetching from CoinGecko: {e}")
            return pd.DataFrame()
    
    def _simulate_4h_from_daily(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        """
        Simulate 4H candles from daily data by creating 6 intervals per day
        This is a simplified approach for demonstration purposes
        """
        simulated_data = []
        
        for i in range(len(daily_df) - 1):
            current_price = daily_df['close'].iloc[i]
            next_price = daily_df['close'].iloc[i + 1]
            current_date = daily_df.index[i]
            
            for interval in range(6):
                timestamp = current_date + timedelta(hours=4 * interval)
                
                progress = interval / 6.0
                interpolated_price = current_price + (next_price - current_price) * progress
                
                volatility = 0.02
                noise = (np.random.random() - 0.5) * volatility
                
                open_price = interpolated_price * (1 + noise)
                high_price = interpolated_price * (1 + abs(noise) + 0.005)
                low_price = interpolated_price * (1 - abs(noise) - 0.005)
                close_price = interpolated_price
                volume = 1000 + np.random.random() * 500  # Simulated volume
                
                simulated_data.append({
                    'timestamp': timestamp,
                    'open': open_price,
                    'high': high_price,
                    'low': low_price,
                    'close': close_price,
                    'volume': volume
                })
        
        result_df = pd.DataFrame(simulated_data)
        result_df.set_index('timestamp', inplace=True)
        result_df = result_df.sort_index()
        
        return result_df
    
    def create_synthetic_multi_year_data(self, start_year: int = 2022, end_year: int = 2025) -> pd.DataFrame:
        """
        Create synthetic Bitcoin 4H data for demonstration purposes
        This uses realistic price movements based on known Bitcoin trends
        """
        logger.info(f"Creating synthetic Bitcoin data for {start_year}-{end_year}")
        
        yearly_prices = {
            2022: {'start': 47000, 'end': 16500, 'high': 48000, 'low': 15500},  # Bear market
            2023: {'start': 16500, 'end': 42000, 'high': 44000, 'low': 15000},  # Recovery
            2024: {'start': 42000, 'end': 95000, 'high': 100000, 'low': 38000}, # Bull run
            2025: {'start': 95000, 'end': 110000, 'high': 125000, 'low': 75000} # Current (partial)
        }
        
        all_data = []
        
        for year in range(start_year, end_year + 1):
            if year not in yearly_prices:
                continue
                
            year_info = yearly_prices[year]
            start_date = datetime(year, 1, 1)
            
            if year == 2025:
                end_date = datetime(2025, 9, 21)  # Current date
            else:
                end_date = datetime(year + 1, 1, 1)
            
            current_date = start_date
            candles_in_year = []
            
            days_in_period = (end_date - start_date).days
            
            while current_date < end_date:
                progress = (current_date - start_date).days / days_in_period
                
                base_price = year_info['start'] + (year_info['end'] - year_info['start']) * progress
                
                daily_volatility = 0.03
                trend_noise = (np.random.random() - 0.5) * daily_volatility
                
                close_price = base_price * (1 + trend_noise)
                open_price = close_price * (1 + (np.random.random() - 0.5) * 0.01)
                high_price = max(open_price, close_price) * (1 + np.random.random() * 0.02)
                low_price = min(open_price, close_price) * (1 - np.random.random() * 0.02)
                volume = 800 + np.random.random() * 400
                
                candles_in_year.append({
                    'timestamp': current_date,
                    'open': open_price,
                    'high': high_price,
                    'low': low_price,
                    'close': close_price,
                    'volume': volume
                })
                
                current_date += timedelta(hours=4)
            
            all_data.extend(candles_in_year)
            logger.info(f"Generated {len(candles_in_year)} candles for {year}")
        
        result_df = pd.DataFrame(all_data)
        result_df.set_index('timestamp', inplace=True)
        result_df = result_df.sort_index()
        
        logger.info(f"Created synthetic dataset with {len(result_df)} total candles")
        logger.info(f"Date range: {result_df.index.min()} to {result_df.index.max()}")
        logger.info(f"Price range: ${result_df['close'].min():.2f} to ${result_df['close'].max():.2f}")
        
        return result_df

def main():
    """Test alternative data sources"""
    fetcher = AlternativeDataFetcher()
    
    synthetic_df = fetcher.create_synthetic_multi_year_data(2022, 2025)
    
    if not synthetic_df.empty:
        from data_fetcher import OKXDataFetcher
        okx_fetcher = OKXDataFetcher()
        okx_fetcher.save_data(synthetic_df, "BTC_USDT_4H_synthetic_multi_year.csv")
        print("Synthetic multi-year data created and saved!")
    
    return synthetic_df

if __name__ == "__main__":
    main()
