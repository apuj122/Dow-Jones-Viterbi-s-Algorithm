"""
Data fetcher module for downloading Dow Jones historical data.
"""

import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime

class DowJonesDataFetcher:
    """Fetches Dow Jones index data from Yahoo Finance."""
    
    def __init__(self, output_dir: str = "data"):
        """
        Initialize the data fetcher.
        
        Args:
            output_dir: Directory to save downloaded data
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.ticker = "^DJI"  # Dow Jones Industrial Average
        
    def fetch_data(self, start_date: str = "1995-01-01", 
                   end_date: str = None) -> pd.DataFrame:
        """
        Fetch Dow Jones data from Yahoo Finance.
        
        Args:
            start_date: Start date in format 'YYYY-MM-DD' (default: 1995-01-01)
            end_date: End date in format 'YYYY-MM-DD' (default: today)
            
        Returns:
            DataFrame with OHLCV data
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        print(f"Fetching {self.ticker} data from {start_date} to {end_date}...")
        
        try:
            data = yf.download(self.ticker, start=start_date, end=end_date, progress=False)
            
            # Handle MultiIndex columns from newer yfinance versions
            if isinstance(data.columns, pd.MultiIndex):
                # Drop the ticker level if MultiIndex
                data.columns = data.columns.droplevel(0)
            
            # Standardize column names
            data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            data['Adj Close'] = data['Close']
            
            print(f"Successfully downloaded {len(data)} trading days of data")
            return data
        except Exception as e:
            print(f"Error fetching data: {e}")
            raise
    
    def save_data(self, data: pd.DataFrame, filename: str = "dow_jones_data.csv"):
        """
        Save downloaded data to CSV.
        
        Args:
            data: DataFrame to save
            filename: Output filename
        """
        filepath = self.output_dir / filename
        data.to_csv(filepath)
        print(f"Data saved to {filepath}")
        return filepath
    
    def load_data(self, filename: str = "dow_jones_data.csv") -> pd.DataFrame:
        """
        Load previously saved data.
        
        Args:
            filename: Input filename
            
        Returns:
            DataFrame with loaded data
        """
        filepath = self.output_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        data = pd.read_csv(filepath, index_col=0, parse_dates=True, date_format='%Y-%m-%d')
        print(f"Loaded data from {filepath}")
        return data
    
    def get_or_fetch(self, start_date: str = "1995-01-01", 
                     end_date: str = None,
                     filename: str = "dow_jones_data.csv") -> pd.DataFrame:
        """
        Load cached data if available, otherwise fetch and save.
        
        Args:
            start_date: Start date for fetching
            end_date: End date for fetching
            filename: Filename to check/save
            
        Returns:
            DataFrame with Dow Jones data
        """
        filepath = self.output_dir / filename
        
        if filepath.exists():
            print(f"Loading cached data from {filepath}")
            return self.load_data(filename)
        else:
            data = self.fetch_data(start_date, end_date)
            self.save_data(data, filename)
            return data
