"""
Data preparation for SMPC protocol
Handles data loading, share generation, and results processing
"""
import numpy as np
import pandas as pd
import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

# Import configuration
from config import NUM_PARTIES, SHARE_MAGNITUDE, RESULTS_DIR


class DataManager:
    """Handles all data operations for SMPC protocol"""
    
    def __init__(self, country_name: str, party_id: int, data_dir: Optional[str] = None, 
                 logger: Optional[logging.Logger] = None):
        self.country_name = country_name
        self.party_id = party_id
        self.data_dir = Path(data_dir or country_name)
        self.logger = logger or logging.getLogger(country_name)
        self.data_column = 'daily_cases'
        self._data_cache = {}
    
    def get_all_dates(self) -> List[str]:
        """Get all unique dates from the dataset"""
        try:
            csv_path = self.data_dir / f"{self.data_column}.csv"
            if not csv_path.exists():
                self.logger.error(f"Data file not found: {csv_path}")
                return []
            
            df = pd.read_csv(csv_path)
            if 'Date' not in df.columns:
                self.logger.error("'Date' column not found in CSV file")
                return []
            
            dates = sorted(df['Date'].unique())
            self.logger.info(f"Found {len(dates)} unique dates")
            return dates
            
        except Exception as e:
            self.logger.error(f"Error reading dates: {e}")
            return []
    
    def get_data_for_date(self, date: str) -> float:
        """Get data value for a specific date"""
        if date in self._data_cache:
            return self._data_cache[date]
        
        try:
            csv_path = self.data_dir / f"{self.data_column}.csv"
            df = pd.read_csv(csv_path)
            
            # Filter for the specific date
            date_data = df[df['Date'] == date]
            if date_data.empty:
                value = 0.0
            else:
                value = float(date_data[self.data_column].iloc[0])
            
            self._data_cache[date] = value
            return value
            
        except Exception as e:
            self.logger.error(f"Error getting data for date {date}: {e}")
            return 0.0
    
    def generate_shares(self, value: float, num_parties: int = NUM_PARTIES) -> List[float]:
        """Generate secret shares with enhanced privacy"""
        # Generate random shares for all parties
        shares = np.random.uniform(-SHARE_MAGNITUDE, SHARE_MAGNITUDE, num_parties)
        
        # Adjust shares to sum to original value
        adjustment = (value - np.sum(shares)) / num_parties
        shares = shares + adjustment
        
        # Verify sum (for debugging)
        share_sum = np.sum(shares)
        if abs(share_sum - value) > 1e-6:
            self.logger.warning(f"Share generation precision issue: {share_sum} vs {value}")
        
        return shares.tolist()
    
    def create_shares_dataframe(self, dates: List[str], values: Dict[str, float]) -> pd.DataFrame:
        """Create DataFrame with shares for all dates and parties"""
        shares_df = pd.DataFrame(index=dates, columns=range(NUM_PARTIES), dtype=float)
        
        for date in dates:
            value = values.get(date, 0.0)
            shares = self.generate_shares(value)
            shares_df.loc[date] = shares
        
        self.logger.info(f"Created shares DataFrame: {shares_df.shape}")
        return shares_df
    
    def save_results(self, results_df: pd.DataFrame, filename: str = "smpc_results.csv"):
        """Save final results to CSV file"""
        os.makedirs(RESULTS_DIR, exist_ok=True)
        filepath = Path(RESULTS_DIR) / filename
        
        try:
            results_df.to_csv(filepath, index=False)
            self.logger.info(f"Results saved to {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving results: {e}")
            return False