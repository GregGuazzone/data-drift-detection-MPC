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
from config import get_num_parties, SHARE_MAGNITUDE, RESULTS_DIR


class DataManager:
    """Handles all data operations for SMPC protocol"""
    
    def __init__(self, party_name: str, party_id: int, data_dir: Optional[str] = None, 
                 logger: Optional[logging.Logger] = None, data_root_dir: str = "."):
        self.party_name = party_name
        self.party_id = party_id
        self.data_root_dir = data_root_dir
        self.data_dir = Path(data_dir or party_name)
        self.logger = logger or logging.getLogger(party_name)
        self.data_column = 'daily_cases'
        self._data_cache = {}
        
        # Get dynamic number of parties
        self.num_parties = get_num_parties(data_root_dir)
    
    def get_all_dates(self) -> List[str]:
        """Get all unique dates from the dataset"""
        try:
            # Find any CSV file in the directory
            csv_files = list(self.data_dir.glob("*.csv"))
            if not csv_files:
                self.logger.error(f"No CSV files found in: {self.data_dir}")
                return []
            
            # Use the first CSV file found
            csv_path = csv_files[0]
            self.logger.info(f"Using data file: {csv_path}")
            
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
            # Find any CSV file in the directory
            csv_files = list(self.data_dir.glob("*.csv"))
            if not csv_files:
                self.logger.error(f"No CSV files found in: {self.data_dir}")
                return 0.0
            
            # Use the first CSV file found
            csv_path = csv_files[0]
            df = pd.read_csv(csv_path)
            
            # Check if the specified column exists
            if self.data_column not in df.columns:
                self.logger.error(f"Column '{self.data_column}' not found in {csv_path}")
                available_cols = [col for col in df.columns if col != 'Date']
                self.logger.info(f"Available columns: {available_cols}")
                return 0.0
            
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
    
    def generate_shares(self, value: float, num_parties: Optional[int] = None) -> List[float]:
        """Generate secret shares with enhanced privacy"""
        if num_parties is None:
            num_parties = self.num_parties
            
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
        shares_df = pd.DataFrame(index=dates, columns=range(self.num_parties), dtype=float)
        
        for date in dates:
            value = values.get(date, 0.0)
            shares = self.generate_shares(value)
            shares_df.loc[date] = shares
        
        self.logger.info(f"Created shares DataFrame: {shares_df.shape}")
        return shares_df
    
    def save_results(self, results_df: pd.DataFrame, filename: Optional[str] = None):
        """Save final results to CSV file in data root directory"""
        # Save in data root directory instead of global results directory
        results_dir = Path(self.data_root_dir)
        
        # Generate filename based on column name if not provided
        if filename is None:
            filename = f"Total_{self.data_column}.csv"
        
        filepath = results_dir / filename
        
        try:
            results_df.to_csv(filepath, index=False)
            self.logger.info(f"Results saved to {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving results: {e}")
            return False