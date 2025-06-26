"""
SMPC Protocol implementation
Coordinates the secure multi-party computation protocol
"""
import argparse
import logging
import os
import sys
import threading
import time
import pandas as pd
from typing import Dict, List, Optional

from server import SMPCServer
from data_prep import DataManager
from config import (COUNTRIES, DEALER_ID, NUM_PARTIES, LOGS_DIR, 
                   SHARE_EXCHANGE_TIMEOUT, AGGREGATE_TIMEOUT)


class SMPCProtocol:
    """Main SMPC protocol coordinator"""
    
    def __init__(self, country_name: str, data_dir: Optional[str] = None):
        self.country_name = country_name
        self.party_id = COUNTRIES[country_name]
        self.is_dealer = (self.party_id == DEALER_ID)
        
        # Setup logging
        self._setup_logging()
        
        # Initialize components
        self.server = SMPCServer(country_name, self.party_id, self.logger)
        self.data_mgr = DataManager(country_name, self.party_id, data_dir, self.logger)
        
        self.logger.info(f"Initialized {country_name} protocol (Party {self.party_id})")
    
    def _setup_logging(self):
        """Configure logging for this party"""
        os.makedirs(LOGS_DIR, exist_ok=True)
        
        log_file = os.path.join(LOGS_DIR, f'{self.country_name}.log')
        logging.basicConfig(
            level=logging.INFO,
            format=f'[{self.country_name}] %(asctime)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(self.country_name)
    
    def run_protocol(self, limit_dates: Optional[int] = None):
        """Execute the complete SMPC protocol"""
        try:
            # Phase 1: Data preparation
            all_dates = self._prepare_data(limit_dates)
            if not all_dates:
                raise ValueError("No dates available for processing")
            
            # Phase 2: Share generation and distribution
            self._distribute_shares(all_dates)
            
            # Phase 3: Local computation
            local_sums = self._compute_local_sums(all_dates)
            
            # Phase 4: Final aggregation (dealer only)
            if self.is_dealer:
                self._aggregate_results(all_dates, local_sums)
            
            self.logger.info("SMPC Protocol completed successfully")
            
        except Exception as e:
            self.logger.error(f"Protocol failed: {e}")
            raise
    
    def _prepare_data(self, limit_dates: Optional[int]) -> List[str]:
        """Prepare data for the protocol"""
        all_dates = self.data_mgr.get_all_dates()
        
        if limit_dates and limit_dates > 0:
            all_dates = all_dates[:limit_dates]
            self.logger.info(f"Limited to {len(all_dates)} dates for testing")
        
        self.logger.info(f"Processing {len(all_dates)} dates")
        return all_dates
    
    def _distribute_shares(self, all_dates: List[str]):
        """Generate and distribute shares to all parties"""
        # Load our data
        my_values = {date: self.data_mgr.get_data_for_date(date) for date in all_dates}
        
        # Generate shares as DataFrame
        shares_df = self.data_mgr.create_shares_dataframe(all_dates, my_values)
        
        # Store our own shares
        for date in all_dates:
            if date not in self.server.shares_received:
                self.server.shares_received[date] = {}
            self.server.shares_received[date][self.party_id] = shares_df.loc[date, self.party_id]
        
        # Send shares to other parties
        self.logger.info("Distributing shares to all parties...")
        success = self.server.send_shares_dataframe(shares_df)
        
        if not success:
            raise RuntimeError("Failed to distribute shares to all parties")
        
        # Wait for shares from other parties
        self._wait_for_shares()
    
    def _wait_for_shares(self):
        """Wait for shares from all other parties"""
        expected_parties = NUM_PARTIES - 1
        start_time = time.time()
        
        while len(self.server.shares_received_from) < expected_parties:
            if time.time() - start_time > SHARE_EXCHANGE_TIMEOUT:
                missing = set(range(NUM_PARTIES)) - self.server.shares_received_from - {self.party_id}
                raise TimeoutError(f"Timeout waiting for shares from parties: {missing}")
            
            self.logger.info(f"Waiting for shares: {len(self.server.shares_received_from)}/{expected_parties}")
            time.sleep(5)
        
        self.logger.info("Received shares from all parties")
    
    def _compute_local_sums(self, all_dates: List[str]) -> Dict[str, float]:
        """Compute local sums from received shares"""
        local_sums = {}
        
        for date in all_dates:
            if date in self.server.shares_received:
                shares = self.server.shares_received[date]
                local_sums[date] = sum(float(share) for share in shares.values())
            else:
                local_sums[date] = 0.0
                self.logger.warning(f"No shares for date {date}")
        
        self.logger.info(f"Computed local sums for {len(local_sums)} dates")
        
        # Send to dealer
        if not self.is_dealer:
            success = self.server.send_aggregates(local_sums)
            if not success:
                raise RuntimeError("Failed to send aggregates to dealer")
        
        return local_sums
    
    def _aggregate_results(self, all_dates: List[str], local_sums: Dict[str, float]):
        """Aggregate final results (dealer only)"""
        self.logger.info("Waiting for aggregates from all parties...")
        
        # Wait for aggregates
        expected_parties = NUM_PARTIES - 1
        start_time = time.time()
        
        while len(self.server.aggregates_received_from) < expected_parties:
            if time.time() - start_time > AGGREGATE_TIMEOUT:
                missing = set(range(NUM_PARTIES)) - self.server.aggregates_received_from - {self.party_id}
                raise TimeoutError(f"Timeout waiting for aggregates from: {missing}")
            
            time.sleep(2)
        
        # Compute final results
        results = []
        for date in all_dates:
            if date in self.server.aggregate_sums:
                party_sums = self.server.aggregate_sums[date]
                party_sums[self.party_id] = local_sums[date]  # Add our sum
                
                final_sum = sum(party_sums.values())
                results.append({
                    'Date': date,
                    'Total_Cases': int(final_sum),
                    'Parties_Reporting': len(party_sums)
                })
        
        # Save results
        results_df = pd.DataFrame(results)
        self.data_mgr.save_results(results_df)
        self.logger.info(f"Final results computed and saved for {len(results)} dates")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='SMPC Protocol')
    parser.add_argument('--country', required=True, choices=list(COUNTRIES.keys()))
    parser.add_argument('--data-dir', help='Data directory')
    parser.add_argument('--limit', type=int, help='Limit number of dates')
    parser.add_argument('--column', default='daily_cases', help='Data column to use')
    
    args = parser.parse_args()
    
    try:
        protocol = SMPCProtocol(args.country, args.data_dir)
        protocol.data_mgr.data_column = args.column
        
        # Start server in background
        server_thread = threading.Thread(target=protocol.server.start_server, daemon=True)
        server_thread.start()
        time.sleep(3)  # Allow server to start
        
        # Run protocol
        protocol.run_protocol(args.limit)
        
        # Keep server running briefly for final communications
        time.sleep(10)
        protocol.server.stop_server()
        
    except KeyboardInterrupt:
        print("\nShutdown requested")
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()