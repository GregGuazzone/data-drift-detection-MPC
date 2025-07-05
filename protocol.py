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
from pathlib import Path
from typing import Dict, List, Optional

from server import SMPCServer
from data_prep import DataManager
from config import (discover_parties, get_num_parties, COUNTRIES, DEALER_ID, LOGS_DIR, 
                   SHARE_EXCHANGE_TIMEOUT, AGGREGATE_TIMEOUT)


class SMPCProtocol:
    """Main SMPC protocol coordinator"""
    
    def __init__(self, party_name: str, party_id: Optional[int] = None, 
                 data_dir: Optional[str] = None, data_root_dir: str = "."):
        self.party_name = party_name
        self.data_root_dir = data_root_dir
        
        # Discover parties from directories
        self.parties = discover_parties(data_root_dir)
        self.num_parties = get_num_parties(data_root_dir)
        
        # Determine party ID from name or use provided ID
        if party_id is not None:
            self.party_id = party_id
        elif party_name in self.parties:
            self.party_id = self.parties[party_name]
        elif party_name in COUNTRIES:  # Legacy support
            self.party_id = COUNTRIES[party_name]
        else:
            available_parties = list(self.parties.keys())
            raise ValueError(f"Unknown party name: {party_name}. Available parties: {available_parties}")
        
        if self.party_id >= self.num_parties:
            raise ValueError(f"Party ID {self.party_id} exceeds number of parties {self.num_parties}")
        
        self.is_dealer = (self.party_id == DEALER_ID)
        
        # Setup logging
        self._setup_logging()
        
        # Initialize components
        # If no specific data_dir provided, construct it from data_root_dir and party_name
        if data_dir is None:
            data_dir = os.path.join(data_root_dir, party_name)
        
        self.server = SMPCServer(self.party_name, self.party_id, self.logger, data_root_dir)
        self.data_mgr = DataManager(self.party_name, self.party_id, data_dir, self.logger, data_root_dir)
        
        self.logger.info(f"Initialized {party_name} protocol (Party {self.party_id}/{self.num_parties})")
    
    def _setup_logging(self):
        """Configure logging for this party"""
        os.makedirs(LOGS_DIR, exist_ok=True)
        
        log_file = os.path.join(LOGS_DIR, f'{self.party_name}.log')
        logging.basicConfig(
            level=logging.INFO,
            format=f'[{self.party_name}] %(asctime)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(self.party_name)
    
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
        expected_parties = self.num_parties - 1
        start_time = time.time()
        
        while len(self.server.shares_received_from) < expected_parties:
            if time.time() - start_time > SHARE_EXCHANGE_TIMEOUT:
                missing = set(range(self.num_parties)) - self.server.shares_received_from - {self.party_id}
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
            
            # Wait a bit to ensure dealer receives the aggregates before we exit
            self.logger.info("Sent aggregates to dealer, waiting for protocol completion...")
            time.sleep(5)
        
        return local_sums
    
    def _aggregate_results(self, all_dates: List[str], local_sums: Dict[str, float]):
        """Aggregate final results (dealer only)"""
        self.logger.info("Waiting for aggregates from all parties...")
        
        # Wait for aggregates
        expected_parties = self.num_parties - 1
        start_time = time.time()
        
        while len(self.server.aggregates_received_from) < expected_parties:
            if time.time() - start_time > AGGREGATE_TIMEOUT:
                missing = set(range(self.num_parties)) - self.server.aggregates_received_from - {self.party_id}
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
                    f'Total_{self.data_mgr.data_column}': int(final_sum),
                    'Parties_Reporting': len(party_sums)
                })
        
        # Save results
        results_df = pd.DataFrame(results)
        self.data_mgr.save_results(results_df)
        self.logger.info(f"Final results computed and saved for {len(results)} dates")
        
        # Create completion marker to signal successful protocol completion
        completion_marker = Path(self.data_mgr.data_root_dir) / "protocol_complete.marker"
        with open(completion_marker, 'w') as f:
            f.write(f"Protocol completed successfully at {time.time()}\n")
            f.write(f"Results saved to: Total_{self.data_mgr.data_column}.csv\n")
            f.write(f"Processed {len(results)} dates\n")
        self.logger.info(f"Protocol completion marker created: {completion_marker}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='SMPC Protocol')
    parser.add_argument('--party', required=True, 
                       help='Party name (directory name) or legacy country name')
    parser.add_argument('--party-id', type=int, 
                       help='Party ID. If not provided, derived from party name')
    parser.add_argument('--data-dir', help='Specific data directory (defaults to party name)')
    parser.add_argument('--data-root', default='.', 
                       help='Root directory containing party subdirectories (default: current directory)')
    parser.add_argument('--limit', type=int, help='Limit number of dates')
    parser.add_argument('--column', default='daily_cases', help='Data column to use')
    
    args = parser.parse_args()
    
    try:
        protocol = SMPCProtocol(args.party, args.party_id, args.data_dir, args.data_root)
        protocol.data_mgr.data_column = args.column
        
        # Start server in background
        server_thread = threading.Thread(target=protocol.server.start_server, daemon=True)
        server_thread.start()
        time.sleep(3)  # Allow server to start
        
        # Run protocol
        protocol.run_protocol(args.limit)
        
        # Signal successful completion
        protocol.logger.info("SMPC Protocol completed successfully")
        
        # Keep server running briefly for final communications
        time.sleep(10)
        protocol.server.stop_server()
        
        # Exit gracefully
        sys.exit(0)
        
    except KeyboardInterrupt:
        print("\nShutdown requested")
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()