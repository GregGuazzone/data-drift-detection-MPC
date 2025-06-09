"""
SMPC Server
Generic protocol implementation that can be used by any country
"""

import json
import socket
import threading
import time
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import os
import argparse
from datetime import datetime

# Configuration
NUM_PARTIES = 8
DEALER_ID = 0  # China is the dealer (Party ID 0)
BASE_PORT = 8000
HOST = 'localhost'

COUNTRIES = {
    'china': 0,
    'france': 1,
    'germany': 2,
    'iran': 3,
    'italy': 4,
    'spain': 5,
    'united_kingdom': 6,
    'us': 7
}

class Server:
    def __init__(self, country_name=None, party_id=None, data_dir=None):
        # Set country-specific parameters
        if country_name is None and party_id is None:
            raise ValueError("Must provide either country_name or party_id")
        if country_name is not None:
            if country_name not in COUNTRIES:
                raise ValueError(f"Invalid country name: {country_name}. Must be one of: {', '.join(COUNTRIES.keys())}")
            self.party_id = COUNTRIES[country_name]
        else:
            if party_id < 0 or party_id >= NUM_PARTIES:
                raise ValueError(f"Invalid party_id: {party_id}. Must be between 0 and {NUM_PARTIES - 1}")
            self.party_id = party_id 
        
        self.country_name = country_name
        self.party_id = COUNTRIES.get(country_name, 0)
        self.port = BASE_PORT + self.party_id
        self.is_dealer = (self.party_id == DEALER_ID)
        self.server_socket = None
        self.messages_received = {}
        
        self.shares_received = {}
        self.aggregate_sums = {}
        self.target_date = '2021-01-01'
        self.data_column = 'daily_cases'
        
        # Initialize protocol tracking variables immediately
        self.shares_sent_to = set()
        self.shares_received_from = set()
        self.aggregates_sent = False
        self.aggregates_received_from = set()
        
        # Use provided data directory or default to country subdirectory
        self.data_dir = data_dir or f"{country_name}"
        
        # Create logs directory properly
        log_dir = os.path.join("logs")
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = os.path.join(log_dir, f'{country_name}_server.log')
        logging.basicConfig(
            level=logging.INFO,
            format=f'[{country_name}] %(asctime)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(country_name)
        self.logger.info(f"Initialized {country_name} server (Party ID: {self.party_id})")
        self.logger.info(f"Using data directory: {self.data_dir}")
        
        # Instance ID for debugging
        self.instance_id = id(self)
        self.logger.info(f"Server instance ID: {hex(self.instance_id)}")

    def start_server(self):
        """Start basic server to receive messages with port fallback"""
        base_port = self.port
        max_attempts = 5
        
        for attempt in range(max_attempts):
            try:
                self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                
                # Try current port
                current_port = base_port + (attempt * 10)
                self.server_socket.bind((HOST, current_port))
                self.server_socket.listen(NUM_PARTIES)
                
                # Update port if we had to use a different one
                if current_port != base_port:
                    self.port = current_port
                    
                self.logger.info(f"Server successfully listening on {HOST}:{current_port}")
                print(f"{self.country_name.upper()} server running on {HOST}:{current_port}")
                print(f"Waiting for connections...")
                
                # Keep accepting connections until interrupted
                while True:
                    try:
                        client_socket, addr = self.server_socket.accept()
                        self.logger.info(f"Connection from {addr}")
                        thread = threading.Thread(
                            target=self.handle_incoming_message, 
                            args=(client_socket, addr)
                        )
                        thread.daemon = True
                        thread.start()
                    except KeyboardInterrupt:
                        print("\nServer shutdown requested")
                        break
                    except Exception as e:
                        self.logger.error(f"Error accepting connection: {e}")
                        
                return  # Exit if server runs successfully
                    
            except socket.error as e:
                if e.errno == 48:  # Address already in use
                    self.logger.warning(f"Port {current_port} is in use, trying port {base_port + ((attempt + 1) * 10)}...")
                else:
                    self.logger.error(f"Socket error: {e}")
                    break
            except Exception as e:
                self.logger.error(f"Server error: {e}")
                break
                
        self.logger.error("Failed to start server after multiple attempts")
        print("Failed to start server after multiple attempts")

    def get_data_for_date(self):
        """Get the daily_cases data for target_date from local CSV"""
        try:
            # Look for CSV files in data directory
            csv_path = Path(self.data_dir)
            self.logger.info(f"Looking for CSV files in: {csv_path.absolute()}")
            
            csv_files = list(csv_path.glob('*.csv'))
            
            if not csv_files:
                self.logger.error(f"No CSV files found in {csv_path.absolute()}")
                return 0.0
            
            # Use the first CSV file found or a specific one if multiple exist
            for csv_file in csv_files:
                self.logger.info(f"Found CSV file: {csv_file}")
                
                # Load the CSV
                df = pd.read_csv(csv_file)
                self.logger.info(f"Loaded CSV with {len(df)} rows")
                
                # Check for the specified date
                if 'Date' in df.columns:
                    filtered_df = df[df['Date'] == self.target_date]
                    
                    if len(filtered_df) > 0:
                        # Get the data for our target column
                        if self.data_column in filtered_df.columns:
                            value = filtered_df[self.data_column].iloc[0]
                            self.logger.info(f"Found value {value} for {self.target_date}")
                            return float(value)
                        else:
                            self.logger.warning(f"Column {self.data_column} not found in CSV")
                    else:
                        self.logger.warning(f"Date {self.target_date} not found in CSV")
            
            self.logger.warning(f"Using default value 0.0 for {self.target_date}")
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            return 0.0

    def generate_shares(self, value, num_parties=NUM_PARTIES):
        """Generate secret shares for a value"""
        # Generate random shares for all parties except the last
        shares = np.random.uniform(-100, 100, num_parties - 1)
        # Last share is calculated to ensure sum equals the original value
        last_share = value - sum(shares)
        # Combine all shares
        all_shares = np.append(shares, last_share)
        self.logger.info(f"Generated {num_parties} shares for value {value}")
        
        return all_shares.tolist()

    def check_server_ready(self, target_id, max_retries=5):
        """Check if server at target_id is ready to receive shares"""
        target_port = BASE_PORT + target_id
        
        for attempt in range(max_retries):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(2)  # Short timeout for quick check
                sock.connect((HOST, target_port))
                
                # Send a ping message
                message = {
                    'type': 'ping',
                    'sender_id': self.party_id,
                    'timestamp': time.time()
                }
                
                sock.send(json.dumps(message).encode('utf-8'))
                response = sock.recv(1048576).decode('utf-8')
                sock.close()
                
                if response == 'ACK':
                    return True
                    
            except (socket.timeout, ConnectionRefusedError):
                print(f"Server {target_id} not ready (attempt {attempt+1}/{max_retries}), retrying...")
                time.sleep(2)
                
            except Exception as e:
                print(f"Error checking if server {target_id} is ready: {e}")
                time.sleep(1)
                
        print(f"Server {target_id} not responding after {max_retries} attempts")
        return False

    def send_share(self, target_id, share_value):
        """Send a share to another party with retries"""
        # First check if server is ready
        if not self.check_server_ready(target_id):
            print(f"WARNING: Server {target_id} not ready, but trying to send anyway")
        
        max_retries = 5
        for attempt in range(max_retries):
            try:
                target_port = BASE_PORT + target_id
                
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(10)
                sock.connect((HOST, target_port))
                
                message = {
                    'type': 'share',
                    'sender_id': self.party_id,
                    'target_date': self.target_date,
                    'share': share_value,
                    'timestamp': time.time()
                }
                
                sock.send(json.dumps(message).encode('utf-8'))
                response = sock.recv(1048576).decode('utf-8')
                sock.close()
                
                print(f"Sent share to Party {target_id}, response: {response}")
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to send share to Party {target_id} (attempt {attempt+1}/{max_retries}): {e}")
                print(f"Failed to send share (attempt {attempt+1}/{max_retries}): {e}")
                time.sleep(2)
        
        print(f"All attempts to send share to Party {target_id} failed")
        return False

    def send_aggregate(self, aggregate_value):
        """Send local aggregate to the dealer with retries"""
        if self.is_dealer:
            # If we are the dealer, just store locally
            self.aggregate_sums[self.party_id] = float(aggregate_value)
            self.logger.info(f"Stored own aggregate: {aggregate_value}")
            return True
            
        max_retries = 3
        for attempt in range(max_retries):
            try:
                dealer_port = BASE_PORT + DEALER_ID
                
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(10)
                sock.connect((HOST, dealer_port))
                
                message = {
                    'type': 'aggregate',
                    'sender_id': int(self.party_id),
                    'target_date': self.target_date,
                    'aggregate': float(aggregate_value),
                    'timestamp': time.time()
                }
                
                sock.send(json.dumps(message).encode('utf-8'))
                response = sock.recv(1048576).decode('utf-8')
                sock.close()
                
                print(f"Sent aggregate to dealer (attempt {attempt+1}/{max_retries}), response: {response}")
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to send aggregate to dealer (attempt {attempt+1}/{max_retries}): {e}")
                print(f"Failed to send aggregate (attempt {attempt+1}/{max_retries}): {e}")
                time.sleep(2)
        
        print("All attempts to send aggregate failed")
        return False

    def handle_incoming_message(self, client_socket, addr):
        """Handle incoming messages - enhanced for SMPC"""
        try:
            # Increase buffer size to 1MB (from 4KB)
            data = client_socket.recv(1048576).decode('utf-8')
            
            try:
                message = json.loads(data)
                message_type = message.get('type', 'unknown')
                sender_id = int(message.get('sender_id', 0))
                
                # Always acknowledge receipt
                client_socket.send(b'ACK')
                
                if message_type == 'batch_shares':
                    # Process incoming batch of shares for multiple dates
                    date_shares = message.get('date_shares', {})
                    
                    # Record that we received shares from this party
                    self.shares_received_from.add(sender_id)
                    
                    print(f"Received batch of {len(date_shares)} shares from Party {sender_id}")
                    self.logger.info(f"Received batch of {len(date_shares)} shares from Party {sender_id}")
                    
                    # Store each share by date
                    for date, share_value in date_shares.items():
                        if date not in self.shares_received:
                            self.shares_received[date] = {}
                        self.shares_received[date][sender_id] = share_value
                        self.logger.info(f"Stored share for date {date} from Party {sender_id}: {share_value}")
            
                elif message_type == 'batch_aggregates' and self.is_dealer:
                    # Process incoming batch of aggregates
                    date_sums = message.get('date_sums', {})
                    
                    # Record that we received aggregates from this party
                    self.aggregates_received_from.add(sender_id)
                    
                    print(f"Received batch of {len(date_sums)} aggregates from Party {sender_id}")
                    self.logger.info(f"Received batch of {len(date_sums)} aggregates from Party {sender_id}")
                    
                    # Store each aggregate
                    for date, sum_value in date_sums.items():
                        if date not in self.aggregate_sums:
                            self.aggregate_sums[date] = {}
                        self.aggregate_sums[date][sender_id] = float(sum_value)
                
            except json.JSONDecodeError:
                client_socket.send(b'ACK')
                
        except Exception as e:
            self.logger.error(f"Error handling message from {addr}: {e}")
        finally:
            client_socket.close()

    def get_all_dates(self):
        """Get all unique dates from the dataset"""
        try:
            csv_path = Path(self.data_dir)
            self.logger.info(f"Looking for CSV files in: {csv_path.absolute()}")
            
            csv_files = list(csv_path.glob('*.csv'))
            
            if not csv_files:
                self.logger.error(f"No CSV files found in {csv_path.absolute()}")
                return []
            
            for csv_file in csv_files:
                self.logger.info(f"Reading CSV file: {csv_file}")
                df = pd.read_csv(csv_file)
                
                if 'Date' in df.columns:
                    # Get all dates where the target column has valid values
                    if self.data_column in df.columns:
                        valid_dates = df[df[self.data_column].notna()]['Date'].unique().tolist()
                        self.logger.info(f"Found {len(valid_dates)} valid dates in dataset")
                        return sorted(valid_dates)
                    else:
                        self.logger.warning(f"Column {self.data_column} not found in CSV")
                else:
                    self.logger.warning(f"'Date' column not found in CSV")    
        except Exception as e:
            self.logger.error(f"Error getting dates: {e}")
        return []

    def get_data_for_specific_date(self, date):
        """Get data for a specific date from local CSV"""
        try:
            csv_path = Path(self.data_dir)
            csv_files = list(csv_path.glob('*.csv'))
            
            if not csv_files:
                return 0.0
            
            for csv_file in csv_files:
                df = pd.read_csv(csv_file)
                
                if 'Date' in df.columns:
                    filtered_df = df[df['Date'] == date]
                    
                    if len(filtered_df) > 0 and self.data_column in filtered_df.columns:
                        value = filtered_df[self.data_column].iloc[0]
                        return float(value)
            
            return 0.0
        except Exception as e:
            self.logger.error(f"Error loading data for date {date}: {e}")
            return 0.0

    def run_smpc_for_all_dates(self, limit_dates=True, limit_size=1):
        print(f"\nStarting SMPC protocol for all dates in dataset")
        # Get data and dates
        if limit_dates:
            all_dates = self._load_data_and_create_progress_marker(limit_dates=True, limit_size=limit_size)
        else:
            all_dates = self._load_data_and_create_progress_marker(limit_dates=False)
        if not all_dates:
            return False
        
        # Process data and generate shares
        my_values, shares_by_recipient = self._generate_and_organize_shares(all_dates)
        
        # Exchange shares with other parties
        self._distribute_and_collect_shares(all_dates, shares_by_recipient)
        
        # Compute local sums and send to dealer
        local_sums = self._compute_and_send_local_sums(all_dates)
        
        # Dealer processes results (or non-dealer finishes)
        if self.is_dealer:
            self._collect_and_process_aggregates(all_dates, local_sums)
        else:
            print(f"Participant {self.country_name}: SMPC Protocol Complete!")
            self.logger.info(f"SMPC Protocol Complete!")
    
        return True

    def _load_data_and_create_progress_marker(self, limit_dates=True, limit_size=1):
        """Load data and create progress marker (not completion marker)"""
        # Get all dates
        all_dates = self.get_all_dates()
        if not all_dates:
            self.logger.error("No dates found in dataset")
            return None
        
        if self.is_dealer:
            os.makedirs("results", exist_ok=True)
            progress_marker = os.path.join("results", "protocol_progress.marker")
            with open(progress_marker, "w") as f:
                f.write(f"Protocol started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Processing {len(all_dates)} dates\n")
            self.logger.info(f"Created progress marker")
            
            # IMPORTANT: Remove any existing completion marker to avoid confusion
            completion_marker = os.path.join("results", "protocol_complete.marker")
            if os.path.exists(completion_marker):
                os.remove(completion_marker)
                self.logger.info("Removed existing completion marker")
        
        if limit_dates:
            # Limit dataset size for initial testing
            print(f"Limiting dataset to first {limit_size} dates for initial testing")
            self.logger.info(f"Limiting dataset to first {limit_size} dates for initial testing")
            all_dates = all_dates[:limit_size]
        
        return all_dates

    def _generate_and_organize_shares(self, all_dates):
        """Generate shares for all dates and organize by recipient"""
        # Load values for our country
        my_values = {}
        for date in all_dates:
            value = self.get_data_for_specific_date(date)
            my_values[date] = value
            self.logger.info(f"Date {date}: loaded value {value}")
        
        print(f"Loaded data for {len(my_values)} dates")
        self.logger.info(f"Loaded data for {len(my_values)} dates with values: {list(my_values.values())[:5]} (first 5)")
        
        # Generate shares for each date
        all_shares = {}
        for date, value in my_values.items():
            shares = self.generate_shares(value)
            all_shares[date] = shares
            share_sum = sum(shares)
            self.logger.info(f"Date {date}: generated shares {shares}")
            self.logger.info(f"Date {date}: sum of shares = {share_sum} (original: {value})")
    
        print(f"Generated shares for {len(all_shares)} dates")
        
        # Reorganize shares by recipient
        shares_by_recipient = {party_id: {} for party_id in range(NUM_PARTIES)}
        for date, shares_list in all_shares.items():
            for party_id, share in enumerate(shares_list):
                shares_by_recipient[party_id][date] = share
    
        for date, share in shares_by_recipient[self.party_id].items():
            if date not in self.shares_received:
                self.shares_received[date] = {}
            self.shares_received[date][self.party_id] = share
    
        print(f"Kept own shares for {len(shares_by_recipient[self.party_id])} dates")
        self.logger.info(f"Kept own shares for {len(shares_by_recipient[self.party_id])} dates")
    
        # Log what's in shares_received after adding our share
        total_parties = sum(len(shares) for shares in self.shares_received.values())
        self.logger.info(f"After adding our share: shares_received has entries for {len(self.shares_received)} dates with {total_parties} total party entries")
    
        return my_values, shares_by_recipient

    def _distribute_and_collect_shares(self, all_dates, shares_by_recipient):
        """Send shares to other parties and wait to receive theirs"""
        # Send shares to all other parties
        for target_id, date_shares in shares_by_recipient.items():
            if target_id == self.party_id:
                continue  # Skip ourselves
                
            self.logger.info(f"Sending batch of {len(date_shares)} shares to Party {target_id}...")
            
            sent = self.send_batch_shares(target_id, date_shares)
            self.logger.info(f"Sent: {sent}")

            if sent:
                self.shares_sent_to.add(target_id)
                self.logger.info(f"Successfully sent shares to Party {target_id}")
            else:
                self.logger.error(f"Failed to send shares to Party {target_id}")
        
        # Wait to confirm all shares are sent and received
        print("Waiting for all share exchanges to complete...")
        
        # Define the confirmation check function
        def all_shares_exchanged():
            self.logger.info("Checking if all shares have been exchanged...")
            self.logger.info(f"Shares sent to: {self.shares_sent_to}")
            self.logger.info(f"Shares received from: {self.shares_received_from}")

            all_sent = len(self.shares_sent_to) == NUM_PARTIES - 1
            all_received = len(self.shares_received_from) == NUM_PARTIES - 1
            
            return all_sent and all_received
        
        # Poll until complete or timeout
        max_wait = 300  # Still have a maximum timeout for safety
        start_time = time.time()
        status_time = time.time()
        
        while not all_shares_exchanged() and time.time() - start_time < max_wait:
            if time.time() - status_time >= 10:
                print(f"Share exchange status: sent to {len(self.shares_sent_to)}/{NUM_PARTIES-1} parties, " +
                      f"Received from {len(self.shares_received_from)}/{NUM_PARTIES-1} parties")
                status_time = time.time()
            time.sleep(1)

    def _compute_and_send_local_sums(self, all_dates):
        """Compute local sums from received shares and send to dealer"""
        # Convert all string dates to a standard format
        standardized_shares = {}
        for date_key in self.shares_received:
            # Standardize date format if it's a string date
            std_date = date_key
            if isinstance(date_key, str):
                # Try different formats
                for fmt in ['%Y-%m-%d', '%Y/%m/%d', '%m/%d/%Y']:
                    try:
                        dt = datetime.strptime(date_key, fmt)
                        std_date = dt.strftime('%Y-%m-%d')
                        break
                    except ValueError:
                        continue
        
            # Store with standard key
            standardized_shares[std_date] = self.shares_received[date_key]
    
        # Replace with standardized version
        self.shares_received = standardized_shares
    
        local_sums = {}
        missing_parties_by_date = {}
        
        for date in all_dates:
            if date in self.shares_received:
                date_shares = self.shares_received[date]
                
                # Check which parties we have shares from
                received_parties = set(date_shares.keys())
                expected_parties = set(range(NUM_PARTIES))
                missing_parties = expected_parties - received_parties
                
                if missing_parties:
                    missing_parties_by_date[date] = missing_parties
                    self.logger.warning(f"Date {date}: Missing shares from parties {missing_parties}")
                
                # Compute sum (we need shares from ALL parties)
                sum_value = sum(date_shares.values())
                local_sums[date] = sum_value
                
                # Log the detailed calculation
                share_details = [f"{p}:{date_shares.get(p, 'MISSING')}" for p in sorted(expected_parties)]
                calc_str = " + ".join([f"{p}:{date_shares[p]:.2f}" for p in sorted(date_shares.keys())])
                self.logger.info(f"Date {date}: sum calculation: {calc_str} = {sum_value}")
                self.logger.info(f"Date {date}: computed sum {sum_value} from {len(date_shares)}/{NUM_PARTIES} shares")
            else:
                local_sums[date] = 0.0
                self.logger.warning(f"No shares received for date {date}")
        
        # Log summary of missing shares
        if missing_parties_by_date:
            self.logger.warning(f"Missing shares for {len(missing_parties_by_date)}/{len(all_dates)} dates")
        else:
            self.logger.info("Complete shares received for all dates")
        
        print(f"Computed local sums for {len(local_sums)} dates")
        self.logger.info(f"Computed local sums for {len(local_sums)} dates")
        
        print(f"Sending all local sums to dealer...")
        self.logger.info(f"Sending all local sums to dealer...")
        
        sent = self.send_batch_aggregates(local_sums)
        if sent:
            self.aggregates_sent = True
            print("Successfully sent aggregates to dealer")
            self.logger.info("Successfully sent aggregates to dealer")
        else:
            print("Failed to send aggregates to dealer")
            self.logger.error("Failed to send aggregates to dealer")
            
        return local_sums

    def _collect_and_process_aggregates(self, all_dates, local_sums):
        """Dealer-only: Collect aggregates and compute final results"""
        print(f"Dealer waiting for aggregates from all parties...")
        self.logger.info(f"Dealer waiting for aggregates from all parties...")

        max_wait = 300  # seconds
        start_time = time.time()
        last_status = time.time()

        # Initialize aggregate sums dictionary if not exist
        if not hasattr(self, 'aggregate_sums'):
            self.aggregate_sums = {}

        # Store own aggregates
        for date, sum_value in local_sums.items():
            if date not in self.aggregate_sums:
                self.aggregate_sums[date] = {}
            self.aggregate_sums[date][self.party_id] = sum_value
            self.logger.info(f"Stored own aggregate for date {date}: {sum_value}")

        # Track which parties have sent their aggregates for each date
        self.logs = {}  # {date: set(party_ids)}
        for date in all_dates:
            self.logs[date] = set()
            if date in self.aggregate_sums:
                self.logs[date].update(self.aggregate_sums[date].keys())

        # Wait for other parties' aggregates
        while time.time() - start_time < max_wait:
            # Count dates with complete aggregates
            complete_dates = 0
            missing_dates = []

            for date in all_dates:
                if date in self.aggregate_sums and len(self.aggregate_sums[date]) == NUM_PARTIES:
                    complete_dates += 1
                else:
                    missing_dates.append(date)
                # Update logs for this date
                if date in self.aggregate_sums:
                    self.logs[date] = set(self.aggregate_sums[date].keys())
                else:
                    self.logs[date] = set()

            # Print status every 10 seconds
            if time.time() - last_status >= 10:
                print(f"Have complete aggregates for {complete_dates}/{len(all_dates)} dates")
                self.logger.info(f"Have complete aggregates for {complete_dates}/{len(all_dates)} dates")
                if len(missing_dates) > 0:
                    self.logger.info(f"First few missing dates: {missing_dates[:5]}")
                # Log which parties have sent their aggregates for the first few missing dates
                for date in missing_dates[:5]:
                    sent_parties = sorted(self.logs[date])
                    self.logger.info(f"Date {date}: Aggregates received from parties: {sent_parties}")
                last_status = time.time()

            if complete_dates == len(all_dates):
                print(f"Received all aggregates for all dates!")
                self.logger.info(f"Received all aggregates for all dates!")
                break

            time.sleep(1)

        # Process final results
        self._compute_and_save_final_results(all_dates)

    def _compute_and_save_final_results(self, all_dates):
        """Compute final results and save to file"""
        print("Computing final results...")
        self.logger.info("Computing final results...")
        
        results_df = pd.DataFrame(columns=['Date', 'Total_Daily_Cases', 'Countries_Reporting', 'Total_Countries'])
        
        for date in all_dates:
            if date in self.aggregate_sums:
                party_sums = self.aggregate_sums[date]
                if party_sums:  # Check if the dictionary is not empty
                    final_sum = sum(party_sums.values())
                    
                    # Log detailed breakdown for debugging
                    agg_details = []
                    for party_id in sorted(party_sums.keys()):
                        agg_value = party_sums[party_id]
                        agg_details.append(f"{party_id}:{agg_value:.2f}")
                    
                    calc_str = " + ".join(agg_details)
                    self.logger.info(f"Final sum calculation: {calc_str} = {final_sum}")
                    
                    new_row = pd.DataFrame({
                        'Date': [date],
                        'Total_Daily_Cases': [final_sum],
                        'Countries_Reporting': [len(party_sums)],
                        'Total_Countries': [NUM_PARTIES]
                    })
                    
                    results_df = pd.concat([results_df, new_row], ignore_index=True)
                    self.logger.info(f"Date {date}: Final sum = {final_sum} from {len(party_sums)} countries")
        
        # Save results and create completion marker
        if len(results_df) > 0:
            results_df = results_df.sort_values('Date')
            output_file = "results/smpc_results_all_dates.csv"
            results_df.to_csv(output_file, index=False)
            
            print(f"Saved results for {len(results_df)} dates to {output_file}")
            self.logger.info(f"Saved results for {len(results_df)} dates to {output_file}")
            
            # Create completion marker ONLY NOW - after results are saved
            completion_marker = os.path.join("results", "protocol_complete.marker")
            with open(completion_marker, "w") as f:
                f.write(f"Protocol completed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Processed {len(results_df)}/{len(all_dates)} dates successfully\n")
                f.write(f"Results saved to: {output_file}\n")
            self.logger.info(f"Created completion marker after saving results")
        else:
            print("No results to save!")
            self.logger.error("No results to save!")
            
            # Create completion marker with error information
            completion_marker = os.path.join("results", "protocol_complete.marker")
            with open(completion_marker, "w") as f:
                f.write(f"Protocol completed with errors at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Processed 0/{len(all_dates)} dates\n")
                f.write(f"No results were generated!\n")
            self.logger.error(f"Created completion marker with error information")

    def send_batch_shares(self, target_id, date_shares):
        self.logger.info(f"!---! Sending batch shares to Party {target_id} for {len(date_shares)} dates")
        """Send a batch of shares for multiple dates to another party"""
        # First check if server is ready
        if not self.check_server_ready(target_id):
            print(f"WARNING: Server {target_id} not ready, but trying to send anyway")
            self.logger.warning(f"Server {target_id} not ready, but trying to send anyway")
        
        max_retries = 5
        for attempt in range(max_retries):
            self.logger.info(f"Attempting to send batch shares to Party {target_id} (attempt {attempt+1}/{max_retries})")
            try:
                target_port = BASE_PORT + target_id
                
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(30)  # Longer timeout for batch sends
                sock.connect((HOST, target_port))
                
                message = {
                    'type': 'batch_shares',
                    'sender_id': self.party_id,
                    'date_shares': date_shares,
                    'timestamp': time.time()
                }
                
                # Convert to JSON and send
                json_data = json.dumps(message)
                sock.send(json_data.encode('utf-8'))
                response = sock.recv(1048576).decode('utf-8')
                sock.close()
                
                print(f"Sent batch of {len(date_shares)} shares to Party {target_id}, response: {response}")
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to send batch shares to Party {target_id} (attempt {attempt+1}/{max_retries}): {e}")
                print(f"Failed to send batch shares (attempt {attempt+1}/{max_retries}): {e}")
                time.sleep(2)
        
        print(f"All attempts to send batch shares to Party {target_id} failed")
        return False

    def send_batch_aggregates(self, date_sums):
        """Send batch of aggregates for all dates to the dealer"""
        if self.is_dealer:
            # If we are the dealer, just store locally
            for date, sum_value in date_sums.items():
                if date not in self.aggregate_sums:
                    self.aggregate_sums[date] = {}
                self.aggregate_sums[date][self.party_id] = sum_value
            self.logger.info(f"Stored own aggregates for {len(date_sums)} dates")
            return True
            
        max_retries = 5
        for attempt in range(max_retries):
            try:
                dealer_port = BASE_PORT + DEALER_ID
                
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(30)  # Longer timeout for batch sends
                sock.connect((HOST, dealer_port))
                
                message = {
                    'type': 'batch_aggregates',
                    'sender_id': int(self.party_id),
                    'date_sums': date_sums,
                    'timestamp': time.time()
                }
                
                json_data = json.dumps(message)
                sock.send(json_data.encode('utf-8'))
                response = sock.recv(1048576).decode('utf-8')
                sock.close()
                
                print(f"Sent {len(date_sums)} aggregates to dealer, response: {response}")
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to send batch aggregates (attempt {attempt+1}/{max_retries}): {e}")
                print(f"Failed to send batch aggregates (attempt {attempt+1}/{max_retries}): {e}")
                time.sleep(2)
        
        print("All attempts to send batch aggregates failed")
        return False
    
    def _debug_dump_shares(self, all_dates):
        """Dump all shares for debugging purposes"""
        try:
            debug_dir = "debug"
            os.makedirs(debug_dir, exist_ok=True)
            
            # Dump received shares
            shares_file = os.path.join(debug_dir, f"shares_{self.country_name}.csv")
            with open(shares_file, 'w') as f:
                f.write("Date,PartyID,ShareValue\n")
                for date in all_dates:
                    if date in self.shares_received:
                        for party_id, share_value in self.shares_received[date].items():
                            f.write(f"{date},{party_id},{share_value}\n")
            
            # Dump aggregates if we're the dealer
            if self.is_dealer and hasattr(self, 'aggregate_sums'):
                agg_file = os.path.join(debug_dir, "aggregates.csv")
                with open(agg_file, 'w') as f:
                    f.write("Date,PartyID,AggregateValue\n")
                    for date in all_dates:
                        if date in self.aggregate_sums:
                            for party_id, agg_value in self.aggregate_sums[date].items():
                                f.write(f"{date},{party_id},{agg_value}\n")
            
            self.logger.info(f"Dumped debug data to {debug_dir} directory")
        except Exception as e:
            self.logger.error(f"Failed to dump debug data: {e}")

def main():
    parser = argparse.ArgumentParser(description='Generic SMPC Protocol')
    parser.add_argument('--country', type=str, required=True, 
                        help='Country name (china, france, germany, iran, italy, spain, united_kingdom, us)')
    parser.add_argument('--run-smpc', action='store_true', 
                        help='Run SMPC protocol')
    parser.add_argument('--all-dates', action='store_true',
                        help='Run protocol for all dates in the dataset')
    parser.add_argument('--date', type=str, default='2021-01-01', 
                        help='Target date for single date mode')
    parser.add_argument('--column', type=str, default='daily_cases', 
                        help='Data column to use')
    parser.add_argument('--data-dir', type=str, 
                        help='Directory containing country data files')
    parser.add_argument('--limit', type=int, default=1,
                        help='Number of dates to limit dataset to for testing')
    
    args = parser.parse_args()
    
    try:
        # Validate country name
        if args.country not in COUNTRIES:
            print(f"Error: Invalid country name. Must be one of: {', '.join(COUNTRIES.keys())}")
            sys.exit(1)
        
        print(f"Starting {args.country.upper()} server (Party ID: {COUNTRIES[args.country]})")
        
        # Create the server instance with the specified country and data directory
        server = Server(country_name=args.country, data_dir=args.data_dir)
        
        # Set target date and column if specified
        if args.date:
            server.target_date = args.date
        if args.column:
            server.data_column = args.column
        
        if args.run_smpc:
            # Run server in background thread
            server_thread = threading.Thread(target=server.start_server)
            server_thread.daemon = True
            server_thread.start()
            
            # Give server time to initialize
            time.sleep(3)
            
            if args.limit >= 1:
                print(f"Limiting dataset to first {args.limit} dates for initial testing")
                server.run_smpc_for_all_dates(limit_dates=True, limit_size=args.limit)
            else:
                server.run_smpc_for_all_dates(limit_dates=False)
            
            # CRITICAL: Keep the main thread alive to handle connections
            print("Protocol execution completed, keeping server alive for communications...")
            try:
                while server_thread.is_alive():
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\nShutdown requested.")
        else:
            # Just run server if no protocol flag
            server.start_server()
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()