#!/usr/bin/env python3
"""
SMPC Server
Basic server setup to receive and handle connections
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

# Configuration
NUM_PARTIES = 8
DEALER_ID = 0
BASE_PORT = 8000
HOST = 'localhost'
COUNTRY_NAME = 'china'

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
PARTY_ID = COUNTRIES[COUNTRY_NAME]

class Server:
    def __init__(self, party_id=PARTY_ID):
        self.party_id = party_id
        self.port = BASE_PORT + party_id
        self.is_dealer = (party_id == DEALER_ID)
        self.server_socket = None
        self.messages_received = {}
        
        self.shares_received = {}
        self.aggregate_sums = {}
        self.target_date = '2021-01-01'
        self.data_column = 'daily_cases'
        
        # Setup simple logging
        log_file = f'{COUNTRY_NAME}_server.log'
        logging.basicConfig(
            level=logging.INFO,
            format=f'[{COUNTRY_NAME}] %(asctime)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(COUNTRY_NAME)
        self.logger.info(f"Initialized {COUNTRY_NAME} server (Party ID: {party_id})")

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
                print(f"{COUNTRY_NAME.upper()} server running on {HOST}:{current_port}")
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
            # Look for CSV files in current directory
            csv_files = list(Path('.').glob('*.csv'))
            
            if not csv_files:
                self.logger.error("No CSV files found in current directory")
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

    def generate_shares(self, value):
        """Generate secret shares for a value"""
        # Generate random shares for all parties except the last
        shares = np.random.uniform(-100, 100, NUM_PARTIES - 1)
        
        # Last share is calculated to ensure sum equals the original value
        last_share = value - sum(shares)
        
        # Combine all shares
        all_shares = np.append(shares, last_share)
        self.logger.info(f"Generated {NUM_PARTIES} shares for value {value}")
        
        return all_shares.tolist()

    def send_share(self, target_id, share_value):
        """Send a share to another party"""
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
            response = sock.recv(1024).decode('utf-8')
            sock.close()
            
            print(f"Sent share to Party {target_id}, response: {response}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send share to Party {target_id}: {e}")
            print(f"Failed to send share: {e}")
            return False

    def send_aggregate(self, aggregate_value):
        """Send local aggregate to the dealer"""
        if self.is_dealer:
            # If we are the dealer, just store locally
            self.aggregate_sums[self.party_id] = float(aggregate_value)
            self.logger.info(f"Stored own aggregate: {aggregate_value}")
            return True
            
        try:
            dealer_port = BASE_PORT + DEALER_ID
            
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(10)
            sock.connect((HOST, dealer_port))
            
            message = {
                'type': 'aggregate',
                'sender_id': int(self.party_id),  # Explicitly send as integer
                'target_date': self.target_date,
                'aggregate': float(aggregate_value),  # Ensure it's a float
                'timestamp': time.time()
            }
            
            sock.send(json.dumps(message).encode('utf-8'))
            response = sock.recv(1024).decode('utf-8')
            sock.close()
            
            print(f"Sent aggregate to dealer, response: {response}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send aggregate to dealer: {e}")
            print(f"Failed to send aggregate: {e}")
            return False

    def handle_incoming_message(self, client_socket, addr):
        """Handle incoming messages - enhanced for SMPC"""
        try:
            data = client_socket.recv(4096).decode('utf-8')
            
            # Try to parse as JSON
            try:
                message = json.loads(data)
                message_type = message.get('type', 'unknown')
                sender_id = message.get('sender_id', 'unknown')
                
                if message_type == 'share':
                    # Process incoming share
                    share_value = message.get('share', 0.0)
                    target_date = message.get('target_date', '')
                    
                    if target_date == self.target_date:
                        # Store share with consistent key type
                        sender_key = int(sender_id) if isinstance(sender_id, str) else sender_id
                        self.shares_received[sender_key] = share_value
                        print(f"Received share from Party {sender_id} for {target_date}")
                        self.logger.info(f"Received share from Party {sender_id}: {share_value}")
                
                elif message_type == 'aggregate' and self.is_dealer:
                    # Process incoming aggregate (dealer only)
                    aggregate_value = message.get('aggregate', 0.0)
                    target_date = message.get('target_date', '')
                    
                    if target_date == self.target_date:
                        # Fix: Ensure sender_id is properly converted and stored
                        try:
                            sender_key = int(sender_id)
                            self.aggregate_sums[sender_key] = float(aggregate_value)
                            
                            print(f"Received aggregate from Party {sender_id} for {target_date}: {aggregate_value}")
                            self.logger.info(f"Received aggregate from Party {sender_id}: {aggregate_value}")
                            print(f"Current aggregates: {self.aggregate_sums}")
                            self.logger.info(f"Current aggregates dictionary: {self.aggregate_sums}")
                        except (ValueError, TypeError) as e:
                            self.logger.error(f"Error processing aggregate from {sender_id}: {e}")
                            print(f"Error processing aggregate: {e}")
                
                # Send acknowledgment
                client_socket.send(b'ACK')
                
            except json.JSONDecodeError:
                # Not JSON, treat as plain text
                print(f"Received message: {data[:50]}...")
                self.logger.info(f"Received text message from {addr}: {data[:100]}...")
                client_socket.send(b'ACK')
                
        except Exception as e:
            self.logger.error(f"Error handling message from {addr}: {e}")
        finally:
            client_socket.close()

    def run_smpc_protocol(self):
        """Run the complete SMPC protocol for target_date"""
        print(f"\nStarting SMPC protocol for {self.target_date}")
        
        # Step 1: Get our own data
        my_value = self.get_data_for_date()
        print(f"{COUNTRY_NAME} value for {self.target_date}: {my_value}")
        
        # Step 2: Generate shares
        shares = self.generate_shares(my_value)
        print(f"Generated {len(shares)} shares")
        
        # Step 3: Distribute shares to all parties
        print(f"Sending shares to all parties...")
        for target_id in range(NUM_PARTIES):
            # Send the appropriate share to each party
            if target_id == self.party_id:
                # Keep our own share
                self.shares_received[self.party_id] = shares[target_id]
                print(f"Kept share for self: {shares[target_id]}")
            else:
                # Send to other party
                sent = self.send_share(target_id, shares[target_id])
                while (not sent):
                    self.logger.warning(f"Retrying to send share to Party {target_id}...")
                    time.sleep(5)
                    sent = self.send_share(target_id, shares[target_id])
                self.logger.info(f"Sent share to Party {target_id}: {shares[target_id]}")
        
        # Step 4: Wait to receive all shares
        print(f"Waiting to receive shares from all parties...")
        max_wait = 60  # seconds
        start_time = time.time()
        
        while len(self.shares_received) < NUM_PARTIES and time.time() - start_time < max_wait:
            print(f" {len(self.shares_received)}/{NUM_PARTIES} shares")
            time.sleep(5)
        
        # Step 5: Compute local sum
        local_sum = sum(self.shares_received.values())
        self.logger.info(f"Local sum of shares: {local_sum}")
        print(f"Local sum of shares: {local_sum}")
        
        # Step 6: Send sum to dealer
        print(f"Sending local sum to dealer...")
        self.send_aggregate(local_sum)
        
        # Step 7: If dealer, combine all aggregates
        if self.is_dealer:
            # Make sure we count our own aggregate
            if self.party_id not in self.aggregate_sums:
                self.aggregate_sums[self.party_id] = float(local_sum)
                self.logger.info(f"Stored own aggregate: {local_sum}")
            
            # Wait for other aggregates
            while len(self.aggregate_sums) < NUM_PARTIES:
                self.logger.info(f"Received {dict(self.aggregate_sums)}")
                print(f"Received {len(self.aggregate_sums)}/{NUM_PARTIES} aggregates")
                time.sleep(5)
            
            # Print final debug info
            self.logger.info(f"Final aggregates dictionary: {dict(self.aggregate_sums)}")
            self.logger.info(f"Aggregate keys: {list(self.aggregate_sums.keys())}")
            print(f"Final aggregates: {dict(self.aggregate_sums)}")
            print(f"Aggregate keys: {list(self.aggregate_sums.keys())}")
            
            # Calculate final sum with detailed logging
            final_sum = sum(self.aggregate_sums.values())
            self.logger.info(f"Final sum of all aggregates: {final_sum}")
            
            # Create output directory if it doesn't exist
            os.makedirs("results", exist_ok=True)
            
            # Save to CSV with more details
            result_df = pd.DataFrame({
                'Date': [self.target_date],
                'Total_Daily_Cases': [final_sum],
                'Countries_Reporting': [len(self.aggregate_sums)],
                'Total_Countries': [NUM_PARTIES]
            })
            
            output_file = f"results/smpc_result_{self.target_date.replace('-','')}.csv"
            result_df.to_csv(output_file, index=False)
            
            print(f"\nSMPC Protocol Complete!")
            print(f"============================================")
            print(f"Date: {self.target_date}")
            print(f"Countries reporting: {len(self.aggregate_sums)}/{NUM_PARTIES}")
            print(f"Total Daily Cases: {final_sum}")
            print(f"Results saved to: {output_file}")
            print(f"============================================")
        else:
            print(f"\nSMPC Protocol Complete! Dealer will compile results.")
        
        return True

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='SMPC Server')
    parser.add_argument('--send', action='store_true', help='Send a test message after starting')
    parser.add_argument('--target', type=int, help='Target party ID to send message to')
    parser.add_argument('--message', type=str, default="Hello World!", help='Message to send')
    parser.add_argument('--run-smpc', action='store_true', help='Run SMPC protocol for 2021-01-01')
    parser.add_argument('--date', type=str, default='2021-01-01', help='Target date for SMPC')
    parser.add_argument('--column', type=str, default='daily_cases', help='Data column to use')
    
    args = parser.parse_args()
    
    print(f"Starting {COUNTRY_NAME.upper()} server (Party ID: {PARTY_ID})")
    
    # Create ONE server instance
    server = Server()
    
    # Set target date and column if specified
    if args.date:
        server.target_date = args.date
    if args.column:
        server.data_column = args.column
    
    if args.run_smpc:
        # If run-smpc flag is passed, run BOTH server AND protocol
        # First start server in background thread
        server_thread = threading.Thread(target=server.start_server)
        server_thread.daemon = True
        server_thread.start()
        
        # Give server time to initialize
        time.sleep(3)
        
        # Then run protocol using same instance
        server.run_smpc_protocol()
    else:
        # Just run server if no protocol flag
        server.start_server()

if __name__ == '__main__':
    main()