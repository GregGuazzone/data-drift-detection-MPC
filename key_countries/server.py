"""
Server component for SMPC protocol
Handles network communication and message passing
"""
import json
import socket
import threading
import time
import logging
import pandas as pd
from typing import Dict, Optional, Set
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import (HOST, BASE_PORT, NUM_PARTIES, SOCKET_SIZE, COUNTRIES, DEALER_ID, CONNECTION_TIMEOUT, MAX_RETRIES) 

class SMPCServer:
    """Handles all network communication for SMPC protocol"""
    
    def __init__(self, country_name: str, party_id: int, logger: Optional[logging.Logger] = None):
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
        self.port = BASE_PORT + self.party_id
        self.logger = logger or logging.getLogger(country_name)
        
        # Communication tracking
        self.shares_sent_to: Set[int] = set()
        self.shares_received_from: Set[int] = set()
        self.aggregates_sent = False
        self.aggregates_received_from: Set[int] = set()
        
        # Data storage
        self.shares_received: Dict[str, Dict[int, float]] = {}
        self.aggregate_sums: Dict[str, Dict[int, float]] = {}
        
        # Server socket
        self.server_socket: Optional[socket.socket] = None
        self._running = False

    def start_server(self):
        """Start the server to listen for incoming connections"""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((HOST, self.port))
            self.server_socket.listen(10)
            self._running = True
            
            self.logger.info(f"Server listening on {HOST}:{self.port}")
            print(f"{self.country_name.upper()} server running on {HOST}:{self.port}")
            print(f"Waiting for connections...")
            
            while self._running:
                try:
                    client_socket, addr = self.server_socket.accept()
                    threading.Thread(
                        target=self._handle_client,
                        args=(client_socket, addr),
                        daemon=True
                    ).start()
                except OSError:
                    if self._running:
                        self.logger.error("Server socket error")
                    break
                    
        except Exception as e:
            self.logger.error(f"Server start error: {e}")
    
    def _handle_client(self, client_socket: socket.socket, addr):
        """Handle incoming client connections"""
        try:
            data = client_socket.recv(SOCKET_SIZE).decode('utf-8')
            message = json.loads(data)
            
            msg_type = message.get('type')
            sender_id = message.get('sender_id')
            
            if msg_type == 'batch_shares':
                self._handle_batch_shares(message)
            elif msg_type == 'batch_aggregates':
                self._handle_batch_aggregates(message)
            
            client_socket.send(b'ACK')
            
        except Exception as e:
            self.logger.error(f"Error handling client {addr}: {e}")
        finally:
            client_socket.close()
    
    def _handle_batch_shares(self, message: Dict):
        """Process incoming batch shares"""
        sender_id = message['sender_id']
        date_shares = message['date_shares']
        
        for date, share in date_shares.items():
            if date not in self.shares_received:
                self.shares_received[date] = {}
            self.shares_received[date][sender_id] = float(share)
        
        self.shares_received_from.add(sender_id)
        self.logger.info(f"Received {len(date_shares)} shares from Party {sender_id}")
        print(f"Received batch of {len(date_shares)} shares from Party {sender_id}")
    
    def _handle_batch_aggregates(self, message: Dict):
        """Process incoming batch aggregates (dealer only)"""
        if self.party_id != DEALER_ID:
            return
        
        sender_id = message['sender_id']
        date_aggregates = message['date_aggregates']
        
        for date, aggregate in date_aggregates.items():
            if date not in self.aggregate_sums:
                self.aggregate_sums[date] = {}
            self.aggregate_sums[date][sender_id] = float(aggregate)
        
        self.aggregates_received_from.add(sender_id)
        self.logger.info(f"Received {len(date_aggregates)} aggregates from Party {sender_id}")
    
    def send_shares_dataframe(self, shares_df: pd.DataFrame) -> bool:
        """Send shares to all parties concurrently"""
        with ThreadPoolExecutor(max_workers=NUM_PARTIES) as executor:
            futures = []
            
            for party_id in range(NUM_PARTIES):
                if party_id == self.party_id:
                    continue
                
                date_shares = shares_df[party_id].to_dict()
                future = executor.submit(self._send_to_party, party_id, 'batch_shares', 
                                       {'date_shares': date_shares})
                futures.append((party_id, future))
            
            # Wait for all sends to complete
            success_count = 0
            for party_id, future in futures:
                try:
                    if future.result(timeout=30):
                        self.shares_sent_to.add(party_id)
                        success_count += 1
                except Exception as e:
                    self.logger.error(f"Failed to send shares to Party {party_id}: {e}")
            
            self.logger.info(f"Successfully sent shares to {success_count}/{len(futures)} parties")
            return success_count == len(futures)
    
    def send_aggregates(self, aggregates: Dict[str, float]) -> bool:
        """Send aggregates to dealer"""
        if self.party_id == DEALER_ID:
            return True  # Dealer doesn't send to itself
        
        return self._send_to_party(DEALER_ID, 'batch_aggregates', 
                                 {'date_aggregates': aggregates})
    
    def _send_to_party(self, target_id: int, msg_type: str, data: Dict) -> bool:
        """Send message to a specific party with retries"""
        target_port = BASE_PORT + target_id
        
        message = {
            'type': msg_type,
            'sender_id': self.party_id,
            'timestamp': time.time(),
            **data
        }
        
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(CONNECTION_TIMEOUT)
                sock.connect((HOST, target_port))
                
                sock.send(json.dumps(message).encode('utf-8'))
                response = sock.recv(1024).decode('utf-8')
                sock.close()
                
                if response == 'ACK':
                    return True
                    
            except Exception as e:
                self.logger.warning(f"Send attempt {attempt}/{MAX_RETRIES} to Party {target_id} failed: {e}")
                if attempt < MAX_RETRIES:
                    time.sleep(2 ** attempt)  # Exponential backoff
        
        return False
    
    def stop_server(self):
        """Stop the server"""
        self._running = False
        if self.server_socket:
            self.server_socket.close()
