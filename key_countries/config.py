"""
Shared configuration for SMPC protocol
"""
import os

# Network configuration
NUM_PARTIES = 8
DEALER_ID = 0  # China is the dealer
BASE_PORT = 8000
HOST = 'localhost'
SOCKET_SIZE = 1048576  # 1MB buffer
CONNECTION_TIMEOUT = 10
MAX_RETRIES = 5

COUNTRIES = {
    'china': 0,       # Dealer
    'france': 1,
    'germany': 2,
    'iran': 3,
    'italy': 4,
    'spain': 5,
    'united_kingdom': 6,
    'us': 7
}

# Reverse mapping for lookups
PARTY_TO_COUNTRY = {v: k for k, v in COUNTRIES.items()}

# Data configuration
DEFAULT_COLUMN = 'daily_cases'
RESULTS_DIR = 'results'
LOGS_DIR = 'logs'
DEBUG_DIR = 'debug'

# Protocol timeouts
SHARE_EXCHANGE_TIMEOUT = 300
AGGREGATE_TIMEOUT = 180

# Privacy parameters
SHARE_MAGNITUDE = 100000