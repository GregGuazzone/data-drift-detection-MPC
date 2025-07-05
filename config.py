"""
Shared configuration for SMPC protocol
"""
import os
from pathlib import Path
from typing import Dict, List

# Network configuration
DEALER_ID = 0  # Party 0 is the dealer
BASE_PORT = 8000
HOST = 'localhost'
SOCKET_SIZE = 1048576
CONNECTION_TIMEOUT = 30
MAX_RETRIES = 8
RETRY_BASE_DELAY = 1
RETRY_MAX_DELAY = 5

# Dynamic party configuration
_parties_cache = None
_num_parties_cache = None

def discover_parties(data_root_dir: str = ".") -> Dict[str, int]:
    """Discover parties from subdirectories in the data root directory"""
    global _parties_cache
    
    if _parties_cache is not None:
        return _parties_cache
    
    data_path = Path(data_root_dir)
    if not data_path.exists():
        raise ValueError(f"Data root directory not found: {data_root_dir}")
    
    # Find all subdirectories that contain data files
    party_dirs = []
    for item in data_path.iterdir():
        if item.is_dir() and not item.name.startswith('.') and not item.name in ['logs', 'results', 'debug']:
            # Check if directory contains CSV files
            csv_files = list(item.glob("*.csv"))
            if csv_files:
                party_dirs.append(item.name)
    
    if not party_dirs:
        raise ValueError(f"No party directories with CSV files found in {data_root_dir}")
    
    # Sort for consistent ordering
    party_dirs.sort()
    
    # Create mapping
    parties = {party_name: party_id for party_id, party_name in enumerate(party_dirs)}
    
    _parties_cache = parties
    print(f"Discovered {len(parties)} parties: {list(parties.keys())}")
    
    return parties

def get_num_parties(data_root_dir: str = ".") -> int:
    """Get the number of parties"""
    global _num_parties_cache
    
    if _num_parties_cache is not None:
        return _num_parties_cache
    
    parties = discover_parties(data_root_dir)
    _num_parties_cache = len(parties)
    return _num_parties_cache

def reset_party_cache():
    """Reset the party cache (useful for testing or changing directories)"""
    global _parties_cache, _num_parties_cache
    _parties_cache = None
    _num_parties_cache = None

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

NUM_PARTIES = 8
DEFAULT_PARTIES = {
    'party_0': 0,
    'party_1': 1,
    'party_2': 2,
    'party_3': 3,
    'party_4': 4,
    'party_5': 5,
    'party_6': 6,
    'party_7': 7
}

# Data configuration
DEFAULT_COLUMN = 'daily_cases'
RESULTS_DIR = 'results'
LOGS_DIR = 'logs'
DEBUG_DIR = 'debug'

# Protocol timeouts
SHARE_EXCHANGE_TIMEOUT = 600
AGGREGATE_TIMEOUT = 300

# Privacy parameters
SHARE_MAGNITUDE = 100000