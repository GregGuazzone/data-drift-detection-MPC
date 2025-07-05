"""
Orchestrates the SMPC protocol across multiple parties
"""
import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

from config import discover_parties, COUNTRIES, LOGS_DIR, RESULTS_DIR


class SMPCOrchestrator:
    """Orchestrates SMPC protocol execution across all parties"""
    
    def __init__(self, data_root_dir: str = ".", limit: int = 0, column: str = 'daily_cases', use_legacy: bool = False):
        self.data_root_dir = data_root_dir
        self.limit = limit
        self.column = column
        self.use_legacy = use_legacy
        self.processes: Dict[str, subprocess.Popen] = {}
        self.log_files = {}
        
        # Choose party configuration
        if use_legacy:
            self.parties = COUNTRIES
            print("Using legacy country-based configuration")
        else:
            self.parties = discover_parties(data_root_dir)
            print(f"Discovered parties from {data_root_dir}: {list(self.parties.keys())}")
        
        # Setup directories
        for directory in [LOGS_DIR, RESULTS_DIR]:
            os.makedirs(directory, exist_ok=True)
    
    def run_protocol(self):
        """Run the complete SMPC protocol"""
        try:
            self._cleanup_previous_run()
            self._start_all_parties()
            self._wait_for_completion()
            self._show_results()
            
        except KeyboardInterrupt:
            print("\n\nShutdown requested by user")
        except Exception as e:
            print(f"\nERROR: {e}")
        finally:
            self._cleanup()
    
    def _cleanup_previous_run(self):
        """Clean up from any previous run"""
        completion_marker = Path(self.data_root_dir) / "protocol_complete.marker"
        if completion_marker.exists():
            completion_marker.unlink()
    
    def _start_all_parties(self):
        """Start all party processes with improved timing for large deployments"""
        print("Starting SMPC Protocol")
        print("=" * 50)
        
        party_names = list(self.parties.keys())
        num_parties = len(party_names)
        
        # Start dealer (party 0) first
        dealer_name = party_names[0]
        self._start_party(dealer_name)
        
        # For large deployments, give dealer more time to initialize
        dealer_wait = 10 if num_parties > 10 else 5
        print(f"Dealer started, waiting {dealer_wait}s for initialization...")
        time.sleep(dealer_wait)
        
        # Start other parties in smaller batches to avoid overwhelming the system
        batch_size = min(5, max(1, num_parties // 10))  # Dynamic batch size
        remaining_parties = party_names[1:]
        
        for i in range(0, len(remaining_parties), batch_size):
            batch = remaining_parties[i:i + batch_size]
            
            print(f"Starting batch {i//batch_size + 1}: {len(batch)} parties")
            for party_name in batch:
                self._start_party(party_name)
                time.sleep(0.5)  # Small delay between parties in batch
            
            # Wait between batches for larger deployments
            if num_parties > 10 and i + batch_size < len(remaining_parties):
                time.sleep(3)
        
        print(f"\nAll {num_parties} parties started")
    
    def _start_party(self, party_name: str):
        """Start a single party's process"""
        cmd = [
            sys.executable, "protocol.py",
            "--party", party_name,
            "--data-root", self.data_root_dir,
            "--column", self.column
        ]
        
        if self.limit > 0:
            cmd.extend(["--limit", str(self.limit)])
        
        log_file = open(Path(LOGS_DIR) / f"{party_name}_combined.log", "w")
        self.log_files[party_name] = log_file
        
        process = subprocess.Popen(
            cmd, stdout=log_file, stderr=log_file, 
            cwd=Path(__file__).parent
        )
        
        self.processes[party_name] = process
        print(f"Started {party_name.upper():<15} (PID: {process.pid})")
    
    def _wait_for_completion(self):
        """Wait for protocol completion or timeout"""
        print(f"\nWaiting for protocol completion...")
        
        if self.limit > 0:
            timeout = 300  # 5 minutes for limited runs
            print(f"Testing mode: {self.limit} dates, timeout: {timeout//60} minutes")
        else:
            timeout = 1800  # 30 minutes for full runs
            print(f"Full protocol: timeout: {timeout//60} minutes")
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            # Check for completion marker in data root directory
            completion_marker = Path(self.data_root_dir) / "protocol_complete.marker"
            if completion_marker.exists():
                print(f"\nProtocol completed successfully!")
                return
            
            # Check if any processes have died
            dead_parties = []
            for party_name, process in self.processes.items():
                if process.poll() is not None:
                    dead_parties.append(party_name)
            
            # If all processes have completed, check for completion marker one more time
            if len(dead_parties) == len(self.processes):
                time.sleep(2)  # Brief wait to ensure marker is written
                if completion_marker.exists():
                    print(f"\nAll parties completed - Protocol successful!")
                    return
                else:
                    print(f"\nAll parties exited but no completion marker found")
                    break
            elif dead_parties:
                # Some but not all parties failed - wait a bit more to see if it's graceful completion
                if len(dead_parties) >= len(self.processes) * 0.8:  # 80% or more completed
                    print(f"\nMost parties completed ({len(dead_parties)}/{len(self.processes)}), checking for completion...")
                    time.sleep(5)  # Give a bit more time for completion marker
                    if completion_marker.exists():
                        print(f"\nProtocol completed successfully!")
                        return
                
                print(f"\n⚠ Some parties failed: {', '.join(dead_parties)}")
                # Don't break immediately - keep checking for completion marker
            
            # Progress update every 30 seconds
            if int(time.time() - start_time) % 30 == 0:
                elapsed = int(time.time() - start_time)
                print(f"Running... ({elapsed//60}m {elapsed%60}s elapsed)")
            
            time.sleep(1)
        
        print(f"\nTimeout reached or processes failed")
    
    def _show_results(self):
        """Display protocol results if available"""
        # Look for results file in data root directory with new naming convention
        data_root_path = Path(self.data_root_dir)
        
        # Try to find the results file with the new naming pattern
        results_files = list(data_root_path.glob(f"Total_{self.column}.csv"))
        
        if results_files:
            results_file = results_files[0]  # Use first match
            print(f"\nResults saved to: {results_file}")
            try:
                import pandas as pd
                df = pd.read_csv(results_file)
                print(f"Processed {len(df)} dates")
                print(f"Sample results:")
                print(df.head().to_string(index=False))
            except ImportError:
                print("(Install pandas to see result preview)")
        else:
            # Fallback: check for old naming convention in results directory
            results_file = Path(RESULTS_DIR) / "smpc_results.csv"
            if results_file.exists():
                print(f"\nResults saved to: {results_file}")
                try:
                    import pandas as pd
                    df = pd.read_csv(results_file)
                    print(f"Processed {len(df)} dates")
                    print(f"Sample results:")
                    print(df.head().to_string(index=False))
                except ImportError:
                    print("(Install pandas to see result preview)")
            else:
                print(f"\nNo results file found")
    
    def _cleanup(self):
        """Clean up processes and files"""
        print(f"\nCleaning up...")
        
        # Terminate all processes
        for party_name, process in self.processes.items():
            if process.poll() is None:
                try:
                    process.terminate()
                    process.wait(timeout=5)
                except:
                    process.kill()
        
        # Close log files
        for log_file in self.log_files.values():
            log_file.close()
        
        print("Cleanup complete")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Run SMPC protocol across all parties",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-discover parties from current directory
  python run_smpc_protocol.py
  
  # Use specific data root directory
  python run_smpc_protocol.py --data-root /path/to/parties
  
  # Use legacy country configuration
  python run_smpc_protocol.py --legacy
        """
    )
    
    parser.add_argument(
        '--data-root', default='.',
        help='Root directory containing party subdirectories (default: current directory)'
    )
    parser.add_argument(
        '--limit', type=int, default=0,
        help='Limit number of dates for testing (0 = all dates)'
    )
    parser.add_argument(
        '--column', default='daily_cases',
        help='Data column to analyze (default: daily_cases)'
    )
    parser.add_argument(
        '--legacy', action='store_true',
        help='Use legacy country-based configuration instead of auto-discovery'
    )
    
    args = parser.parse_args()
    
    orchestrator = SMPCOrchestrator(args.data_root, args.limit, args.column, args.legacy)
    orchestrator.run_protocol()


if __name__ == "__main__":
    main()