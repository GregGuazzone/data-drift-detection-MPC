"""
Orchestrates the SMPC protocol across multiple countries
"""
import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

from config import COUNTRIES, LOGS_DIR, RESULTS_DIR


class SMPCOrchestrator:
    """Orchestrates SMPC protocol execution across all countries"""
    
    def __init__(self, limit: int = 0, column: str = 'daily_cases'):
        self.limit = limit
        self.column = column
        self.processes: Dict[str, subprocess.Popen] = {}
        self.log_files = {}
        
        # Setup directories
        for directory in [LOGS_DIR, RESULTS_DIR]:
            os.makedirs(directory, exist_ok=True)
    
    def run_protocol(self):
        """Run the complete SMPC protocol"""
        try:
            self._cleanup_previous_run()
            self._start_all_countries()
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
        completion_marker = Path(RESULTS_DIR) / "protocol_complete.marker"
        if completion_marker.exists():
            completion_marker.unlink()
    
    def _start_all_countries(self):
        """Start all country processes"""
        print("Starting SMPC Protocol")
        print("=" * 50)
        
        # Start dealer (China) first
        self._start_country('china')
        time.sleep(5)  # Give dealer time to initialize
        
        # Start all other countries
        for country in COUNTRIES:
            if country != 'china':
                self._start_country(country)
                time.sleep(1)  # Stagger starts
        
        print(f"\nAll {len(COUNTRIES)} countries started")
    
    def _start_country(self, country: str):
        """Start a single country's process"""
        cmd = [
            sys.executable, "protocol.py",
            "--country", country,
            "--column", self.column
        ]
        
        if self.limit > 0:
            cmd.extend(["--limit", str(self.limit)])
        
        log_file = open(Path(LOGS_DIR) / f"{country}_combined.log", "w")
        self.log_files[country] = log_file
        
        process = subprocess.Popen(
            cmd, stdout=log_file, stderr=log_file, 
            cwd=Path(__file__).parent
        )
        
        self.processes[country] = process
        print(f"Started {country.upper():<15} (PID: {process.pid})")
    
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
            # Check for completion marker
            completion_marker = Path(RESULTS_DIR) / "protocol_complete.marker"
            if completion_marker.exists():
                print(f"\n✓ Protocol completed successfully!")
                return
            
            # Check if any processes have died
            dead_countries = []
            for country, process in self.processes.items():
                if process.poll() is not None:
                    dead_countries.append(country)
            
            if dead_countries:
                print(f"\n✗ Countries failed: {', '.join(dead_countries)}")
                break
            
            # Progress update every 30 seconds
            if int(time.time() - start_time) % 30 == 0:
                elapsed = int(time.time() - start_time)
                print(f"  Running... ({elapsed//60}m {elapsed%60}s elapsed)")
            
            time.sleep(1)
        
        print(f"\n⚠ Timeout reached or processes failed")
    
    def _show_results(self):
        """Display protocol results if available"""
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
        for country, process in self.processes.items():
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
        description="Run SMPC protocol across all countries",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_smpc_protocol.py --limit 5          # Test with 5 dates
  python run_smpc_protocol.py --column deaths    # Use deaths column
  python run_smpc_protocol.py                    # Full protocol
        """
    )
    
    parser.add_argument(
        '--limit', type=int, default=0,
        help='Limit number of dates for testing (0 = all dates)'
    )
    parser.add_argument(
        '--column', default='daily_cases',
        help='Data column to analyze (default: daily_cases)'
    )
    
    args = parser.parse_args()
    
    orchestrator = SMPCOrchestrator(args.limit, args.column)
    orchestrator.run_protocol()


if __name__ == "__main__":
    main()