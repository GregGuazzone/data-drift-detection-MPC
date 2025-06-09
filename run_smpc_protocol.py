import os
import signal
import subprocess
import time
import sys
import argparse
from pathlib import Path
import shutil

def main():
    # Add command-line argument parsing
    parser = argparse.ArgumentParser(description="Run SMPC protocol across multiple countries")
    parser.add_argument("--limit", type=int, default=0, 
                      help="Limit the number of dates to process (0 = process all)")
    args = parser.parse_args()
    
    # Configuration
    countries = ["china", "france", "germany", "iran", "italy", "spain", "united_kingdom", "us"]
    data_column = "daily_cases"
    results_dir = "./results"
    log_dir = "./logs"
    completion_marker = "key_countries/results/protocol_complete.marker"
    
    # Create directories
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Clean up any existing marker
    if os.path.exists(completion_marker):
        os.remove(completion_marker)
    
    # Store processes
    processes = {}
    
    def cleanup(sig=None, frame=None):
        print("\nShutting down all servers...")
        time.sleep(30)
        for country, process in processes.items():
            if process.poll() is None:  # If process is still running
                print(f"Stopping {country} (PID: {process.pid})...")
                try:
                    process.terminate()
                    # Give it a moment to terminate gracefully
                    time.sleep(0.5)
                    if process.poll() is None:
                        process.kill()
                except:
                    pass
        print("All servers stopped.")
        sys.exit(0)
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)
    
    # Kill existing Python processes
    print("Cleaning up existing processes...")
    try:
        subprocess.run(["pkill", "-f", "python.*\.py"], check=False)
        time.sleep(2)
    except:
        pass
    
    try:
        # Start China (dealer) first
        print("Starting China (dealer)...")
        china_cmd = [
            "python3", "protocol.py",
            "--country", "china",
            "--run-smpc", 
            "--all-dates",
            "--column", data_column,
            "--data-dir", "china"
        ]
        
        # Add limit parameter if specified
        if args.limit:
            china_cmd.extend(["--limit", str(args.limit)])
            
        china_log = open(os.path.join(log_dir, "china_combined.log"), "w")
        processes["china"] = subprocess.Popen(
            china_cmd, 
            stdout=china_log, 
            stderr=china_log, 
            cwd="key_countries"
        )
        print(f"Started China (PID: {processes['china'].pid})")
        time.sleep(5)  # Give dealer time to initialize
        
        # Start all other countries with the generic protocol
        for country in countries:
            if country != "china":
                print(f"Starting {country}...")
                country_cmd = [
                    "python3", "protocol.py",
                    "--country", country,
                    "--run-smpc", 
                    "--all-dates",
                    "--column", data_column,
                    "--data-dir", country
                ]
                
                # Add limit parameter if specified
                if args.limit:
                    country_cmd.extend(["--limit", str(args.limit)])
                
                country_log = open(os.path.join(log_dir, f"{country}_combined.log"), "w")
                processes[country] = subprocess.Popen(
                    country_cmd, 
                    stdout=country_log, 
                    stderr=country_log, 
                    cwd="key_countries"
                )
                
                print(f"Started {country} (PID: {processes[country].pid})")
                time.sleep(1)  # Small delay between country starts
            
        # Wait for completion marker with timeout
        start_time = time.time()
        max_wait = 600
        
        while not os.path.exists(completion_marker) and time.time() - start_time < max_wait:
            elapsed = int(time.time() - start_time)
            print(f"\rWaiting for completion... {elapsed}s elapsed", end="")
            
            # Check if any process has crashed
            for country, process in list(processes.items()):
                if process.poll() is not None:
                    print(f"\nWARNING: {country} process terminated early with exit code {process.poll()}")
                    # Could restart it here if needed
            
            time.sleep(3)
        
        print("\n")  # New line after wait loop
        
        # Check result file - updated for all-dates mode
        result_file = "key_countries/results/smpc_results_all_dates.csv"
        
        if os.path.exists(completion_marker):
            print("Protocol completed successfully!")
            print("--------------------------------------------")
            with open(completion_marker, 'r') as f:
                print(f.read())
            print("--------------------------------------------")
            
            if os.path.exists(result_file):
                print("Results summary:")
                try:
                    with open(result_file, 'r') as f:
                        lines = f.readlines()
                        print(f"Header: {lines[0].strip()}")
                        print(f"First date: {lines[1].strip() if len(lines) > 1 else 'No data'}")
                        if len(lines) > 2:
                            print(f"Last date: {lines[-1].strip()}")
                        print(f"Total dates processed: {len(lines)-1}")
                except Exception as e:
                    print(f"Error reading results file: {e}")
                
                # Copy to results directory
                shutil.copy(result_file, results_dir)
                print(f"Results saved to: {os.path.join(results_dir, os.path.basename(result_file))}")
                
                # Success! We can clean up
                cleanup()
            else:
                print("WARNING: Completion marker found but no results file.")
        else:
            print(f"Timeout waiting for protocol completion after {max_wait} seconds")
            print("Protocol may have failed or is taking longer than expected")
            
            if os.path.exists(result_file):
                print("Results file found despite timeout:")
                # Print summary of results file instead of whole file
                try:
                    with open(result_file, 'r') as f:
                        lines = f.readlines()
                        print(f"Header: {lines[0].strip()}")
                        print(f"Total dates processed: {len(lines)-1}")
                except Exception as e:
                    print(f"Error reading results file: {e}")
                    
                shutil.copy(result_file, results_dir)
            else:
                print("No result file found. Protocol may not have completed.")
                print("Checking logs for errors:")
                
                try:
                    all_logs = []
                    for log_file in os.listdir(log_dir):
                        if log_file.endswith("combined.log"):
                            log_path = os.path.join(log_dir, log_file)
                            with open(log_path, 'r') as f:
                                log_contents = f.read()
                                for line in log_contents.splitlines():
                                    if any(error in line.lower() for error in ["error", "exception", "failed"]):
                                        all_logs.append(f"{log_file}: {line}")
                    
                    # Print last 10 errors
                    for log_line in all_logs[-10:]:
                        print(log_line)
                except Exception as e:
                    print(f"Error checking logs: {e}")
        
        print("\nPress Ctrl+C to stop all servers and exit")
        
        # Keep script running until user presses Ctrl+C
        while True:
            time.sleep(1)
            
            # Check if processes are still alive
            for country, process in list(processes.items()):
                if process.poll() is not None:
                    print(f"{country} process terminated with exit code {process.poll()}")
                    del processes[country]
            
            if not processes:
                print("All processes have terminated. Exiting.")
                break
                
    except Exception as e:
        print(f"Error in protocol execution: {e}")
        cleanup()

if __name__ == "__main__":
    main()