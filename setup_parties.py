#!/usr/bin/env python3
"""
Setup script for dynamic SMPC parties
Creates party directories with data for testing
"""
import argparse
import os
import shutil
import csv
from pathlib import Path


def create_sample_parties(num_parties: int, data_root: str = "sample_parties"):
    """Create sample party directories for testing"""
    print(f"Creating {num_parties} sample party directories in {data_root}/...")
    
    root_path = Path(data_root)
    root_path.mkdir(exist_ok=True)
    
    # Sample data template
    sample_dates = ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05"]
    
    for i in range(num_parties):
        party_name = f"party_{i}"
        party_dir = root_path / party_name
        party_dir.mkdir(exist_ok=True)
        
        # Create sample CSV data (different values for each party)
        csv_file = party_dir / "daily_cases.csv"
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Date", "daily_cases", "daily_cases_7d_avg"])
            
            for j, date in enumerate(sample_dates):
                # Generate different values for each party
                base_value = (i + 1) * 10 + j * 5
                avg_value = base_value - 2
                writer.writerow([date, base_value, avg_value])
        
        print(f"  Created: {party_dir}/daily_cases.csv")
    
    print(f"\nSetup complete! Created {num_parties} parties in {data_root}/")
    print(f"To run SMPC protocol:")
    print(f"  python run_smpc_protocol.py --data-root {data_root}")


def copy_from_existing(source_dir: str, target_dir: str = "parties_copy"):
    """Copy existing party directories to new location"""
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    if not source_path.exists():
        print(f"Error: Source directory {source_dir} does not exist")
        return
    
    target_path.mkdir(exist_ok=True)
    
    # Find subdirectories with CSV files
    party_dirs = []
    for item in source_path.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            csv_files = list(item.glob("*.csv"))
            if csv_files:
                party_dirs.append(item)
    
    if not party_dirs:
        print(f"No directories with CSV files found in {source_dir}")
        return
    
    print(f"Copying {len(party_dirs)} party directories...")
    
    for party_dir in party_dirs:
        dest_dir = target_path / party_dir.name
        dest_dir.mkdir(exist_ok=True)
        
        # Copy all CSV files
        csv_files = list(party_dir.glob("*.csv"))
        for csv_file in csv_files:
            dest_file = dest_dir / csv_file.name
            shutil.copy2(csv_file, dest_file)
        
        print(f"  Copied: {party_dir.name} ({len(csv_files)} files)")
    
    print(f"\nCopy complete! Parties available in {target_dir}/")
    print(f"To run SMPC protocol:")
    print(f"  python run_smpc_protocol.py --data-root {target_dir}")


def list_parties(data_root: str = "."):
    """List discovered parties in a directory"""
    from config import discover_parties, reset_party_cache
    
    try:
        reset_party_cache()  # Reset cache to force re-discovery
        parties = discover_parties(data_root)
        
        print(f"Discovered parties in {data_root}:")
        print("=" * 40)
        
        for party_name, party_id in sorted(parties.items(), key=lambda x: x[1]):
            party_dir = Path(data_root) / party_name
            csv_files = list(party_dir.glob("*.csv"))
            print(f"  {party_id}: {party_name:<20} ({len(csv_files)} CSV files)")
        
        print(f"\nTotal: {len(parties)} parties")
        
        if parties:
            print(f"\nTo run SMPC protocol:")
            print(f"  python run_smpc_protocol.py --data-root {data_root}")
        
    except Exception as e:
        print(f"Error discovering parties: {e}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Setup and manage SMPC party directories',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create 5 sample parties
  python setup_parties.py --create 5
  
  # Copy existing party directories
  python setup_parties.py --copy-from /path/to/existing/parties
  
  # List parties in current directory
  python setup_parties.py --list
  
  # List parties in specific directory
  python setup_parties.py --list --data-root /path/to/parties
        """
    )
    
    parser.add_argument('--create', type=int, metavar='N',
                       help='Create N sample party directories')
    parser.add_argument('--copy-from', metavar='SOURCE_DIR',
                       help='Copy party directories from existing location')
    parser.add_argument('--list', action='store_true',
                       help='List discovered parties in directory')
    parser.add_argument('--data-root', default='.',
                       help='Root directory for party operations (default: current directory)')
    parser.add_argument('--output-dir', 
                       help='Output directory for created/copied parties')
    
    args = parser.parse_args()
    
    if args.create:
        output_dir = args.output_dir or "sample_parties"
        create_sample_parties(args.create, output_dir)
    elif args.copy_from:
        output_dir = args.output_dir or "parties_copy"
        copy_from_existing(args.copy_from, output_dir)
    elif args.list:
        list_parties(args.data_root)
    else:
        # Default: list parties in current directory
        list_parties(args.data_root)


if __name__ == '__main__':
    main()
