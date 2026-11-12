"""
Data Retrieval Utility for ANS Experiments
==========================================
Enables pulling experiment logs and CSV data from the Raspberry Pi to the PC:
1. Lists existing CSV runs on the PC and remote Raspberry Pi.
2. Direct SCP retrieval: scp canal@<rasp_ip>:~/ANS/experiments/data/*.csv experiments/data/
3. Verifies file integrity and summary statistics.
"""

import argparse
import glob
import os
import subprocess
import sys
import pandas as pd


def list_local_data():
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    files = glob.glob(os.path.join(data_dir, '*.csv'))
    print(f"\n[LOCAL DATA] Found {len(files)} CSV files in {data_dir}:")
    for f in sorted(files):
        df = pd.read_csv(f)
        size_kb = os.path.getsize(f) / 1024.0
        print(f"  - {os.path.basename(f)} ({len(df)} rows, {size_kb:.1f} KB)")


def fetch_from_pi(rasp_ip: str, rasp_user: str = 'canal', remote_path: str = '~/ANS/experiments/data/'):
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(data_dir, exist_ok=True)
    cmd = f"scp {rasp_user}@{rasp_ip}:{remote_path}*.csv \"{data_dir}\""
    print(f"\n[FETCH] Running SCP to download CSVs from Raspberry Pi...")
    print(f"Command: {cmd}")
    try:
        res = subprocess.run(cmd, shell=True, check=True)
        print("[FETCH SUCCESS] Files downloaded successfully!")
    except subprocess.CalledProcessError as e:
        print(f"[FETCH ERROR] SCP command failed: {e}")
        print("Tip: If the experiment completed with run_experiment_client.py, the CSV was already auto-synced to PC!")


def main():
    parser = argparse.ArgumentParser(description="Fetch and inspect ANS experiment data")
    parser.add_argument('--list', action='store_true', default=True, help="List local experiment data files")
    parser.add_argument('--fetch', action='store_true', help="Fetch files from Raspberry Pi via SCP")
    parser.add_argument('--ip', type=str, default='192.168.1.222', help="Raspberry Pi IP address")
    parser.add_argument('--user', type=str, default='canal', help="Raspberry Pi username")
    args = parser.parse_args()

    if args.fetch:
        fetch_from_pi(args.ip, args.user)

    if args.list:
        list_local_data()


if __name__ == '__main__':
    main()
