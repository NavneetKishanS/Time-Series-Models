"""
Script to run a full day simulation using pre-generated buckets.
"""
import os
import sys
from datetime import datetime

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from generation.day_simulator import DaySimulator
from config import BUCKETS_DIR

def main():
    print("Initializing Day Simulation...")
    
    # 1. Check if buckets exist
    if not os.path.exists(BUCKETS_DIR):
        print(f"Error: Buckets directory not found at {BUCKETS_DIR}")
        print("Please run generate_buckets.py first.")
        return

    # 2. Initialize Simulator and Load Buckets
    print("Loading buckets...")
    simulator = DaySimulator(buckets_dir=BUCKETS_DIR)
    
    if len(simulator.buckets.exchange_buckets) == 0:
        print("Error: No buckets loaded. Check if the buckets directory is empty.")
        return

    # 3. Create a realistic patient sequence for a day
    # Typically 10-20 patients in a 12-hour shift
    print("\nCreating ground truth patient sequence...")
    # We'll create a specific sequence to see different transitions
    # Note: CHEST is excluded as it's not present in the training data
    ground_truth = [
        {'patient_id': 'PAT001', 'body_region': 'HEAD'},
        {'patient_id': 'PAT002', 'body_region': 'HEAD'},
        {'patient_id': 'PAT003', 'body_region': 'SPINE'},
        {'patient_id': 'PAT004', 'body_region': 'ABDOMEN'},
        {'patient_id': 'PAT005', 'body_region': 'PELVIS'},
        {'patient_id': 'PAT006', 'body_region': 'HEAD'},
        {'patient_id': 'PAT007', 'body_region': 'LEG'},
        {'patient_id': 'PAT008', 'body_region': 'NECK'},
    ]
    
    for i, p in enumerate(ground_truth):
        print(f"  Patient {i+1}: {p['patient_id']} - {p['body_region']}")

    # 4. Simulate the day
    print(f"\nSimulating day with {len(ground_truth)} patients...")
    start_time = datetime.now().replace(hour=8, minute=0, second=0, microsecond=0)
    schedule = simulator.simulate_day(ground_truth, start_time=start_time)

    # 5. Save the schedule
    print("\nSaving generated schedule...")
    output_path = simulator.save_schedule(schedule, 'simulated_day_output.csv')
    
    # 6. Print some statistics
    if schedule:
        total_duration_secs = schedule[-1]['cumulative_time'] - schedule[0]['timestamp']
        total_hours = total_duration_secs / 3600
        print(f"\nSimulation Results:")
        print(f"  Total events generated: {len(schedule)}")
        print(f"  Total duration: {total_hours:.2f} hours")
        print(f"  Schedule saved to: {output_path}")
    
    print("\nDay simulation complete!")

if __name__ == "__main__":
    main()
