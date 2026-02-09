"""
Customer Simulator: Simulate a day for each customer (MRI scanner).

Each raw CSV file represents one customer/scanner. This module uses real
daily patient sequences from each customer as ground truth, then runs
the day simulator to generate a synthetic version for comparison.
"""
import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    BUCKETS_DIR, OUTPUT_DIR, BODY_REGIONS, BODY_REGION_TO_ID
)
from generation.day_simulator import DaySimulator
from validation.metrics import compare_real_vs_predicted, print_comparison_report


class CustomerSimulator:
    """
    Simulates days for specific customers using their real patient sequences.

    Usage:
        simulator = CustomerSimulator()
        schedule = simulator.simulate_customer_day('141049')
        simulator.simulate_all_customers()
    """

    def __init__(self, buckets_dir=None, customer_schedules=None,
                 preprocessed_path=None):
        """
        Initialize the customer simulator.

        Args:
            buckets_dir: Directory containing pre-generated buckets
            customer_schedules: Dict of {customer_id: {date: [patients]}}
            preprocessed_path: Path to preprocessed_data.pkl (loads customer_schedules from it)
        """
        # Load buckets via DaySimulator
        self.day_simulator = DaySimulator(buckets_dir=buckets_dir or BUCKETS_DIR)

        # Load customer schedules
        if customer_schedules is not None:
            self.customer_schedules = customer_schedules
        else:
            self.customer_schedules = self._load_customer_schedules(preprocessed_path)

    def _load_customer_schedules(self, preprocessed_path=None):
        """Load customer schedules from preprocessed data."""
        if preprocessed_path is None:
            preprocessed_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'data', 'preprocessed', 'preprocessed_data.pkl'
            )

        with open(preprocessed_path, 'rb') as f:
            data = pickle.load(f)

        schedules = data.get('customer_schedules', {})
        if not schedules:
            print("Warning: No customer_schedules found in preprocessed data.")
            print("Re-run preprocessing (step 1) to generate them.")
        return schedules

    def list_customers(self):
        """
        List all available customer IDs.

        Returns:
            List of customer ID strings (scanner serial numbers)
        """
        return sorted(self.customer_schedules.keys())

    def list_days(self, customer_id):
        """
        List available days for a customer.

        Args:
            customer_id: Customer/scanner ID string

        Returns:
            List of date strings sorted chronologically
        """
        if customer_id not in self.customer_schedules:
            print(f"Customer '{customer_id}' not found.")
            return []
        return sorted(self.customer_schedules[customer_id].keys())

    def get_real_day(self, customer_id, date=None):
        """
        Get a real day's patient sequence for a customer.

        Args:
            customer_id: Customer/scanner ID string
            date: Specific date string, or None for a random day

        Returns:
            Tuple of (date_str, patient_list) or (None, None) if not found
        """
        if customer_id not in self.customer_schedules:
            print(f"Customer '{customer_id}' not found.")
            return None, None

        days = self.customer_schedules[customer_id]
        if not days:
            print(f"No daily schedules for customer '{customer_id}'.")
            return None, None

        if date is not None:
            if date not in days:
                print(f"Date '{date}' not found for customer '{customer_id}'.")
                return None, None
            return date, days[date]

        # Pick a random day
        date_str = np.random.choice(list(days.keys()))
        return date_str, days[date_str]

    def simulate_customer_day(self, customer_id, date=None, verbose=True):
        """
        Simulate a day for a customer using their real patient sequence.

        Args:
            customer_id: Customer/scanner ID string
            date: Specific date string, or None for random day
            verbose: Print progress

        Returns:
            Dict with 'date', 'customer_id', 'ground_truth', 'schedule'
        """
        date_str, patients = self.get_real_day(customer_id, date)
        if patients is None:
            return None

        if verbose:
            print(f"\nSimulating day for customer {customer_id}, date {date_str}")
            print(f"  {len(patients)} patients in ground truth:")
            for i, p in enumerate(patients[:10]):
                print(f"    {i+1}. {p['body_region']} (age={p['age']:.0f})")
            if len(patients) > 10:
                print(f"    ... and {len(patients) - 10} more")

        # Build ground truth in the format DaySimulator expects
        ground_truth = []
        for p in patients:
            ground_truth.append({
                'patient_id': p['patient_id'],
                'body_region': p['body_region'],
                'age': p.get('age', 50),
                'weight': p.get('weight', 75),
                'height': p.get('height', 1.75),
                'direction': p.get('direction', 'Head First'),
            })

        # Simulate the day
        start_time = datetime.now().replace(
            hour=p.get('hour_of_day', 8) if patients else 8,
            minute=0, second=0, microsecond=0
        )
        # Use first patient's hour as start time
        if patients and 'hour_of_day' in patients[0]:
            start_time = start_time.replace(hour=patients[0]['hour_of_day'])

        schedule = self.day_simulator.simulate_day(ground_truth, start_time=start_time)

        if verbose and schedule:
            total_secs = schedule[-1]['cumulative_time']
            print(f"  Generated {len(schedule)} events, "
                  f"total duration: {total_secs/3600:.2f} hours")

        return {
            'customer_id': customer_id,
            'date': date_str,
            'ground_truth': ground_truth,
            'schedule': schedule,
        }

    def simulate_all_customers(self, verbose=True, output_dir=None):
        """
        Simulate one day per customer.

        Args:
            verbose: Print progress
            output_dir: Directory to save per-customer outputs

        Returns:
            Dict of {customer_id: simulation_result}
        """
        if output_dir is None:
            output_dir = os.path.join(OUTPUT_DIR, 'customers')

        os.makedirs(output_dir, exist_ok=True)
        results = {}

        customers = self.list_customers()
        if verbose:
            print(f"\nSimulating days for {len(customers)} customers...")

        for i, customer_id in enumerate(customers):
            if verbose:
                print(f"\n[{i+1}/{len(customers)}] Customer {customer_id}")

            result = self.simulate_customer_day(customer_id, verbose=verbose)
            if result is None:
                continue

            results[customer_id] = result

            # Save per-customer schedule
            customer_dir = os.path.join(output_dir, customer_id)
            os.makedirs(customer_dir, exist_ok=True)

            if result['schedule']:
                df = pd.DataFrame(result['schedule'])
                csv_path = os.path.join(
                    customer_dir,
                    f"simulated_{result['date']}.csv"
                )
                df.to_csv(csv_path, index=False)

        if verbose:
            print(f"\n{'='*60}")
            print(f"Simulated {len(results)} customers")
            print(f"Outputs saved to: {output_dir}")

            # Print summary table
            print(f"\n{'Customer':<12} {'Date':<12} {'Patients':<10} "
                  f"{'Events':<10} {'Duration':<12}")
            print("-" * 56)
            for cid, r in sorted(results.items()):
                n_patients = len(r['ground_truth'])
                n_events = len(r['schedule']) if r['schedule'] else 0
                total_secs = (r['schedule'][-1]['cumulative_time']
                              if r['schedule'] else 0)
                print(f"{cid:<12} {r['date']:<12} {n_patients:<10} "
                      f"{n_events:<10} {total_secs/60:.1f} min")

        return results


if __name__ == "__main__":
    print("Customer Simulator")
    print("=" * 60)

    try:
        sim = CustomerSimulator()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Run preprocessing first: python run_all.py --steps 1")
        sys.exit(1)

    customers = sim.list_customers()
    print(f"\nAvailable customers: {len(customers)}")
    for cid in customers[:5]:
        days = sim.list_days(cid)
        print(f"  {cid}: {len(days)} days")

    if customers:
        print(f"\nSimulating a day for first customer ({customers[0]})...")
        result = sim.simulate_customer_day(customers[0])
        if result and result['schedule']:
            print(f"\nDone! Generated {len(result['schedule'])} events")
