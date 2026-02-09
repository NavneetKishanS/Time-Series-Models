"""
Unified Pipeline Runner for AlternatingPipeline.

Executes the full pipeline in sequence:
1. Preprocess raw data -> extract exchange/examination sequences
2. Train Exchange model -> body region transitions
3. Train Examination model -> MRI event sequences
4. Generate buckets -> pre-compute 1000 samples per transition
5. Run day simulation -> generate a sample day schedule
6. Generate visualizations -> evaluate results with charts

Usage:
    python run_all.py                    # Run all steps
    python run_all.py --skip-preprocess  # Skip preprocessing (use existing data)
    python run_all.py --skip-training    # Skip training (use existing models)
    python run_all.py --steps 1,2,3      # Run specific steps only
"""
import os
import sys
import argparse
import time
from datetime import datetime

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    DATA_DIR, MODEL_SAVE_DIR, BUCKETS_DIR, OUTPUT_DIR,
    EXCLUDED_BODY_REGIONS, DURATION_MULTIPLIER
)


def step_1_preprocess():
    """Step 1: Preprocess raw CSV data."""
    print("\n" + "=" * 70)
    print("STEP 1: PREPROCESSING RAW DATA")
    print("=" * 70)

    from data.preprocessing import preprocess_all_data

    print(f"Input directory: {DATA_DIR}")
    result = preprocess_all_data(verbose=True)

    print(f"\nPreprocessing complete:")
    print(f"  Exchange sequences: {len(result['exchange'])}")
    print(f"  Examination sequences: {len(result['examination'])}")

    return result


def step_2_train_exchange():
    """Step 2: Train the Exchange Model."""
    print("\n" + "=" * 70)
    print("STEP 2: TRAINING EXCHANGE MODEL")
    print("=" * 70)

    from training.train_exchange import train_exchange_model

    model, history = train_exchange_model(verbose=True)

    print(f"\nExchange model training complete:")
    print(f"  Best validation loss: {min(history['val_loss']):.4f}")
    print(f"  Best validation accuracy: {max(history['val_acc']):.4f}")

    return model, history


def step_3_train_examination():
    """Step 3: Train the Examination Model."""
    print("\n" + "=" * 70)
    print("STEP 3: TRAINING EXAMINATION MODEL")
    print("=" * 70)

    from training.train_examination import train_examination_model

    model, history = train_examination_model(verbose=True)

    print(f"\nExamination model training complete:")
    print(f"  Best validation loss: {min(history['val_loss']):.4f}")
    print(f"  Best validation perplexity: {min(history['val_perplexity']):.2f}")

    return model, history


def step_4_generate_buckets():
    """Step 4: Generate all buckets using trained models."""
    print("\n" + "=" * 70)
    print("STEP 4: GENERATING BUCKETS")
    print("=" * 70)

    import torch
    from models.exchange_model import create_exchange_model
    from models.examination_model import create_examination_model
    from generation.bucket_generator import BucketGenerator
    from config import BUCKET_SIZE

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Duration multiplier: {DURATION_MULTIPLIER}x")

    # Load Exchange Model
    print("\nLoading Exchange Model...")
    exchange_model = create_exchange_model()
    exchange_path = os.path.join(MODEL_SAVE_DIR, 'exchange', 'exchange_model_best.pt')
    if not os.path.exists(exchange_path):
        exchange_path = os.path.join(MODEL_SAVE_DIR, 'exchange', 'exchange_model_final.pt')

    if os.path.exists(exchange_path):
        exchange_model.load_state_dict(torch.load(exchange_path, map_location=device, weights_only=True))
        print(f"  Loaded: {exchange_path}")
    else:
        raise FileNotFoundError(f"Exchange model not found at {exchange_path}")

    # Load Examination Model
    print("\nLoading Examination Model...")
    examination_model = create_examination_model()
    examination_path = os.path.join(MODEL_SAVE_DIR, 'examination', 'examination_model_best.pt')
    if not os.path.exists(examination_path):
        examination_path = os.path.join(MODEL_SAVE_DIR, 'examination', 'examination_model_final.pt')

    if os.path.exists(examination_path):
        examination_model.load_state_dict(torch.load(examination_path, map_location=device, weights_only=True))
        print(f"  Loaded: {examination_path}")
    else:
        raise FileNotFoundError(f"Examination model not found at {examination_path}")

    # Generate buckets
    print(f"\nGenerating buckets ({BUCKET_SIZE} samples each)...")
    if EXCLUDED_BODY_REGIONS:
        print(f"  Excluding body regions: {EXCLUDED_BODY_REGIONS}")

    generator = BucketGenerator(
        exchange_model=exchange_model,
        examination_model=examination_model,
        device=device
    )

    generator.generate_all_buckets(num_samples=BUCKET_SIZE)
    generator.save_buckets()

    print(f"\nBucket generation complete:")
    print(f"  Exchange buckets: {len(generator.exchange_buckets)}")
    print(f"  Examination buckets: {len(generator.examination_buckets)}")
    print(f"  Saved to: {BUCKETS_DIR}")

    return generator


def step_5_simulate_day():
    """Step 5: Run a day simulation."""
    print("\n" + "=" * 70)
    print("STEP 5: RUNNING DAY SIMULATION")
    print("=" * 70)

    from generation.day_simulator import DaySimulator
    from config import VALID_BODY_REGIONS
    import numpy as np

    # Load buckets
    print("Loading pre-generated buckets...")
    simulator = DaySimulator(buckets_dir=BUCKETS_DIR)

    if len(simulator.buckets.exchange_buckets) == 0:
        raise RuntimeError("No buckets loaded. Run step 4 first.")

    print(f"  Loaded {len(simulator.buckets.exchange_buckets)} exchange buckets")
    print(f"  Loaded {len(simulator.buckets.examination_buckets)} examination buckets")

    # Create ground truth patient sequence (using only valid body regions)
    print("\nCreating ground truth patient sequence...")
    np.random.seed(42)  # For reproducibility

    num_patients = 10
    ground_truth = []
    for i in range(num_patients):
        body_region = np.random.choice(VALID_BODY_REGIONS)
        ground_truth.append({
            'patient_id': f'PAT{i+1:03d}',
            'body_region': body_region
        })

    print(f"  Generated {num_patients} patients:")
    for p in ground_truth:
        print(f"    {p['patient_id']}: {p['body_region']}")

    # Simulate the day
    print(f"\nSimulating day...")
    start_time = datetime.now().replace(hour=8, minute=0, second=0, microsecond=0)
    schedule = simulator.simulate_day(ground_truth, start_time=start_time)

    # Save the schedule
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'simulated_day_{timestamp}.csv'
    output_path = simulator.save_schedule(schedule, filename)

    # Print statistics
    if schedule:
        total_duration_secs = schedule[-1]['cumulative_time'] - schedule[0]['timestamp']
        total_hours = total_duration_secs / 3600

        print(f"\nSimulation Results:")
        print(f"  Total events: {len(schedule)}")
        print(f"  Total duration: {total_hours:.2f} hours ({total_duration_secs:.0f} seconds)")
        print(f"  Average per patient: {total_duration_secs / num_patients / 60:.1f} minutes")
        print(f"  Output saved to: {output_path}")

    return schedule


def step_6_visualize(schedule=None):
    """Step 6: Generate visualizations for evaluation."""
    print("\n" + "=" * 70)
    print("STEP 6: GENERATING VISUALIZATIONS")
    print("=" * 70)

    import pandas as pd
    import numpy as np
    from glob import glob

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("Warning: plotly not installed. Skipping visualizations.")
        print("Install with: pip install plotly")
        return None

    # Load the most recent schedule if not provided
    if schedule is None:
        print("Loading most recent simulated schedule...")
        schedule_files = glob(os.path.join(OUTPUT_DIR, 'simulated_day_*.csv'))
        if not schedule_files:
            print("  No schedule files found. Run step 5 first.")
            return None
        latest_file = max(schedule_files, key=os.path.getmtime)
        print(f"  Loading: {latest_file}")
        schedule_df = pd.read_csv(latest_file)
    else:
        schedule_df = pd.DataFrame(schedule)

    print(f"  Loaded {len(schedule_df)} events")

    # Create output directory for visualizations
    viz_dir = os.path.join(OUTPUT_DIR, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)

    # =========================================================================
    # Visualization 1: Duration Distribution by Event Type
    # =========================================================================
    print("\nCreating duration distribution chart...")

    fig1 = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Exchange Durations", "Examination Durations")
    )

    exchange_events = schedule_df[schedule_df['event_type'] == 'exchange']
    exam_events = schedule_df[schedule_df['event_type'] == 'examination']

    if len(exchange_events) > 0:
        fig1.add_trace(
            go.Histogram(
                x=exchange_events['duration'],
                name='Exchange',
                marker_color='#636EFA',
                nbinsx=30
            ),
            row=1, col=1
        )

    if len(exam_events) > 0:
        fig1.add_trace(
            go.Histogram(
                x=exam_events['duration'],
                name='Examination',
                marker_color='#EF553B',
                nbinsx=30
            ),
            row=1, col=2
        )

    fig1.update_layout(
        title_text=f"Duration Distribution (Multiplier: {DURATION_MULTIPLIER}x)",
        showlegend=False
    )
    fig1.update_xaxes(title_text="Duration (seconds)", row=1, col=1)
    fig1.update_xaxes(title_text="Duration (seconds)", row=1, col=2)
    fig1.update_yaxes(title_text="Count", row=1, col=1)
    fig1.update_yaxes(title_text="Count", row=1, col=2)

    duration_path = os.path.join(viz_dir, 'duration_distribution.html')
    fig1.write_html(duration_path)
    print(f"  Saved: {duration_path}")

    # =========================================================================
    # Visualization 2: Body Region Distribution
    # =========================================================================
    print("Creating body region distribution chart...")

    # Count events per body region (for examination events)
    if 'body_region' in schedule_df.columns:
        body_counts = exam_events['body_region'].value_counts().reset_index()
        body_counts.columns = ['body_region', 'count']

        # Calculate average duration per body region
        avg_duration = exam_events.groupby('body_region')['duration'].mean().reset_index()
        avg_duration.columns = ['body_region', 'avg_duration']

        body_stats = pd.merge(body_counts, avg_duration, on='body_region')

        fig2 = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Event Count by Body Region", "Avg Duration by Body Region")
        )

        fig2.add_trace(
            go.Bar(
                x=body_stats['body_region'],
                y=body_stats['count'],
                name='Count',
                marker_color='#00CC96'
            ),
            row=1, col=1
        )

        fig2.add_trace(
            go.Bar(
                x=body_stats['body_region'],
                y=body_stats['avg_duration'],
                name='Avg Duration',
                marker_color='#AB63FA'
            ),
            row=1, col=2
        )

        fig2.update_layout(title_text="Body Region Analysis", showlegend=False)
        fig2.update_xaxes(title_text="Body Region", row=1, col=1)
        fig2.update_xaxes(title_text="Body Region", row=1, col=2)
        fig2.update_yaxes(title_text="Event Count", row=1, col=1)
        fig2.update_yaxes(title_text="Avg Duration (seconds)", row=1, col=2)

        body_region_path = os.path.join(viz_dir, 'body_region_analysis.html')
        fig2.write_html(body_region_path)
        print(f"  Saved: {body_region_path}")

    # =========================================================================
    # Visualization 3: Timeline / Gantt Chart
    # =========================================================================
    print("Creating timeline chart...")

    # Create a timeline showing patient flow
    if 'patient_id' in schedule_df.columns and 'timestamp' in schedule_df.columns:
        # Aggregate by patient
        patient_summary = schedule_df.groupby('patient_id').agg({
            'timestamp': 'min',
            'cumulative_time': 'max',
            'event_type': 'count'
        }).reset_index()
        patient_summary.columns = ['patient_id', 'start_time', 'end_time', 'num_events']
        patient_summary['duration'] = patient_summary['end_time'] - patient_summary['start_time']

        # Convert to minutes for readability
        patient_summary['start_min'] = patient_summary['start_time'] / 60
        patient_summary['duration_min'] = patient_summary['duration'] / 60

        fig3 = go.Figure()

        colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A',
                  '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']

        for i, row in patient_summary.iterrows():
            fig3.add_trace(go.Bar(
                x=[row['duration_min']],
                y=[row['patient_id']],
                base=[row['start_min']],
                orientation='h',
                name=row['patient_id'],
                marker_color=colors[i % len(colors)],
                text=f"{row['duration_min']:.1f} min",
                textposition='inside',
                hovertemplate=(
                    f"<b>{row['patient_id']}</b><br>"
                    f"Start: {row['start_min']:.1f} min<br>"
                    f"Duration: {row['duration_min']:.1f} min<br>"
                    f"Events: {row['num_events']}<extra></extra>"
                )
            ))

        fig3.update_layout(
            title_text="Patient Timeline (Gantt Chart)",
            xaxis_title="Time (minutes from start)",
            yaxis_title="Patient ID",
            showlegend=False,
            barmode='stack'
        )

        timeline_path = os.path.join(viz_dir, 'patient_timeline.html')
        fig3.write_html(timeline_path)
        print(f"  Saved: {timeline_path}")

    # =========================================================================
    # Visualization 4: Summary Statistics Dashboard
    # =========================================================================
    print("Creating summary dashboard...")

    # Calculate summary statistics
    total_duration_sec = schedule_df['duration'].sum()
    total_duration_min = total_duration_sec / 60
    total_duration_hr = total_duration_sec / 3600

    num_patients = schedule_df['patient_id'].nunique() if 'patient_id' in schedule_df.columns else 0
    num_exchanges = len(exchange_events)
    num_exams = len(exam_events)

    avg_exchange_duration = exchange_events['duration'].mean() if len(exchange_events) > 0 else 0
    avg_exam_duration = exam_events['duration'].mean() if len(exam_events) > 0 else 0
    avg_per_patient = total_duration_min / num_patients if num_patients > 0 else 0

    # Create summary figure
    fig4 = make_subplots(
        rows=2, cols=2,
        specs=[
            [{"type": "indicator"}, {"type": "indicator"}],
            [{"type": "indicator"}, {"type": "indicator"}]
        ],
        subplot_titles=("Total Duration", "Patients", "Avg Exchange", "Avg per Patient")
    )

    fig4.add_trace(
        go.Indicator(
            mode="number",
            value=total_duration_hr,
            number={"suffix": " hours", "valueformat": ".1f"},
            title={"text": "Total Duration"}
        ),
        row=1, col=1
    )

    fig4.add_trace(
        go.Indicator(
            mode="number",
            value=num_patients,
            title={"text": "Patients"}
        ),
        row=1, col=2
    )

    fig4.add_trace(
        go.Indicator(
            mode="number",
            value=avg_exchange_duration,
            number={"suffix": " sec", "valueformat": ".0f"},
            title={"text": "Avg Exchange Duration"}
        ),
        row=2, col=1
    )

    fig4.add_trace(
        go.Indicator(
            mode="number",
            value=avg_per_patient,
            number={"suffix": " min", "valueformat": ".1f"},
            title={"text": "Avg per Patient"}
        ),
        row=2, col=2
    )

    fig4.update_layout(title_text="Simulation Summary")

    summary_path = os.path.join(viz_dir, 'summary_dashboard.html')
    fig4.write_html(summary_path)
    print(f"  Saved: {summary_path}")

    # =========================================================================
    # Visualization 5: Event Type Breakdown Pie Chart
    # =========================================================================
    print("Creating event breakdown chart...")

    event_counts = schedule_df['event_type'].value_counts()

    fig5 = go.Figure(data=[go.Pie(
        labels=event_counts.index,
        values=event_counts.values,
        hole=0.4,
        marker_colors=['#636EFA', '#EF553B']
    )])

    fig5.update_layout(
        title_text="Event Type Breakdown",
        annotations=[dict(text='Events', x=0.5, y=0.5, font_size=20, showarrow=False)]
    )

    event_breakdown_path = os.path.join(viz_dir, 'event_breakdown.html')
    fig5.write_html(event_breakdown_path)
    print(f"  Saved: {event_breakdown_path}")

    # =========================================================================
    # Print Summary to Console
    # =========================================================================
    print("\n" + "-" * 50)
    print("SIMULATION SUMMARY")
    print("-" * 50)
    print(f"  Total Duration:      {total_duration_hr:.2f} hours ({total_duration_min:.0f} minutes)")
    print(f"  Number of Patients:  {num_patients}")
    print(f"  Exchange Events:     {num_exchanges}")
    print(f"  Examination Events:  {num_exams}")
    print(f"  Avg Exchange:        {avg_exchange_duration:.0f} seconds ({avg_exchange_duration/60:.1f} min)")
    print(f"  Avg Examination:     {avg_exam_duration:.0f} seconds ({avg_exam_duration/60:.1f} min)")
    print(f"  Avg per Patient:     {avg_per_patient:.1f} minutes")
    print("-" * 50)
    print(f"\nVisualizations saved to: {viz_dir}")

    return viz_dir


def step_7_customer_simulation(customer_id=None):
    """Step 7: Run per-customer day simulation."""
    print("\n" + "=" * 70)
    print("STEP 7: PER-CUSTOMER DAY SIMULATION")
    print("=" * 70)

    from generation.customer_simulator import CustomerSimulator
    from config import CUSTOMER_OUTPUT_DIR

    print("Initializing customer simulator...")
    simulator = CustomerSimulator()

    customers = simulator.list_customers()
    print(f"Available customers: {len(customers)}")

    if not customers:
        print("No customer schedules found. Run step 1 (preprocessing) first.")
        return None

    if customer_id and customer_id != 'all':
        # Simulate specific customer
        if customer_id not in customers:
            print(f"Customer '{customer_id}' not found. Available: {customers}")
            return None
        result = simulator.simulate_customer_day(customer_id, verbose=True)
        if result and result['schedule']:
            import pandas as pd
            customer_dir = os.path.join(CUSTOMER_OUTPUT_DIR, customer_id)
            os.makedirs(customer_dir, exist_ok=True)
            df = pd.DataFrame(result['schedule'])
            csv_path = os.path.join(customer_dir, f"simulated_{result['date']}.csv")
            df.to_csv(csv_path, index=False)
            print(f"\nSaved to: {csv_path}")
        return {customer_id: result}
    else:
        # Simulate all customers
        results = simulator.simulate_all_customers(
            verbose=True,
            output_dir=CUSTOMER_OUTPUT_DIR
        )
        return results


def main():
    parser = argparse.ArgumentParser(
        description="Run the AlternatingPipeline end-to-end",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Steps:
  1. Preprocess raw data
  2. Train Exchange model
  3. Train Examination model
  4. Generate buckets
  5. Run day simulation
  6. Generate visualizations
  7. Per-customer day simulation

Examples:
  python run_all.py                    # Run steps 1-6
  python run_all.py --skip-preprocess  # Skip step 1
  python run_all.py --skip-training    # Skip steps 2 and 3
  python run_all.py --steps 4,5,6      # Only run steps 4, 5, and 6
  python run_all.py --steps 7          # Only run customer simulation
  python run_all.py --customer 141049  # Simulate a specific customer
  python run_all.py --customer all     # Simulate all customers
        """
    )

    parser.add_argument('--skip-preprocess', action='store_true',
                        help='Skip preprocessing (use existing preprocessed data)')
    parser.add_argument('--skip-training', action='store_true',
                        help='Skip training (use existing trained models)')
    parser.add_argument('--no-viz', action='store_true',
                        help='Skip visualization generation')
    parser.add_argument('--steps', type=str, default=None,
                        help='Comma-separated list of steps to run (e.g., "1,2,3")')
    parser.add_argument('--customer', type=str, default=None,
                        help='Customer ID to simulate (or "all" for all customers)')

    args = parser.parse_args()

    # Determine which steps to run
    if args.customer:
        # --customer implies step 7 only (unless --steps also specified)
        if args.steps:
            steps_to_run = set(int(s.strip()) for s in args.steps.split(','))
        else:
            steps_to_run = {7}
    elif args.steps:
        steps_to_run = set(int(s.strip()) for s in args.steps.split(','))
    else:
        steps_to_run = {1, 2, 3, 4, 5, 6}
        if args.skip_preprocess:
            steps_to_run.discard(1)
        if args.skip_training:
            steps_to_run.discard(2)
            steps_to_run.discard(3)
        if args.no_viz:
            steps_to_run.discard(6)

    print("=" * 70)
    print("ALTERNATING PIPELINE - FULL EXECUTION")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Steps to run: {sorted(steps_to_run)}")
    print(f"Duration multiplier: {DURATION_MULTIPLIER}x (corrects for time compression)")
    if EXCLUDED_BODY_REGIONS:
        print(f"Excluded body regions: {EXCLUDED_BODY_REGIONS}")

    start_time = time.time()

    schedule = None  # Will hold the schedule for visualization

    try:
        if 1 in steps_to_run:
            step_1_preprocess()

        if 2 in steps_to_run:
            step_2_train_exchange()

        if 3 in steps_to_run:
            step_3_train_examination()

        if 4 in steps_to_run:
            step_4_generate_buckets()

        if 5 in steps_to_run:
            schedule = step_5_simulate_day()

        if 6 in steps_to_run:
            step_6_visualize(schedule)

        if 7 in steps_to_run:
            step_7_customer_simulation(customer_id=args.customer)

        elapsed = time.time() - start_time
        print("\n" + "=" * 70)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print(f"Total time: {elapsed / 60:.1f} minutes")

    except Exception as e:
        elapsed = time.time() - start_time
        print("\n" + "=" * 70)
        print("PIPELINE FAILED")
        print("=" * 70)
        print(f"Error: {e}")
        print(f"Time elapsed: {elapsed / 60:.1f} minutes")
        raise


if __name__ == "__main__":
    main()
