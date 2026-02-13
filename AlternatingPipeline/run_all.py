"""
Unified Pipeline Runner for AlternatingPipeline.

Executes the full pipeline in sequence:
1. Preprocess raw data -> extract exchange/examination sequences (with phase_type)
2. Train Exchange model -> unified Transformer for exchange sequences
3. Train Examination model -> unified Transformer for examination sequences
4. Run day simulation -> on-the-fly generation (no buckets)
5. Generate visualizations -> evaluate results with charts
6. Per-customer day simulation (generates individual customer visualizations)
7. General visualizations (aggregates customer data)

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
    DATA_DIR, MODEL_SAVE_DIR, OUTPUT_DIR,
    EXCLUDED_BODY_REGIONS, DURATION_MULTIPLIER
)


def _load_models(device):
    """
    Load trained exchange and examination models.

    Args:
        device: torch device

    Returns:
        (exchange_model, examination_model)
    """
    import torch
    from models.exchange_model import create_exchange_model
    from models.examination_model import create_examination_model

    # Load Exchange Model
    print("Loading Exchange Model...")
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
    print("Loading Examination Model...")
    examination_model = create_examination_model()
    examination_path = os.path.join(MODEL_SAVE_DIR, 'examination', 'examination_model_best.pt')
    if not os.path.exists(examination_path):
        examination_path = os.path.join(MODEL_SAVE_DIR, 'examination', 'examination_model_final.pt')

    if os.path.exists(examination_path):
        examination_model.load_state_dict(torch.load(examination_path, map_location=device, weights_only=True))
        print(f"  Loaded: {examination_path}")
    else:
        raise FileNotFoundError(f"Examination model not found at {examination_path}")

    return exchange_model, examination_model


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
    print(f"  Best validation perplexity: {min(history['val_perplexity']):.2f}")

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


def step_4_simulate_day():
    """Step 4: Run a day simulation with on-the-fly generation."""
    print("\n" + "=" * 70)
    print("STEP 4: RUNNING DAY SIMULATION (ON-THE-FLY)")
    print("=" * 70)

    import torch
    import numpy as np
    from generation.day_simulator import DaySimulator
    from config import VALID_BODY_REGIONS

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    exchange_model, examination_model = _load_models(device)
    simulator = DaySimulator(exchange_model, examination_model, device)

    # Create ground truth patient sequence
    print("\nCreating ground truth patient sequence...")
    np.random.seed(42)

    num_patients = 10
    ground_truth = []
    for i in range(num_patients):
        body_region = np.random.choice(VALID_BODY_REGIONS)
        ground_truth.append({
            'patient_id': f'PAT{i+1:03d}',
            'body_region': body_region,
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


def _generate_visualizations_for_dataframe(schedule_df, output_viz_dir, title_prefix=""):
    """
    Helper function to generate a set of standard visualizations for a given schedule DataFrame.
    """
    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from config import DURATION_MULTIPLIER

    os.makedirs(output_viz_dir, exist_ok=True)

    exchange_events = schedule_df[schedule_df['event_type'] == 'exchange']
    exam_events = schedule_df[schedule_df['event_type'] == 'examination']

    # Duration Distribution
    fig1 = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Exchange Durations", "Examination Durations")
    )

    if len(exchange_events) > 0:
        fig1.add_trace(
            go.Histogram(x=exchange_events['duration'], name='Exchange',
                        marker_color='#636EFA', nbinsx=30),
            row=1, col=1
        )

    if len(exam_events) > 0:
        fig1.add_trace(
            go.Histogram(x=exam_events['duration'], name='Examination',
                        marker_color='#EF553B', nbinsx=30),
            row=1, col=2
        )

    fig1.update_layout(
        title_text=f"{title_prefix}Duration Distribution (Multiplier: {DURATION_MULTIPLIER}x)",
        showlegend=False
    )
    fig1.update_xaxes(title_text="Duration (seconds)", row=1, col=1)
    fig1.update_xaxes(title_text="Duration (seconds)", row=1, col=2)
    fig1.update_yaxes(title_text="Count", row=1, col=1)
    fig1.update_yaxes(title_text="Count", row=1, col=2)

    duration_path = os.path.join(output_viz_dir, 'duration_distribution.html')
    fig1.write_html(duration_path)
    print(f"  Saved: {duration_path}")

    # Body Region Distribution
    if 'body_region' in schedule_df.columns:
        body_counts = exam_events['body_region'].value_counts().reset_index()
        body_counts.columns = ['body_region', 'count']

        avg_duration = exam_events.groupby('body_region')['duration'].mean().reset_index()
        avg_duration.columns = ['body_region', 'avg_duration']

        body_stats = pd.merge(body_counts, avg_duration, on='body_region')

        fig2 = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Event Count by Body Region", "Avg Duration by Body Region")
        )

        fig2.add_trace(
            go.Bar(x=body_stats['body_region'], y=body_stats['count'],
                   name='Count', marker_color='#00CC96'),
            row=1, col=1
        )

        fig2.add_trace(
            go.Bar(x=body_stats['body_region'], y=body_stats['avg_duration'],
                   name='Avg Duration', marker_color='#AB63FA'),
            row=1, col=2
        )

        fig2.update_layout(title_text=f"{title_prefix}Body Region Analysis", showlegend=False)
        body_region_path = os.path.join(output_viz_dir, 'body_region_analysis.html')
        fig2.write_html(body_region_path)
        print(f"  Saved: {body_region_path}")

    # Timeline / Gantt Chart
    if 'patient_id' in schedule_df.columns and 'timestamp' in schedule_df.columns:
        patient_summary = schedule_df.groupby('patient_id').agg({
            'timestamp': 'min',
            'cumulative_time': 'max',
            'event_type': 'count'
        }).reset_index()
        patient_summary.columns = ['patient_id', 'start_time', 'end_time', 'num_events']
        patient_summary['duration'] = patient_summary['end_time'] - patient_summary['start_time']
        patient_summary['start_min'] = patient_summary['start_time'] / 60
        patient_summary['duration_min'] = patient_summary['duration'] / 60

        fig3 = go.Figure()
        colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A',
                  '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']

        for i, row in patient_summary.iterrows():
            fig3.add_trace(go.Bar(
                x=[row['duration_min']], y=[row['patient_id']],
                base=[row['start_min']], orientation='h',
                name=row['patient_id'], marker_color=colors[i % len(colors)],
                text=f"{row['duration_min']:.1f} min", textposition='inside',
                hovertemplate=(
                    f"<b>{row['patient_id']}</b><br>"
                    f"Start: {row['start_min']:.1f} min<br>"
                    f"Duration: {row['duration_min']:.1f} min<br>"
                    f"Events: {row['num_events']}<extra></extra>"
                )
            ))

        fig3.update_layout(
            title_text=f"{title_prefix}Patient Timeline (Gantt Chart)",
            xaxis_title="Time (minutes from start)",
            yaxis_title="Patient ID",
            showlegend=False, barmode='stack'
        )

        timeline_path = os.path.join(output_viz_dir, 'patient_timeline.html')
        fig3.write_html(timeline_path)
        print(f"  Saved: {timeline_path}")

    # Summary Statistics
    total_duration_sec = schedule_df['duration'].sum()
    total_duration_min = total_duration_sec / 60
    total_duration_hr = total_duration_sec / 3600

    num_patients = schedule_df['patient_id'].nunique() if 'patient_id' in schedule_df.columns else 0
    num_exchanges = len(exchange_events)
    num_exams = len(exam_events)

    avg_exchange_duration = exchange_events['duration'].mean() if len(exchange_events) > 0 else 0
    avg_exam_duration = exam_events['duration'].mean() if len(exam_events) > 0 else 0
    avg_per_patient = total_duration_min / num_patients if num_patients > 0 else 0

    fig4 = make_subplots(
        rows=2, cols=2,
        specs=[[{"type": "indicator"}, {"type": "indicator"}],
               [{"type": "indicator"}, {"type": "indicator"}]],
        subplot_titles=("Total Duration", "Patients", "Avg Exchange", "Avg per Patient")
    )
    fig4.add_trace(go.Indicator(mode="number", value=total_duration_hr,
                                number={"suffix": " hours", "valueformat": ".1f"},
                                title={"text": "Total Duration"}), row=1, col=1)
    fig4.add_trace(go.Indicator(mode="number", value=num_patients,
                                title={"text": "Patients"}), row=1, col=2)
    fig4.add_trace(go.Indicator(mode="number", value=avg_exchange_duration,
                                number={"suffix": " sec", "valueformat": ".0f"},
                                title={"text": "Avg Exchange Duration"}), row=2, col=1)
    fig4.add_trace(go.Indicator(mode="number", value=avg_per_patient,
                                number={"suffix": " min", "valueformat": ".1f"},
                                title={"text": "Avg per Patient"}), row=2, col=2)

    fig4.update_layout(title_text=f"{title_prefix}Simulation Summary")
    summary_path = os.path.join(output_viz_dir, 'summary_dashboard.html')
    fig4.write_html(summary_path)
    print(f"  Saved: {summary_path}")

    # Event Breakdown Pie
    event_counts = schedule_df['event_type'].value_counts()
    fig5 = go.Figure(data=[go.Pie(
        labels=event_counts.index, values=event_counts.values,
        hole=0.4, marker_colors=['#636EFA', '#EF553B']
    )])
    fig5.update_layout(
        title_text=f"{title_prefix}Event Type Breakdown",
        annotations=[dict(text='Events', x=0.5, y=0.5, font_size=20, showarrow=False)]
    )
    event_breakdown_path = os.path.join(output_viz_dir, 'event_breakdown.html')
    fig5.write_html(event_breakdown_path)
    print(f"  Saved: {event_breakdown_path}")

    # Console Summary
    print("\n" + "-" * 50)
    print(f"{title_prefix} SIMULATION SUMMARY")
    print("-" * 50)
    print(f"  Total Duration:      {total_duration_hr:.2f} hours ({total_duration_min:.0f} minutes)")
    print(f"  Number of Patients:  {num_patients}")
    print(f"  Exchange Events:     {num_exchanges}")
    print(f"  Examination Events:  {num_exams}")
    print(f"  Avg Exchange:        {avg_exchange_duration:.0f} seconds ({avg_exchange_duration/60:.1f} min)")
    print(f"  Avg Examination:     {avg_exam_duration:.0f} seconds ({avg_exam_duration/60:.1f} min)")
    print(f"  Avg per Patient:     {avg_per_patient:.1f} minutes")
    print("-" * 50)

    summary_metrics = {
        'total_duration_hr': total_duration_hr,
        'total_duration_min': total_duration_min,
        'num_patients': num_patients,
        'num_exchanges': num_exchanges,
        'num_exams': num_exams,
        'avg_exchange_duration_sec': avg_exchange_duration,
        'avg_exchange_duration_min': avg_exchange_duration / 60,
        'avg_exam_duration_sec': avg_exam_duration,
        'avg_exam_duration_min': avg_exam_duration / 60,
        'avg_per_patient_min': avg_per_patient
    }

    return output_viz_dir, summary_metrics


def step_5_visualize(schedule=None):
    """Step 5: Generate visualizations for evaluation."""
    print("\n" + "=" * 70)
    print("STEP 5: GENERATING VISUALIZATIONS")
    print("=" * 70)

    import pandas as pd
    from glob import glob

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("Warning: plotly not installed. Skipping visualizations.")
        print("Install with: pip install plotly")
        return None

    if schedule is None:
        print("Loading most recent simulated schedule...")
        schedule_files = glob(os.path.join(OUTPUT_DIR, 'simulated_day_*.csv'))
        if not schedule_files:
            print("  No schedule files found. Run step 4 first.")
            return None
        latest_file = max(schedule_files, key=os.path.getmtime)
        print(f"  Loading: {latest_file}")
        schedule_df = pd.read_csv(latest_file)
    else:
        schedule_df = pd.DataFrame(schedule)

    print(f"  Loaded {len(schedule_df)} events")

    viz_dir = os.path.join(OUTPUT_DIR, 'visualizations')
    print("Creating visualizations for overall simulation...")
    return _generate_visualizations_for_dataframe(schedule_df, viz_dir, title_prefix="Overall Simulation: ")


def _visualize_customer_simulation(customer_id, schedule_df, output_dir):
    """Generates visualizations for a single customer's simulated data."""
    print(f"\n  Generating visualizations for customer {customer_id}...")
    customer_viz_dir = os.path.join(output_dir, customer_id, 'visualizations')
    os.makedirs(customer_viz_dir, exist_ok=True)
    _, summary_metrics = _generate_visualizations_for_dataframe(
        schedule_df, customer_viz_dir, title_prefix=f"Customer {customer_id}: "
    )
    return customer_viz_dir, summary_metrics


def step_6_customer_simulation(customer_id=None):
    """Step 6: Run per-customer day simulation."""
    print("\n" + "=" * 70)
    print("STEP 6: PER-CUSTOMER DAY SIMULATION")
    print("=" * 70)

    import torch
    import pandas as pd
    from generation.customer_simulator import CustomerSimulator
    from config import CUSTOMER_OUTPUT_DIR

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Loading models...")
    exchange_model, examination_model = _load_models(device)

    print("Initializing customer simulator...")
    simulator = CustomerSimulator(exchange_model, examination_model, device)

    customers = simulator.list_customers()
    print(f"Available customers: {len(customers)}")

    if not customers:
        print("No customer schedules found. Run step 1 (preprocessing) first.")
        return None

    all_customer_summaries = []

    if customer_id and customer_id != 'all':
        # Simulate specific customer
        if customer_id not in customers:
            print(f"Customer '{customer_id}' not found. Available: {customers}")
            return None
        result = simulator.simulate_customer_day(customer_id, verbose=True)
        if result and result['schedule']:
            customer_dir = os.path.join(CUSTOMER_OUTPUT_DIR, customer_id)
            os.makedirs(customer_dir, exist_ok=True)
            df = pd.DataFrame(result['schedule'])
            csv_path = os.path.join(customer_dir, f"simulated_{result['date']}.csv")
            df.to_csv(csv_path, index=False)
            print(f"\nSaved to: {csv_path}")
            _, summary_metrics = _visualize_customer_simulation(customer_id, df, CUSTOMER_OUTPUT_DIR)
            all_customer_summaries.append({'customer_id': customer_id, **summary_metrics})
        return {customer_id: result, 'all_customer_summaries': all_customer_summaries}
    else:
        # Simulate all customers
        results = simulator.simulate_all_customers(
            verbose=True,
            output_dir=CUSTOMER_OUTPUT_DIR
        )
        if results:
            print("\nGenerating visualizations for all customers...")
            for cust_id, cust_data in results.items():
                if cust_data and cust_data['schedule']:
                    df = pd.DataFrame(cust_data['schedule'])
                    _, summary_metrics = _visualize_customer_simulation(cust_id, df, CUSTOMER_OUTPUT_DIR)
                    all_customer_summaries.append({'customer_id': cust_id, **summary_metrics})
        return {'results': results, 'all_customer_summaries': all_customer_summaries}


def step_7_general_visualizations(customer_summaries):
    """Step 7: Generate general visualizations across all customers."""
    print("\n" + "=" * 70)
    print("STEP 7: GENERATING GENERAL VISUALIZATIONS")
    print("=" * 70)

    import pandas as pd
    import plotly.express as px
    from config import OUTPUT_DIR

    if not customer_summaries:
        print("No customer summaries provided for general visualizations.")
        return None

    df_summaries = pd.DataFrame(customer_summaries)

    general_viz_dir = os.path.join(OUTPUT_DIR, 'general_visualizations')
    os.makedirs(general_viz_dir, exist_ok=True)

    print(f"Generating general visualizations in: {general_viz_dir}")

    # Total Duration per Customer
    fig1 = px.bar(
        df_summaries.sort_values('total_duration_hr', ascending=False),
        x='customer_id', y='total_duration_hr',
        title='Total Simulation Duration per Customer',
        labels={'total_duration_hr': 'Total Duration (hours)', 'customer_id': 'Customer ID'},
        hover_data=['num_patients', 'num_exchanges', 'num_exams', 'avg_per_patient_min']
    )
    fig1.update_xaxes(tickangle=45)
    fig1.write_html(os.path.join(general_viz_dir, 'total_duration_per_customer.html'))

    # Average Duration per Patient
    fig2 = px.bar(
        df_summaries.sort_values('avg_per_patient_min', ascending=False),
        x='customer_id', y='avg_per_patient_min',
        title='Average Duration per Patient per Customer',
        labels={'avg_per_patient_min': 'Avg Duration per Patient (minutes)', 'customer_id': 'Customer ID'},
    )
    fig2.update_xaxes(tickangle=45)
    fig2.write_html(os.path.join(general_viz_dir, 'avg_duration_per_patient_per_customer.html'))

    # Scatter: Duration vs Patients
    fig3 = px.scatter(
        df_summaries, x='num_patients', y='total_duration_hr',
        color='avg_per_patient_min', size='num_exchanges',
        hover_name='customer_id',
        title='Total Duration vs Number of Patients',
        labels={'num_patients': 'Number of Patients', 'total_duration_hr': 'Total Duration (hours)'},
    )
    fig3.write_html(os.path.join(general_viz_dir, 'duration_vs_patients_scatter.html'))

    # Distribution histograms
    metrics_to_plot = [
        ('total_duration_hr', 'Total Duration (Hours)'),
        ('num_patients', 'Number of Patients'),
        ('num_exchanges', 'Number of Exchange Events'),
        ('num_exams', 'Number of Examination Events'),
        ('avg_per_patient_min', 'Average Duration per Patient (Minutes)')
    ]

    for metric_col, metric_title in metrics_to_plot:
        if metric_col in df_summaries.columns and not df_summaries[metric_col].isnull().all():
            fig = px.histogram(
                df_summaries, x=metric_col,
                title=f'Distribution of {metric_title} Across Customers',
                nbins=20, hover_name='customer_id',
                labels={metric_col: metric_title}
            )
            fig.write_html(os.path.join(general_viz_dir, f'distribution_{metric_col}.html'))

    print(f"Saved general visualizations to: {general_viz_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Run the AlternatingPipeline end-to-end",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Steps:
  1. Preprocess raw data (with phase_type + shutdown extraction)
  2. Train Exchange model (unified Transformer)
  3. Train Examination model (unified Transformer)
  4. Run day simulation (on-the-fly, no buckets)
  5. Generate visualizations (overall)
  6. Per-customer day simulation
  7. Generate general visualizations

Examples:
  python run_all.py                    # Run steps 1-5
  python run_all.py --skip-preprocess  # Skip step 1
  python run_all.py --skip-training    # Skip steps 2 and 3
  python run_all.py --steps 4,5        # Only run steps 4 and 5
  python run_all.py --steps 6          # Only run customer simulation
  python run_all.py --customer all     # Simulate all customers (steps 6+7)
  python run_all.py --customer 141049  # Simulate specific customer
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
    if args.customer == 'all':
        if args.steps:
            steps_to_run = set(int(s.strip()) for s in args.steps.split(','))
            steps_to_run.add(6)
            steps_to_run.add(7)
        else:
            steps_to_run = {6, 7}
    elif args.customer:
        if args.steps:
            steps_to_run = set(int(s.strip()) for s in args.steps.split(','))
            steps_to_run.add(6)
        else:
            steps_to_run = {6}
    elif args.steps:
        steps_to_run = set(int(s.strip()) for s in args.steps.split(','))
    else:
        # Default run: 1-5
        steps_to_run = {1, 2, 3, 4, 5}
        if args.skip_preprocess:
            steps_to_run.discard(1)
        if args.skip_training:
            steps_to_run.discard(2)
            steps_to_run.discard(3)
        if args.no_viz:
            steps_to_run.discard(5)

    print("=" * 70)
    print("ALTERNATING PIPELINE - FULL EXECUTION")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Steps to run: {sorted(steps_to_run)}")
    if EXCLUDED_BODY_REGIONS:
        print(f"Excluded body regions: {EXCLUDED_BODY_REGIONS}")

    start_time = time.time()

    schedule = None
    customer_sim_results = None

    try:
        if 1 in steps_to_run:
            step_1_preprocess()

        if 2 in steps_to_run:
            step_2_train_exchange()

        if 3 in steps_to_run:
            step_3_train_examination()

        if 4 in steps_to_run:
            schedule = step_4_simulate_day()

        if 5 in steps_to_run:
            if not args.no_viz:
                step_5_visualize(schedule)

        if 6 in steps_to_run:
            customer_sim_results = step_6_customer_simulation(customer_id=args.customer)

        if 7 in steps_to_run and customer_sim_results and 'all_customer_summaries' in customer_sim_results:
            step_7_general_visualizations(customer_sim_results['all_customer_summaries'])
        elif 7 in steps_to_run and args.customer == 'all' and (not customer_sim_results or 'all_customer_summaries' not in customer_sim_results):
            print("\nWarning: Step 7 requested, but no customer summaries available. "
                  "Ensure step 6 runs for 'all' customers before step 7.")

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
