import os
import pandas as pd
import sys
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add the project root to the python path so imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from AlternatingPipeline.data.preprocessing import load_raw_csv, extract_exchange_events, extract_examination_events, detect_patient_changes
from AlternatingPipeline.generation.day_simulator import DaySimulator
from AlternatingPipeline.config import ID_TO_BODY_REGION

def get_patient_sequence_from_df(df):
    patient_sequence = []
    change_indices = detect_patient_changes(df)
    boundaries = [0] + change_indices + [len(df)]
    for i in range(len(boundaries) - 1):
        start_idx = boundaries[i]
        end_idx = boundaries[i+1]
        segment = df.iloc[start_idx:end_idx]
        if not segment.empty:
            patient_id = segment['PatientId'].iloc[0]
            body_region = segment['BodyGroup_to'].iloc[0]
            patient_sequence.append({'patient_id': patient_id, 'body_region': body_region})
    return patient_sequence

def main():
    # --- Load and process data ---
    real_schedule_path = os.path.join(os.path.dirname(__file__), '..', 'PXChange_Refactored', 'data', '175832.csv')
    real_df = load_raw_csv(real_schedule_path)
    patient_sequence = get_patient_sequence_from_df(real_df)
    
    simulator = DaySimulator(buckets_dir='buckets')
    predicted_schedule_events = simulator.simulate_day(patient_sequence)

    real_exchange_events = extract_exchange_events(real_df)
    for event in real_exchange_events:
        event['event_type'] = 'exchange'
        if 'total_duration' in event:
            event['duration'] = event.pop('total_duration')
            
    real_examination_events = extract_examination_events(real_df)
    for event in real_examination_events:
        event['event_type'] = 'examination'
        if 'total_duration' in event:
            event['duration'] = event.pop('total_duration')

    real_schedule_events = real_exchange_events + real_examination_events

    # --- Create DataFrames for plotting ---
    real_events_df = pd.DataFrame(real_schedule_events)
    predicted_events_df = pd.DataFrame(predicted_schedule_events)

    # --- Visualization 1: Event Count Comparison ---
    real_counts = real_events_df[real_events_df['event_type'] == 'examination']['body_region'].value_counts().reset_index()
    real_counts.columns = ['body_region_id', 'count']
    real_counts['body_region_name'] = real_counts['body_region_id'].map(ID_TO_BODY_REGION)

    predicted_counts = predicted_events_df[predicted_events_df['event_type'] == 'examination']['body_region'].value_counts().reset_index()
    predicted_counts.columns = ['body_region_name', 'count']
    
    # Align the data for plotting
    plot_df = pd.merge(real_counts, predicted_counts, on='body_region_name', suffixes=('_real', '_predicted'), how='outer').fillna(0)

    fig1 = go.Figure(data=[
        go.Bar(name='Real', x=plot_df['body_region_name'], y=plot_df['count_real']),
        go.Bar(name='Predicted', x=plot_df['body_region_name'], y=plot_df['count_predicted'])
    ])
    fig1.update_layout(
        barmode='group',
        title='Comparison of Examination Event Counts per Body Region',
        xaxis_title='Body Region',
        yaxis_title='Event Count'
    )
    fig1.write_html("event_counts_comparison.html")
    print("Saved event counts comparison to event_counts_comparison.html")

    # --- Visualization 2: Duration Distribution ---
    fig2 = make_subplots(rows=1, cols=2, subplot_titles=("Real Event Durations", "Predicted Event Durations"))
    fig2.add_trace(go.Histogram(x=real_events_df['duration'], name='Real'), row=1, col=1)
    fig2.add_trace(go.Histogram(x=predicted_events_df['duration'], name='Predicted'), row=1, col=2)
    
    fig2.update_layout(title_text="Distribution of Event Durations")
    fig2.update_xaxes(title_text="Duration (seconds)", row=1, col=1)
    fig2.update_xaxes(title_text="Duration (seconds)", row=1, col=2)
    fig2.update_yaxes(title_text="Count", type="log", row=1, col=1)
    fig2.update_yaxes(title_text="Count", type="log", row=1, col=2)
    
    fig2.write_html("duration_distribution_comparison.html")
    print("Saved duration distribution comparison to duration_distribution_comparison.html")


if __name__ == "__main__":
    main()
