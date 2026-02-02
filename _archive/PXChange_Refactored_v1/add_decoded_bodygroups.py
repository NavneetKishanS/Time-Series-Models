"""
Add decoded BodyGroup text columns to existing generated_sequences.csv
This is a quick fix to avoid regenerating the entire dataset
"""
import pandas as pd
import os

def decode_bodygroup(encoded_value):
    """Decode body group integer back to text."""
    bodygroup_map = {
        0: 'HEAD',
        1: 'NECK',
        2: 'CHEST',
        3: 'ABDOMEN',
        4: 'PELVIS',
        5: 'SPINE',
        6: 'ARM',
        7: 'LEG',
        8: 'HAND',
        9: 'FOOT',
        10: 'KNEE'
    }
    return bodygroup_map.get(encoded_value, 'UNKNOWN')


def add_decoded_columns(input_file, output_file=None):
    """Add decoded BodyGroup columns to the CSV."""

    if not os.path.exists(input_file):
        print(f"[ERROR] File not found: {input_file}")
        return False

    print(f"Reading {input_file}...")
    df = pd.read_csv(input_file)

    print(f"Original columns: {list(df.columns)}")
    print(f"Total rows: {len(df)}")

    # Add decoded columns
    print("\nAdding decoded BodyGroup columns...")
    df['BodyGroup_from_text'] = df['BodyGroup_from'].apply(decode_bodygroup)
    df['BodyGroup_to_text'] = df['BodyGroup_to'].apply(decode_bodygroup)

    # Reorder columns for better readability
    # Put text columns right after encoded columns
    cols = list(df.columns)

    # Find position of BodyGroup_to and insert text columns after it
    bodygroup_to_idx = cols.index('BodyGroup_to')

    # Remove the text columns from their current position
    cols.remove('BodyGroup_from_text')
    cols.remove('BodyGroup_to_text')

    # Insert them after BodyGroup_to
    cols.insert(bodygroup_to_idx + 1, 'BodyGroup_from_text')
    cols.insert(bodygroup_to_idx + 2, 'BodyGroup_to_text')

    df = df[cols]

    print(f"New columns: {list(df.columns)}")

    # Show sample of decoded values
    print("\nSample decoded values:")
    sample = df[['BodyGroup_from', 'BodyGroup_from_text', 'BodyGroup_to', 'BodyGroup_to_text']].head(5)
    print(sample.to_string(index=False))

    # Show unique body groups
    print("\nUnique BodyGroup_from values:")
    unique_from = df[['BodyGroup_from', 'BodyGroup_from_text']].drop_duplicates().sort_values('BodyGroup_from')
    print(unique_from.to_string(index=False))

    print("\nUnique BodyGroup_to values:")
    unique_to = df[['BodyGroup_to', 'BodyGroup_to_text']].drop_duplicates().sort_values('BodyGroup_to')
    print(unique_to.to_string(index=False))

    # Save
    if output_file is None:
        output_file = input_file

    print(f"\nSaving to {output_file}...")
    df.to_csv(output_file, index=False)
    print("[OK] File updated successfully!")

    return True


if __name__ == "__main__":
    input_file = os.path.join('outputs', 'generated_sequences.csv')
    add_decoded_columns(input_file)
