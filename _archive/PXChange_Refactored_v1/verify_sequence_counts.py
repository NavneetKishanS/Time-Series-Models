"""
Verification script to show how many sequences will be generated per customer
"""
import pandas as pd
import os

print("\n" + "="*80)
print("SEQUENCE COUNT VERIFICATION")
print("="*80 + "\n")

# Load preprocessed data
preprocessed_file = os.path.join("data", "preprocessed", "all_preprocessed.csv")

if not os.path.exists(preprocessed_file):
    print(f"Error: {preprocessed_file} not found. Run preprocessing first!")
    exit(1)

df = pd.read_csv(preprocessed_file)

print(f"Loaded preprocessed data: {len(df):,} total rows\n")

# Calculate sequences per customer
sequences_per_customer = df.groupby('dataset_id')['SeqOrder'].nunique().sort_values(ascending=False)

print("Sequences per customer (will match this in generation):")
print("="*80)
print(f"{'Customer':<15} {'Sequences':>10} {'Total Rows':>12} {'Avg Seq Length':>15}")
print("-"*80)

total_sequences = 0
total_rows_without_special = 0

for customer_id, num_sequences in sequences_per_customer.items():
    customer_df = df[df['dataset_id'] == customer_id]
    # Count rows excluding START and END tokens
    rows_without_special = len(customer_df[~customer_df['sourceID'].isin([11, 14])])
    avg_length = rows_without_special / num_sequences

    print(f"{customer_id:<15} {num_sequences:>10} {rows_without_special:>12} {avg_length:>15.1f}")

    total_sequences += num_sequences
    total_rows_without_special += rows_without_special

print("-"*80)
print(f"{'TOTAL':<15} {total_sequences:>10} {total_rows_without_special:>12} {total_rows_without_special/total_sequences:>15.1f}")
print("="*80)

print(f"\nSummary:")
print(f"  Total customers: {len(sequences_per_customer)}")
print(f"  Total sequences to generate: {total_sequences:,}")
print(f"  Expected output rows (approx): {total_rows_without_special:,}")
print(f"  Average sequences per customer: {total_sequences/len(sequences_per_customer):.1f}")
print(f"  Average row count per customer: {total_rows_without_special/len(sequences_per_customer):.1f}")

print("\n" + "="*80)
print("When you run generation with --match-input-volume:")
print("  - Customer 141049 will generate", sequences_per_customer.get(141049, sequences_per_customer.get('141049', 'N/A')), "sequences")
print("  - Each customer will generate the number of sequences shown above")
print("  - Total output will be approximately", f"{total_rows_without_special:,}", "rows")
print("="*80 + "\n")
