"""
Simple script to rerun the entire pipeline
This is the easiest way to retrain models and regenerate data matching input volume
"""
import subprocess
import sys

print("\n" + "="*80)
print("RERUNNING COMPLETE PXCHANGE PIPELINE")
print("="*80)
print("\nThis will:")
print("  1. Train sequence generator model")
print("  2. Train counts generator model")
print("  3. Generate sequences matching input data volume for each customer")
print("\nNote: Preprocessing is skipped (already done)")
print("="*80 + "\n")

response = input("Continue? [Y/n]: ")
if response.lower() not in ['', 'y', 'yes']:
    print("Aborted.")
    sys.exit(0)

# Run the full pipeline (train + generate)
result = subprocess.run([
    sys.executable,
    "run_full_pipeline.py",
    "--train",
    "--generate",
    "--match-input-volume"
], shell=False)

sys.exit(result.returncode)
