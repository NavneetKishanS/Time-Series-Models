# PXChange_Refactored Modifications Summary

## Overview
This document summarizes the modifications made to the PXChange_Refactored project to fix the MRI_MSR_104 repetition issue and add new features to the generated sequences output.

## Changes Made

### 1. Fixed MRI_MSR_104 Repetition Issue

**Problem**: The generated sequences had excessive repetitions of MRI_MSR_104 (token_id=12), with up to 39 consecutive occurrences in a single sequence, even though it typically appears only once in the training data.

**Solution**: Implemented a post-processing filter `remove_excessive_repetitions()` in `generation/generate_pipeline.py` that limits consecutive repetitions of any token to a maximum of 2 occurrences.

**Results**:
- Before: Average 39+ consecutive MRI_MSR_104 tokens
- After: Average 1.23 occurrences per sequence
  - 143/185 samples (77%) have exactly 1 occurrence
  - 41/185 samples (22%) have 2 occurrences
  - Only 1/185 samples (0.5%) has 3 occurrences

### 2. Added BodyGroup Fields to Output

**Feature**: Added `BodyGroup_from` and `BodyGroup_to` columns to the generated sequences CSV.

**Implementation**: These fields are extracted from the conditioning data for each customer and included in the output for every token in the generated sequence.

**Example values**:
- `BodyGroup_from`: 0 (HEAD)
- `BodyGroup_to`: 5 (SPINE)

### 3. Added Unique PatientID Fields

**Feature**: Generate unique 40-character hexadecimal PatientID strings for each generated sample, matching the format of the original training data.

**Implementation**:
- Added `generate_patient_id()` function that creates cryptographically secure random 40-character hex strings
- Each sample gets unique `PatientID_from` and `PatientID_to` values
- All 185 test samples have unique PatientID pairs

**Example**:
- `PatientID_from`: f9bcad005a084f2d462dbd089fc0c0ea8cd2a85a
- `PatientID_to`: 4bc72e0fb9bc2467bcbd0ddad17d0f210ab68e6f

## Modified Files

1. **generation/generate_pipeline.py**
   - Added `generate_patient_id()` function
   - Added `remove_excessive_repetitions()` function
   - Modified `generate_sequences_and_counts()` to:
     - Apply repetition filtering to generated tokens
     - Extract and include BodyGroup fields
     - Generate unique PatientIDs for each sample
     - Track filtered steps correctly
   - Fixed encoding issues (replaced checkmark characters with [OK])

## Output Format

The generated CSV now includes the following columns:
- SN (customer serial number)
- customer_idx
- sample_idx
- step
- token_id
- token_name
- **BodyGroup_from** (NEW - encoded integer)
- **BodyGroup_to** (NEW - encoded integer)
- **BodyGroup_from_text** (NEW - decoded text: HEAD, SPINE, KNEE, etc.)
- **BodyGroup_to_text** (NEW - decoded text: HEAD, SPINE, KNEE, etc.)
- **PatientID_from** (NEW - 40-char hex string)
- **PatientID_to** (NEW - 40-char hex string)
- predicted_mu
- predicted_sigma
- sampled_duration
- total_time

### Body Group Encoding
- 0 = HEAD
- 1 = NECK
- 2 = CHEST
- 3 = ABDOMEN
- 4 = PELVIS
- 5 = SPINE
- 6 = ARM
- 7 = LEG
- 8 = HAND
- 9 = FOOT
- 10 = KNEE

## Verification

Run `verify_output.py` to verify all modifications:
```bash
python verify_output.py
```

This will check:
- MRI_MSR_104 repetition counts
- Presence of BodyGroup fields
- Uniqueness of PatientID fields

## Usage

Generate sequences with the updated code:
```bash
python main_pipeline.py generate --num-samples-per-customer 15
```

This will create `outputs/generated_sequences.csv` with all the new features and fixes.
