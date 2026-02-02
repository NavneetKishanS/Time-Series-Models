# Architecture Comparison: Old vs. New System

## Overview

This document compares the original time-series prediction system with the new conditional generation system.

---

## High-Level Comparison

| Aspect | Old System (PXChange) | New System (PXChange_Refactored) |
|--------|----------------------|----------------------------------|
| **Paradigm** | Prediction | Generation |
| **Goal** | Predict time for given sequences | Generate plausible sequences with times |
| **Stages** | 2 (Proportions → Total Time) | 2 (Sequences → Counts) |
| **Model Type** | Regression | Conditional Generation |
| **Uncertainty** | No explicit uncertainty | Explicit σ per prediction |
| **Output** | Single point estimates | Distribution parameters |

---

## Detailed Architectural Differences

### Stage 1: Symbolic/Temporal Modeling

#### Old System: Transformer for Proportions

**File**: `TF_TimeSeries_Refactored.py`

**Purpose**: Predict how time is *distributed* across steps in a **given** sequence

**Architecture**:
```
Input: Full sequence (sourceID, features, timediff)
  ↓
Transformer Encoder (6 layers)
  ↓
MaskedSoftmax
  ↓
Output: Proportion per step (sums to 1.0)
```

**Key Properties**:
- **Parallel prediction**: All proportions predicted simultaneously
- **Known sequence**: Input sequence is ground truth
- **Constrained output**: Proportions must sum to 1.0
- **Loss**: KL Divergence

**Use Case**: Given a sequence, predict timing distribution

---

#### New System: Conditional Sequence Generator

**File**: `conditional_sequence_generator.py`

**Purpose**: *Generate* plausible symbolic sequences from scratch

**Architecture**:
```
Input: Conditioning (patient/scan context)
  ↓
Encoder (6 layers): Encode conditioning
  ↓
Decoder (6 layers): Auto-regressive generation
  ↓
Output: Token probabilities at each step
```

**Key Properties**:
- **Auto-regressive**: Each token depends on previous tokens
- **Generative**: Creates new sequences, not given
- **Categorical output**: Discrete token probabilities
- **Loss**: Cross-entropy with label smoothing

**Use Case**: Generate novel scan sequences for planning/simulation

---

### Stage 2: Quantitative Modeling

#### Old System: LSTM for Total Time

**File**: `LSTM_TF_Optimized.py`

**Purpose**: Predict *total sequence time* given proportions and features

**Architecture**:
```
Three Branches:
  1. Bidirectional LSTM + Attention (sequence features)
  2. Bidirectional GRU + Attention (predicted proportions)
  3. Dense layers (patient metadata)
  ↓
Concatenate
  ↓
Dense layers (256 → 128 → 64)
  ↓
Output: Single value (total time)
```

**Key Properties**:
- **Multi-branch fusion**: Combines proportions, features, metadata
- **Recurrent processing**: LSTM/GRU for temporal dependencies
- **Point estimate**: Single total time prediction
- **Loss**: Huber loss (robust to outliers)

**Use Case**: Given sequence and proportions, predict total duration

---

#### New System: Conditional Counts Generator

**File**: `conditional_counts_generator.py`

**Purpose**: Predict *step-level durations* with uncertainty

**Architecture**:
```
Input: Conditioning + Generated Sequence
  ↓
Conditioning Encoder (6 layers)
  ↓
Sequence Encoder (6 layers)
  ↓
Cross-Attention (4 layers): Sequence ← Conditioning
  ↓
Dual Output Heads: μ and σ
  ↓
Output: Mean and std dev per step
```

**Key Properties**:
- **Cross-attention**: Explicitly models conditioning influence
- **Parallel prediction**: All steps predicted simultaneously
- **Uncertainty quantification**: Outputs μ and σ
- **Distributional output**: Gamma distribution parameters
- **Loss**: Negative log-likelihood (Gamma NLL)

**Use Case**: Given generated sequence, predict durations with uncertainty

---

## Conceptual Differences

### Old System: Predictive Pipeline

```
Known Sequence → Predict Timing
```

**Workflow**:
1. You have a sequence of scans
2. Predict proportions: "How is time distributed?"
3. Predict total time: "How long will this take?"

**Purpose**: Time estimation for scheduling

**Characteristics**:
- **Discriminative**: P(time | sequence)
- **Supervised**: Learn from observed sequences
- **Point estimates**: Single predicted value
- **Deterministic**: Same input → same output

---

### New System: Generative Pipeline

```
Context → Generate Sequence → Predict Counts
```

**Workflow**:
1. You have patient/scan context
2. Generate plausible sequences: "What scans might we do?"
3. Predict durations with uncertainty: "How long might each step take?"
4. Sample realistic timings: "What are plausible outcomes?"

**Purpose**: Simulation, scenario planning, synthetic data

**Characteristics**:
- **Generative**: P(sequence | context) × P(counts | sequence, context)
- **Conditional**: Both stages condition on context
- **Distributional**: Outputs probability distributions
- **Stochastic**: Same input → different outputs (sampled)

---

## Mathematical Formulation

### Old System

**Stage 1: Proportions**
```
proportions = Transformer(sequence_features)
where: sum(proportions) = 1.0
```

**Stage 2: Total Time**
```
total_time = LSTM([sequence_features, proportions, metadata])
```

**Final Output**:
```
step_durations = proportions × total_time
```

---

### New System

**Stage 1: Sequence Generation**
```
P(x | c) = ∏ₜ P(xₜ | x₁...xₜ₋₁, c)

where:
  x = symbolic sequence
  c = conditioning (context)
  xₜ ~ Categorical(logits_t)
```

**Stage 2: Count Generation**
```
P(y | c, x) = ∏ₜ P(yₜ | c, x)

where:
  y = duration sequence
  yₜ ~ Gamma(α, β)
  α = (μₜ/σₜ)²
  β = μₜ/σₜ²
  μₜ, σₜ = CountsModel(c, x)
```

**Sampling**:
```python
# Generate sequence
x_generated = sample_tokens(P(x | c))

# Predict parameters
μ, σ = CountsModel(c, x_generated)

# Sample durations
y_sampled ~ Gamma(α, β)
```

---

## Key Innovations in New System

### 1. Explicit Uncertainty Quantification

**Old**: Single point estimate
```python
total_time = 450.2  # seconds
```

**New**: Distribution with uncertainty
```python
mu = 450.2, sigma = 45.3
# Can sample: 430.1, 465.8, 442.3, ...
```

### 2. Auto-regressive Sequence Generation

**Old**: Sequence given as input
```python
sequence = [START, scan1, scan2, ..., END]  # Known
```

**New**: Sequence generated step-by-step
```python
sequence = []
sequence.append(START)
while not done:
    next_token = model.predict_next(sequence, context)
    sequence.append(next_token)
```

### 3. Parallel Count Prediction

**Old**: Sequential processing with LSTM/GRU

**New**: Parallel prediction via Transformer
- All positions predicted simultaneously
- No autoregressive dependency for counts
- Faster inference

### 4. Gamma Distribution for Counts

**Old**: Direct regression (can produce negative values)

**New**: Gamma distribution (always positive)
- Shape enforced by parameterization
- Realistic for duration data
- Supports sampling

---

## When to Use Each System

### Use Old System (PXChange) When:

✅ You have a **known sequence** and need to predict its duration
✅ You want **deterministic predictions** for scheduling
✅ You need **single point estimates** for planning
✅ Your goal is **time estimation** for existing workflows
✅ You have **limited data** and want simpler models

**Example**: "We're doing scans A, B, C on this patient. How long will it take?"

---

### Use New System (PXChange_Refactored) When:

✅ You want to **generate plausible sequences** from context
✅ You need **uncertainty estimates** for risk analysis
✅ You want to **simulate multiple scenarios**
✅ Your goal is **synthetic data generation** or **what-if analysis**
✅ You need **distributional predictions** for probabilistic planning

**Example**: "For patients with these characteristics, what scan sequences might occur and how long might they take?"

---

## Performance Considerations

| Aspect | Old System | New System |
|--------|-----------|------------|
| **Training Time** | Moderate (2 models) | Longer (2 larger models) |
| **Inference Time** | Fast (single pass) | Moderate (sequential + parallel) |
| **Memory Usage** | Lower (LSTM/GRU) | Higher (full Transformers) |
| **Data Requirements** | Moderate | Higher (needs diverse examples) |
| **Interpretability** | High (direct predictions) | Moderate (sampled outputs) |

---

## Migration Path

If you want to transition from old to new system:

1. **Keep old system** for production scheduling
2. **Use new system** for:
   - Long-term capacity planning
   - Scenario analysis
   - Synthetic data for training
   - Risk assessment

3. **Hybrid approach**:
   ```python
   # Generate possible sequences (new system)
   sequences = new_system.generate(context)

   # Estimate timing for each (old system)
   for seq in sequences:
       time_estimate = old_system.predict(seq)
   ```

---

## Summary Table

| Feature | Old System | New System |
|---------|-----------|------------|
| **Paradigm** | Discriminative (prediction) | Generative (sampling) |
| **Input** | Known sequence | Context only |
| **Output** | Point estimates | Distributions |
| **Uncertainty** | ❌ No | ✅ Yes (σ per step) |
| **Sequence** | Given | Generated |
| **Sampling** | ❌ No | ✅ Multiple outcomes |
| **Architecture** | LSTM + Transformer | Full Transformers |
| **Use Case** | Time estimation | Scenario simulation |
| **Strengths** | Fast, accurate predictions | Flexible, uncertainty-aware |
| **Weaknesses** | No generation, no uncertainty | Slower, more complex |

---

## Conclusion

Both systems are valuable for different purposes:

- **Old System** excels at **time prediction** for **known sequences**
- **New System** excels at **scenario generation** with **uncertainty quantification**

Choose based on your use case:
- **Operational scheduling** → Old System
- **Strategic planning/simulation** → New System
- **Comprehensive analysis** → Use both!
