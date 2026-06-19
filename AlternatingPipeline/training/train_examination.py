"""
Training script for the Examination Model (Unified Transformer).

Trains the model to generate MRI event sequences for specific body regions.
"""
import math
import os
import sys
import json
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import pickle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    EXAMINATION_MODEL_CONFIG, EXAMINATION_TRAINING_CONFIG,
    MODEL_SAVE_DIR, RANDOM_SEED, USE_GPU, MAX_SEQ_LEN,
    START_TOKEN_ID, END_TOKEN_ID, PAD_TOKEN_ID, VOCAB_SIZE,
    SOURCEID_VOCAB
)
from models.examination_model import create_examination_model
from data.preprocessing import load_preprocessed_data
from data.archive_duration_priors import load_examination_priors
from data.examination_duration_calibration import calibrate_examination_durations
from training.utils import temporal_split, build_conditioning_tensor, make_pad_collate


class ExaminationDataset(Dataset):
    """Dataset for examination (scan sequence) training."""

    def __init__(self, examination_sequences, max_seq_len=None, augment=False,
                 oversample=1, duration_scale=1.0, abort_oversample=1):
        if max_seq_len is None:
            max_seq_len = MAX_SEQ_LEN

        self.max_seq_len = max_seq_len
        self.augment = augment
        self.duration_scale = duration_scale
        self.data = []

        abort_id = SOURCEID_VOCAB['MRI_MSR_34']  # "Stopped by User"

        for seq in examination_sequences:
            conditioning = build_conditioning_tensor(seq['conditioning'])

            body_region = seq['body_region']
            # Scan-type / scanner conditioning — default to 0 ('other' /
            # first scanner) for pkls built before these fields existed.
            sequence_type = int(seq.get('sequence_type', 0))
            serial_idx = int(seq.get('serial_idx', 0))
            tokens = seq['sequence']
            durations = seq.get('durations', [0.0] * len(tokens))

            # Input: [START, tok1, tok2, ..., tokN]
            # Target: [tok1, tok2, ..., tokN, END]
            input_seq = [START_TOKEN_ID] + tokens[:max_seq_len - 1]
            target_seq = tokens[:max_seq_len - 1] + [END_TOKEN_ID]

            # Duration target = SPAN TOTAL on the finish token, zeros elsewhere.
            # Real per-token gaps are ~10 s for EVERY scan type — the scan-type
            # duration spread (scout ~19 s ... space ~235 s) lives almost
            # entirely in the NUMBER of events per span (corr(total, n_tokens)
            # ≈ 0.83), not in per-token magnitude. Per-token targets therefore
            # gave the duration head nothing to learn from its scan-type
            # conditioning (generated mu pinned flat at ~0.215 across all
            # types), and generated spans of ~3 tokens collapsed every scan to
            # ~50 s. Concentrating the total on the finish token turns the
            # length signal into a magnitude signal the conditioned head CAN
            # learn; generation sums per-token durations across the span, so
            # the emitted scan duration becomes the predicted total.
            n_kept = len(tokens[:max_seq_len - 1])
            span_total = sum(max(0.0, d) for d in durations)
            target_durations = [0.0] * n_kept + [0.0]
            if n_kept > 0:
                target_durations[n_kept - 1] = span_total

            # Pad
            pad_len = max_seq_len - len(input_seq)
            input_seq = input_seq + [PAD_TOKEN_ID] * pad_len
            target_seq = target_seq + [PAD_TOKEN_ID] * pad_len
            target_durations = target_durations + [0.0] * pad_len

            self.data.append({
                'conditioning': conditioning,
                'body_region': body_region,
                'sequence_type': sequence_type,
                'serial_idx': serial_idx,
                'input_seq': torch.tensor(input_seq, dtype=torch.long),
                'target_seq': torch.tensor(target_seq, dtype=torch.long),
                'target_durations': target_durations,  # kept as list for augmentation
                'is_abort': abort_id in tokens,
            })

        if oversample > 1:
            self.data = self.data * oversample

        # Targeted oversampling of the rare "Stopped by User" (MRI_MSR_34) abort
        # sequences. This REPLACES the inverse-frequency class weighting that was
        # removed from the loss: that weighting gave the never-counted START/END/
        # UNK tokens a phantom inverse-frequency weight, poisoned the mean-1.0
        # normaliser, and collapsed the token decoder into emitting MSR_104
        # immediately. Duplicating abort sequences boosts the rare token safely at
        # the data layer instead of destabilising the cross-entropy scale.
        self.num_abort_sequences = sum(1 for d in self.data if d['is_abort'])
        if abort_oversample > 1:
            abort_items = [d for d in self.data if d['is_abort']]
            self.data = self.data + abort_items * (abort_oversample - 1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        durations = list(item['target_durations'])
        if self.augment:
            noise = np.random.normal(0, 0.10, len(durations))
            durations = [max(0.0, d * (1 + n)) for d, n in zip(durations, noise)]
        # Normalise raw seconds so Gaussian NLL stays in a reasonable range
        if self.duration_scale != 1.0:
            durations = [d / self.duration_scale for d in durations]
        return (
            item['conditioning'],
            torch.tensor(item['body_region'], dtype=torch.long),
            torch.tensor(item['sequence_type'], dtype=torch.long),
            torch.tensor(item['serial_idx'], dtype=torch.long),
            item['input_seq'],
            item['target_seq'],
            torch.tensor(durations, dtype=torch.float32),
        )


def compute_token_class_weights(sequences, vocab_size=VOCAB_SIZE, smoothing=0.5):
    """Inverse-frequency class weights for the token cross-entropy.

    *** CURRENTLY UNUSED — DO NOT RE-ENABLE AS-IS. ***
    This implementation is buggy: START/END/UNK never appear in
    `seq['sequence']` (END is appended only in the dataset), so they get count
    0 and a huge inverse-frequency weight. Those phantom weights dominate the
    `nonzero.mean()` normaliser (~180x), crushing every real token to ~0.01–0.23
    and, via PyTorch's mean-reduced weighted cross-entropy, letting the single
    END target soak up ~96% of the token-loss gradient. The result is a token
    decoder that never learns structure and collapses to emitting MSR_104
    immediately (StepCount=1). It also failed at its own goal: MRI_MSR_34 got a
    weight of ~0.057 (below neutral 1.0). Rare-event surfacing is now handled by
    targeted abort oversampling in ExaminationDataset. If reintroducing weights,
    first force PAD/START/END/UNK weight = 1.0 and clip weights to e.g. [0.5, 3].

    Rare workflow events — most importantly MRI_MSR_34 ("Stopped by User") —
    are otherwise crowded out of the softmax by frequent tokens and never
    appear in synthetic data. Weights are normalised to mean 1.0 so the
    overall loss scale is unchanged. `smoothing` dampens the weighting so a
    very rare token does not dominate the gradient.
    """
    counts = np.zeros(vocab_size, dtype=np.float64)
    for seq in sequences:
        for tok in seq['sequence']:
            if 0 <= tok < vocab_size:
                counts[tok] += 1
    counts = counts + counts.sum() * 1e-6  # avoid div-by-zero for unseen tokens
    freq = counts / counts.sum()
    weights = (1.0 / freq) ** smoothing
    weights[PAD_TOKEN_ID] = 0.0  # padding is ignored anyway
    nonzero = weights[weights > 0]
    weights = weights / nonzero.mean()  # normalise so mean weight ≈ 1.0
    return torch.tensor(weights, dtype=torch.float32)


def train_examination_model(data_path=None, config=None, training_config=None,
                            save_dir=None, verbose=True):
    """
    Train the Examination Model.

    Args:
        data_path: Path to preprocessed data pickle file
        config: Model config dict
        training_config: Training config dict
        save_dir: Directory to save model
        verbose: Print progress

    Returns:
        Trained model, training history
    """
    if config is None:
        config = EXAMINATION_MODEL_CONFIG
    if training_config is None:
        training_config = EXAMINATION_TRAINING_CONFIG
    if save_dir is None:
        save_dir = os.path.join(MODEL_SAVE_DIR, 'examination')

    os.makedirs(save_dir, exist_ok=True)

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    device = torch.device('cuda' if USE_GPU and torch.cuda.is_available() else 'cpu')
    if verbose:
        print(f"Using device: {device}")

    # Load archive duration priors (used for calibration)
    archive_priors = load_examination_priors()
    if verbose:
        if archive_priors:
            print(f"Archive priors loaded for {len(archive_priors)} body groups: {list(archive_priors.keys())}")
        else:
            print("No archive priors found — duration calibration disabled.")

    # Load data
    if verbose:
        print("Loading data...")

    if data_path is None:
        preprocessed = load_preprocessed_data()
    else:
        with open(data_path, 'rb') as f:
            preprocessed = pickle.load(f)

    examination_sequences = preprocessed['examination']
    if verbose:
        print(f"Loaded {len(examination_sequences)} examination sequences")

    # Temporal split
    train_sequences, val_sequences = temporal_split(examination_sequences, val_days=2)

    if verbose:
        print(f"Temporal split: Train={len(train_sequences)}, Val={len(val_sequences)}")
        if train_sequences and val_sequences:
            train_dates = [s['start_datetime'] for s in train_sequences]
            val_dates = [s['start_datetime'] for s in val_sequences]
            print(f"  Train date range: {min(train_dates)} to {max(train_dates)}")
            print(f"  Val date range: {min(val_dates)} to {max(val_dates)}")

    # Calibrate durations using archive priors
    train_sequences = calibrate_examination_durations(train_sequences, archive_priors)
    val_sequences = calibrate_examination_durations(val_sequences, archive_priors)
    if verbose:
        print("Duration calibration applied to train and val sequences")

    # ---- Self-verifying scan-type duration-spread guard ----------------------
    # The examination model conditions duration on scan type (use_exam_conditioning),
    # so the TRAINING TARGETS must still vary by sequence_type AFTER calibration.
    # Historically the calibration rescaled every sequence to the body_region
    # archive prior; with body_region 100% UNKNOWN that flattened all scan types
    # to one ~49 s mean, and the model dutifully learned flat per-type durations
    # (synthetic mu pinned ~0.215 for scout==tse==space). db44198 fixed the
    # calibration, but a retrain that does not reflect it silently reintroduces
    # the bug. This block makes that failure LOUD at train time instead of
    # surfacing only after a full generate + eval cycle.
    if verbose:
        from collections import defaultdict
        from config import ID_TO_SEQUENCE_TYPE
        _by_type = defaultdict(list)
        for _s in train_sequences:
            _total = sum(_s.get('durations', []) or [0.0])
            _by_type[int(_s.get('sequence_type', 0))].append(_total)
        _means = {st: (sum(v) / len(v)) for st, v in _by_type.items() if v}
        if len(_means) >= 2:
            _lo, _hi = min(_means.values()), max(_means.values())
            _spread = _hi / max(1e-9, _lo)
            print(f"Scan-type duration spread (calibrated train targets): "
                  f"{_spread:.1f}x across {len(_means)} types "
                  f"[{_lo:.0f}s .. {_hi:.0f}s]")
            for st in sorted(_means, key=_means.get):
                _name = ID_TO_SEQUENCE_TYPE.get(st, str(st))
                print(f"    {_name:<8} n={len(_by_type[st]):>6}  mean={_means[st]:>7.1f}s")
            if _spread < 5.0:
                print("  !! WARNING: per-scan-type duration spread has COLLAPSED "
                      "(<5x). The duration head will learn flat per-type durations.\n"
                      "     This is the db44198 calibration-flattening regression — "
                      "verify calibrate_examination_durations and the source pkl's\n"
                      "     `durations`/`sequence_type` fields BEFORE spending a retrain.")
        else:
            print("  !! WARNING: training sequences carry <2 distinct sequence_type "
                  "values — scan-type conditioning cannot be learned (stale pkl?).")
    # --------------------------------------------------------------------------

    # Create datasets
    augment = training_config.get('augment_training', False)
    oversample = training_config.get('oversample_factor', 1)
    duration_scale = training_config.get('duration_scale', 1.0)
    abort_oversample = training_config.get('abort_oversample_factor', 1)
    train_dataset = ExaminationDataset(
        train_sequences, augment=augment, oversample=oversample,
        duration_scale=duration_scale, abort_oversample=abort_oversample,
    )
    val_dataset = ExaminationDataset(
        val_sequences, augment=False, oversample=1,
        duration_scale=duration_scale, abort_oversample=1,
    )

    if verbose:
        print(f"Train dataset: {len(train_dataset)}, Val dataset: {len(val_dataset)}")
        print(f"  Abort (MRI_MSR_34) sequences in train: {train_dataset.num_abort_sequences} "
              f"(oversampled x{abort_oversample})")

    # Trim each batch to its longest real sequence — the tuple layout is
    # (conditioning, body_region, sequence_type, serial_idx, input_seq,
    # target_seq, durations); positions 4/5/6 are the per-token fields,
    # measured off the PAD-terminated input_seq at position 4. This is the
    # dominant CPU speedup: examination scans are short but pad to 128, and
    # attention is O(L^2). See make_pad_collate.
    collate = make_pad_collate(seq_indices=(4, 5, 6), length_index=4,
                               pad_token_id=PAD_TOKEN_ID)
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config['batch_size'],
        shuffle=True,
        num_workers=0,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config['batch_size'],
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
    )

    # Create model
    model = create_examination_model(config)
    model = model.to(device)

    if verbose:
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # NOTE: inverse-frequency token class weighting was REMOVED here — it
    # collapsed the token decoder (see compute_token_class_weights docstring).
    # Rare aborts (MRI_MSR_34) are now surfaced via targeted oversampling in
    # ExaminationDataset (abort_oversample_factor).

    # Optimizer with warmup
    optimizer = optim.AdamW(
        model.parameters(),
        lr=training_config['learning_rate'],
        weight_decay=1e-4,
    )

    total_steps = training_config['epochs'] * len(train_loader)

    def lr_lambda(step):
        warmup_steps = training_config['warmup_steps']
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.05, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Training loop
    history = {'train_loss': [], 'val_loss': [], 'val_perplexity': [], 'train_duration_loss': []}
    best_val_loss = float('inf')
    patience_counter = 0
    global_step = 0

    for epoch in range(training_config['epochs']):
        model.train()
        train_loss = 0.0
        train_dur_loss = 0.0

        for conditioning, body_region, sequence_type, serial_idx, input_seq, target_seq, target_durations in tqdm(
            train_loader, disable=not verbose, desc=f"Epoch {epoch+1}"
        ):
            conditioning = conditioning.to(device)
            body_region = body_region.to(device)
            sequence_type = sequence_type.to(device)
            serial_idx = serial_idx.to(device)
            input_seq = input_seq.to(device)
            target_seq = target_seq.to(device)
            target_durations = target_durations.to(device)

            optimizer.zero_grad()

            logits, duration_mu, duration_sigma = model(
                conditioning,
                {'body_region': body_region,
                 'sequence_type': sequence_type, 'serial_idx': serial_idx},
                input_seq,
            )

            token_loss = model.compute_loss(
                logits, target_seq,
                label_smoothing=training_config['label_smoothing'],
            )

            # Supervise the duration head ONLY at positions carrying a span
            # total (target > 0). The span-total encoding zero-fills ~90% of
            # positions; averaging the NLL over them let the optimizer park
            # mu≈0 everywhere and hedge sigma on the finish token — the 06-11
            # run hit dur_loss≈-1.9 that way, exactly the all-zeros optimum,
            # and generated durations collapsed to ~7 s flat. Masking the
            # zeros puts 100% of the duration gradient on the signal.
            pad_mask = (target_seq == PAD_TOKEN_ID)
            dur_mask = pad_mask | (target_durations <= 0)
            duration_loss = model.compute_duration_loss(
                duration_mu, duration_sigma, target_durations, ignore_mask=dur_mask
            )

            duration_weight = training_config.get('duration_loss_weight', 0.3)
            loss = token_loss + duration_weight * duration_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), training_config['gradient_clip'])
            optimizer.step()
            scheduler.step()
            global_step += 1

            train_loss += loss.item()
            train_dur_loss += duration_loss.item()

        train_loss /= len(train_loader)
        train_dur_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for conditioning, body_region, sequence_type, serial_idx, input_seq, target_seq, target_durations in val_loader:
                conditioning = conditioning.to(device)
                body_region = body_region.to(device)
                sequence_type = sequence_type.to(device)
                serial_idx = serial_idx.to(device)
                input_seq = input_seq.to(device)
                target_seq = target_seq.to(device)
                target_durations = target_durations.to(device)

                logits, duration_mu, duration_sigma = model(
                    conditioning,
                    {'body_region': body_region,
                     'sequence_type': sequence_type, 'serial_idx': serial_idx},
                    input_seq,
                )
                token_loss = model.compute_loss(logits, target_seq)
                pad_mask = (target_seq == PAD_TOKEN_ID)
                dur_mask = pad_mask | (target_durations <= 0)
                dur_loss = model.compute_duration_loss(
                    duration_mu, duration_sigma, target_durations, ignore_mask=dur_mask
                )
                duration_weight = training_config.get('duration_loss_weight', 0.3)
                loss = token_loss + duration_weight * dur_loss
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_perplexity = np.exp(val_loss)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_perplexity'].append(val_perplexity)
        history['train_duration_loss'].append(train_dur_loss)

        if verbose:
            print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, "
                  f"val_loss={val_loss:.4f}, perplexity={val_perplexity:.2f}, "
                  f"dur_loss={train_dur_loss:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(save_dir, 'examination_model_best.pt'))
        else:
            patience_counter += 1

        if patience_counter >= training_config['early_stopping_patience']:
            if verbose:
                print(f"Early stopping at epoch {epoch+1}")
            break

    # Save final model
    torch.save(model.state_dict(), os.path.join(save_dir, 'examination_model_final.pt'))

    with open(os.path.join(save_dir, 'training_history.pkl'), 'wb') as f:
        pickle.dump(history, f)

    # ---- Post-train UNPADDED duration probe (go/no-go canary) ----------------
    # Step 05 queries estimate_durations with short unpadded sequences and
    # reads the span total at index len(tokens)-1; every duration collapse so
    # far was invisible at train time because the losses only ever see padded
    # batches. This probe asks the BEST checkpoint (the one step 05 loads)
    # the exact same question on real val sequences and prints predicted vs
    # target seconds per scan type. If the spread line below is flat, do NOT
    # spend a generate+eval cycle on this checkpoint.
    best_path = os.path.join(save_dir, 'examination_model_best.pt')
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()
    # The probe ALWAYS runs (not gated on verbose) and writes its result to a
    # small JSON next to the checkpoint on DBFS. Databricks drops cell output
    # for very long (~24h) runs, so the printed table can be lost on refresh —
    # duration_probe.json survives exactly like MODEL_MANIFEST.json and is the
    # decisive go/no-go a human (or the next session) can read after the fact.
    probe_rows = []
    if val_sequences:
        from collections import defaultdict
        from config import ID_TO_SEQUENCE_TYPE
        _probe_by_type = defaultdict(list)
        for _s in val_sequences:
            _probe_by_type[int(_s.get('sequence_type', 0))].append(_s)
        if verbose:
            print("\nPost-train duration probe (unpadded input, best checkpoint):")
        with torch.no_grad():
            for _st, _seqs in sorted(_probe_by_type.items()):
                _p, _t = [], []
                for _s in _seqs[:25]:
                    _tokens = _s['sequence'][:model.max_seq_len - 1]
                    if not _tokens:
                        continue
                    _inp = torch.tensor([[START_TOKEN_ID] + _tokens],
                                        dtype=torch.long, device=device)
                    _cond = build_conditioning_tensor(_s['conditioning']).unsqueeze(0).to(device)
                    _info = {'body_region': torch.tensor([_s['body_region']], device=device),
                             'sequence_type': torch.tensor([_st], device=device),
                             'serial_idx': torch.tensor([int(_s.get('serial_idx', 0))], device=device)}
                    _mu, _ = model.estimate_durations(_inp, _cond, _info)
                    _m = _mu[0, len(_tokens) - 1].item()
                    _pred_sec = (math.expm1(_m) if model.duration_mode == 'log' else _m) * duration_scale
                    _p.append(_pred_sec)
                    _t.append(sum(max(0.0, d) for d in _s.get('durations', [])))
                if _p:
                    _name = ID_TO_SEQUENCE_TYPE.get(_st, str(_st))
                    probe_rows.append({
                        'sequence_type': _name,
                        'n': len(_p),
                        'predicted_s': round(sum(_p) / len(_p), 2),
                        'target_s': round(sum(_t) / len(_t), 2),
                    })
                    if verbose:
                        print(f"    {_name:<8} n={len(_p):>3}  "
                              f"predicted={probe_rows[-1]['predicted_s']:>7.1f}s"
                              f"  target={probe_rows[-1]['target_s']:>7.1f}s")

    probe = {'rows': probe_rows}
    if len(probe_rows) >= 2:
        _vals = [r['predicted_s'] for r in probe_rows]
        _lo, _hi = min(_vals), max(_vals)
        _spread = _hi / max(1e-9, _lo)
        _flat = _spread < 3.0 or _hi < 30.0
        probe.update({'predicted_lo_s': _lo, 'predicted_hi_s': _hi,
                      'spread_x': round(_spread, 2), 'flat_warning': _flat})
        if verbose:
            print(f"  Probe spread: {_spread:.1f}x  [{_lo:.0f}s .. {_hi:.0f}s]")
            if _flat:
                print("  !! WARNING: duration head is NOT separating scan types on "
                      "unpadded input — this checkpoint will reproduce flat synthetic "
                      "durations. Stop and investigate before running step 05.")
    try:
        with open(os.path.join(save_dir, 'duration_probe.json'), 'w') as _pf:
            json.dump(probe, _pf, indent=2)
        if verbose:
            print(f"  Wrote duration probe → {os.path.join(save_dir, 'duration_probe.json')}")
    except Exception as _e:  # never let an artifact-write failure kill a 24h train
        if verbose:
            print(f"  (could not write duration_probe.json: {_e})")
    # --------------------------------------------------------------------------

    if verbose:
        print(f"\nTraining complete. Models saved to {save_dir}")

    return model, history


if __name__ == "__main__":
    print("Training Examination Model...")
    print("=" * 60)

    model, history = train_examination_model(verbose=True)

    print("\nFinal Results:")
    print(f"Best validation loss: {min(history['val_loss']):.4f}")
    print(f"Best validation perplexity: {min(history['val_perplexity']):.2f}")
