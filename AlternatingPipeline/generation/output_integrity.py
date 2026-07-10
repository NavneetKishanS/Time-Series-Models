"""Strict, dependency-free checks for generated token streams and CSV rows."""

from collections import Counter, defaultdict
from datetime import datetime
import math


class GenerationIntegrityError(RuntimeError):
    """Raised when generated content cannot be rendered without data loss."""


def _token_list(tokens):
    if hasattr(tokens, "detach"):
        tokens = tokens.detach().cpu().tolist()
    return [int(token) for token in tokens]


def _closed_content(tokens, start_token_id, end_token_id, pad_token_id):
    sequence = _token_list(tokens)
    if not sequence or sequence[0] != start_token_id:
        raise GenerationIntegrityError("sequence does not begin with START")

    try:
        end_idx = sequence.index(end_token_id)
    except ValueError as exc:
        raise GenerationIntegrityError("sequence does not contain END") from exc

    trailing = sequence[end_idx + 1:]
    if any(token not in (end_token_id, pad_token_id) for token in trailing):
        raise GenerationIntegrityError("sequence contains events after END")

    content = sequence[1:end_idx]
    if any(token in (start_token_id, end_token_id, pad_token_id) for token in content):
        raise GenerationIntegrityError("sequence contains a control token inside its payload")
    return content


def validate_examination_sequence(tokens, start_token_id, end_token_id,
                                  pad_token_id, measurement_start_id,
                                  finish_token_ids, unknown_token_id=None):
    """Return the finish token after validating one complete measurement span."""
    content = _closed_content(
        tokens, start_token_id, end_token_id, pad_token_id
    )
    finish_token_ids = {int(token) for token in finish_token_ids}

    if not content or content[0] != measurement_start_id:
        raise GenerationIntegrityError("examination does not open with MRI_MSR_100")
    if content.count(measurement_start_id) != 1:
        raise GenerationIntegrityError("examination contains multiple MRI_MSR_100 starts")
    if content[-1] not in finish_token_ids:
        raise GenerationIntegrityError("examination does not close with MRI_MSR_104/34")
    if sum(token in finish_token_ids for token in content) != 1:
        raise GenerationIntegrityError("examination contains multiple finish events")
    if unknown_token_id is not None and unknown_token_id in content:
        raise GenerationIntegrityError("examination contains an UNK event")
    return content[-1]


def repair_examination_sequence(tokens, start_token_id, end_token_id,
                                pad_token_id, measurement_start_id,
                                finish_token_ids, unknown_token_id=None):
    """Normalize a malformed sample to one renderable measurement span.

    The latest measurement start preceding a sampled finish wins, matching the
    recovery behavior of the original renderer without silently dropping a
    scan. The returned duration index points into the original shifted duration
    arrays and is ``None`` when the model did not sample a finish token.
    """
    sequence = _token_list(tokens)
    finish_token_ids = tuple(int(token) for token in finish_token_ids)
    if not finish_token_ids:
        raise ValueError("finish_token_ids must contain at least one token")

    finish_idx = next(
        (idx for idx, token in enumerate(sequence) if token in finish_token_ids),
        None,
    )
    finish_token = (
        sequence[finish_idx] if finish_idx is not None else finish_token_ids[0]
    )

    if finish_idx is None:
        payload_end = sequence.index(end_token_id) if end_token_id in sequence else len(sequence)
    else:
        payload_end = finish_idx

    preceding_starts = [
        idx for idx, token in enumerate(sequence[:payload_end])
        if token == measurement_start_id
    ]
    payload_start = preceding_starts[-1] + 1 if preceding_starts else 0

    blocked = {
        start_token_id,
        end_token_id,
        pad_token_id,
        measurement_start_id,
        *finish_token_ids,
    }
    if unknown_token_id is not None:
        blocked.add(int(unknown_token_id))
    events = [
        token for token in sequence[payload_start:payload_end]
        if token not in blocked
    ]

    repaired = [
        int(start_token_id),
        int(measurement_start_id),
        *events,
        int(finish_token),
        int(end_token_id),
    ]
    duration_source_idx = finish_idx - 1 if finish_idx is not None and finish_idx > 0 else None

    validate_examination_sequence(
        repaired,
        start_token_id,
        end_token_id,
        pad_token_id,
        measurement_start_id,
        finish_token_ids,
        unknown_token_id,
    )
    return repaired, duration_source_idx


def validate_exchange_sequence(tokens, start_token_id, end_token_id,
                               pad_token_id, unknown_token_id=None):
    """Validate that an exchange is closed and contains renderable events."""
    content = _closed_content(
        tokens, start_token_id, end_token_id, pad_token_id
    )
    if not content:
        raise GenerationIntegrityError("exchange contains no events")
    if unknown_token_id is not None and unknown_token_id in content:
        raise GenerationIntegrityError("exchange contains an UNK event")
    return content


def validate_orchestration_sequence(tokens, start_token_id, end_token_id,
                                    pad_token_id, break_token_id,
                                    allowed_region_ids):
    """Return body regions after validating the full day-plan grammar."""
    content = _closed_content(
        tokens, start_token_id, end_token_id, pad_token_id
    )
    allowed_region_ids = {int(token) for token in allowed_region_ids}
    if not content:
        raise GenerationIntegrityError("orchestration produced an empty day")
    if content[0] == break_token_id or content[-1] == break_token_id:
        raise GenerationIntegrityError("orchestration starts or ends with BREAK")

    previous = None
    body_regions = []
    for token in content:
        if token == break_token_id:
            if previous == break_token_id:
                raise GenerationIntegrityError("orchestration contains consecutive BREAK tokens")
        elif token in allowed_region_ids:
            body_regions.append(token)
        else:
            raise GenerationIntegrityError(
                f"orchestration emitted unsupported region token {token}"
            )
        previous = token

    if not body_regions:
        raise GenerationIntegrityError("orchestration contains no patients")
    return body_regions


def _parse_datetime(value, field_name, errors):
    try:
        return datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        errors.append(f"invalid {field_name}: {value!r}")
        return None


def validate_rendered_output(exchange_rows, exam_rows, expected_exam_counts,
                             expected_exchange_blocks, valid_body_regions):
    """Validate row-level schema invariants and planned-vs-rendered counts."""
    errors = []
    expected_exam_counts = Counter({
        str(patient_id): int(count)
        for patient_id, count in expected_exam_counts.items()
    })
    actual_exam_counts = Counter(str(row.get('PatientID')) for row in exam_rows)

    if actual_exam_counts != expected_exam_counts:
        missing = expected_exam_counts - actual_exam_counts
        extra = actual_exam_counts - expected_exam_counts
        errors.append(
            f"planned/rendered scan counts differ; missing={dict(missing)}, extra={dict(extra)}"
        )

    valid_body_regions = {str(region) for region in valid_body_regions}
    patient_steps = defaultdict(list)
    previous_exam_end = None
    for row_idx, row in enumerate(exam_rows):
        patient_id = str(row.get('PatientID'))
        patient_steps[patient_id].append(row.get('StepCount'))

        try:
            duration = float(row.get('duration'))
            if not math.isfinite(duration) or duration <= 0:
                errors.append(f"exam row {row_idx} has invalid duration {duration!r}")
        except (TypeError, ValueError):
            errors.append(f"exam row {row_idx} has non-numeric duration")

        start = _parse_datetime(row.get('startTime'), 'startTime', errors)
        end = _parse_datetime(row.get('endTime'), 'endTime', errors)
        if start is not None and end is not None and end <= start:
            errors.append(f"exam row {row_idx} does not end after it starts")
        if start is not None and previous_exam_end is not None and start < previous_exam_end:
            errors.append(f"exam row {row_idx} overlaps the preceding examination")
        if end is not None:
            previous_exam_end = end

        finish_event = row.get('FinishEvent')
        if finish_event not in ('Successful', 'Stopped by User'):
            errors.append(f"exam row {row_idx} has invalid FinishEvent")
        expected_source = {
            'Successful': 'MRI_MSR_104',
            'Stopped by User': 'MRI_MSR_34',
        }.get(finish_event)
        if expected_source is not None and row.get('sourceID') != expected_source:
            errors.append(f"exam row {row_idx} has inconsistent sourceID/FinishEvent")
        if str(row.get('BodyPart')) not in valid_body_regions:
            errors.append(f"exam row {row_idx} has invalid BodyPart {row.get('BodyPart')!r}")

    for patient_id, steps in patient_steps.items():
        try:
            ordered = sorted(int(step) for step in steps)
        except (TypeError, ValueError):
            errors.append(f"patient {patient_id} has non-numeric StepCount")
            continue
        if ordered != list(range(1, len(ordered) + 1)):
            errors.append(f"patient {patient_id} has non-contiguous StepCount values")

    exchange_blocks = defaultdict(list)
    for row_idx, row in enumerate(exchange_rows):
        sample_idx = row.get('sample_idx')
        exchange_blocks[sample_idx].append(row)
        token_name_value = row.get('token_name')
        token_name = str(token_name_value)
        if not token_name_value or token_name in ('START', 'END', 'PAD', 'UNK'):
            errors.append(f"exchange row {row_idx} contains {token_name}")
        try:
            duration = float(row.get('sampled_duration'))
            if not math.isfinite(duration) or duration < 0:
                errors.append(f"exchange row {row_idx} has invalid sampled_duration")
        except (TypeError, ValueError):
            errors.append(f"exchange row {row_idx} has non-numeric sampled_duration")
        _parse_datetime(row.get('datetime'), 'exchange datetime', errors)

    if len(exchange_blocks) != int(expected_exchange_blocks):
        errors.append(
            f"planned {expected_exchange_blocks} exchange blocks but rendered "
            f"{len(exchange_blocks)}"
        )

    for sample_idx, rows in exchange_blocks.items():
        try:
            steps = sorted(int(row.get('step')) for row in rows)
        except (TypeError, ValueError):
            errors.append(f"exchange block {sample_idx} has non-numeric steps")
            continue
        if steps != list(range(len(steps))):
            errors.append(f"exchange block {sample_idx} has non-contiguous steps")
        ordered_rows = sorted(rows, key=lambda row: int(row.get('step')))
        timestamps = []
        for row in ordered_rows:
            timestamp = _parse_datetime(
                row.get('datetime'), 'exchange datetime', errors
            )
            if timestamp is not None:
                timestamps.append(timestamp)
        if any(current < previous for previous, current in zip(timestamps, timestamps[1:])):
            errors.append(f"exchange block {sample_idx} has non-monotonic timestamps")

    if errors:
        preview = '\n'.join(f"  - {error}" for error in errors[:20])
        remainder = len(errors) - 20
        suffix = f"\n  - ... and {remainder} more" if remainder > 0 else ""
        raise GenerationIntegrityError(
            f"Synthetic output integrity check failed ({len(errors)} issue(s)):\n"
            f"{preview}{suffix}"
        )

    return {
        'patients': len(expected_exam_counts),
        'exam_rows': len(exam_rows),
        'exchange_rows': len(exchange_rows),
        'exchange_blocks': len(exchange_blocks),
    }
