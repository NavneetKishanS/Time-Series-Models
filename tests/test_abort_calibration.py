from AlternatingPipeline.generation.abort_calibration import (
    build_abort_rate_table,
    select_abort_rate,
)


ABORT = 15


def _row(serial=0, region=1, sequence_type=2, abort=False):
    return {
        'serial_idx': serial,
        'body_region': region,
        'sequence_type': sequence_type,
        'sequence': [1, ABORT] if abort else [1, 12],
    }


def test_uses_specific_stratum_when_supported():
    rows = [_row(abort=i < 10) for i in range(50)]
    rows += [_row(serial=9, region=9, sequence_type=9, abort=False) for _ in range(100)]
    table = build_abort_rate_table(rows, ABORT)
    rate, key, meta = select_abort_rate(table, 0, 1, 2)
    assert key == ('serial_region_type', (0, 1, 2))
    assert meta['total'] == 50
    assert 0.20 < rate < 0.22


def test_falls_back_to_global_for_thin_strata():
    rows = [_row(abort=i < 3) for i in range(100)]
    table = build_abort_rate_table(rows, ABORT)
    rate, key, meta = select_abort_rate(table, 7, 8, 9)
    assert key == ('global', ())
    assert meta['aborts'] == 3
    assert 0.03 < rate < 0.04
