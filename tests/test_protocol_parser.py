"""Tests for `_protocol_from_msg` in csv_pipeline_seqparams/03.

The protocol name explains 82.0% of examination-duration variance held out
(MAE 17.2s) against 31.2% for the sequence_type bucket the model conditions on
today, so this parser is now on the critical path for the duration model.
"""

import ast
import os
import re
import unittest


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PARSER_PATH = os.path.join(
    _REPO_ROOT, 'DatabricksPipeline', 'csv_pipeline_seqparams',
    '03_build_preprocessed_pkl.py',
)


def _load_protocol_from_msg():
    """Extract just `_protocol_from_msg` from the notebook source.

    Same approach as test_sut_parser: the rest of the file queries Spark at
    module level and cannot be imported outside Databricks, and pulling the
    real source via ast keeps this tied to the actual implementation instead of
    a hand-copied regex.
    """
    with open(_PARSER_PATH) as f:
        source = f.read()
    tree = ast.parse(source)
    namespace = {'re': re}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == '_protocol_from_msg':
            exec(compile(ast.get_source_segment(source, node), _PARSER_PATH, 'exec'),
                 namespace)
    return namespace['_protocol_from_msg']


# Synthesised from the documented MRI_MSR_100 field layout — these are NOT
# captured messages. The field ORDER matters and is inferred from a working
# production regex: csv_pipeline/02_exam_preprocessing.py:102 reads Sequence
# with a greedy `'(.*)'` and yields clean values, which is only possible if
# Sequence is the last quoted field, i.e. Protocol comes before it. The
# protocol names themselves are real values taken from the exam CSVs.
def _msr_100(protocol, sequence=r'%SiemensSeq%\haste'):
    return (
        f"Measurement started. Protocol: '{protocol}', "
        f"Sequence: '{sequence}'"
    )


class ProtocolFromMsgTests(unittest.TestCase):
    def setUp(self):
        self.parse = _load_protocol_from_msg()

    def test_extracts_the_protocol_name(self):
        self.assertEqual(
            self.parse(_msr_100('LOCA HASTE RESPI LIBRE')),
            'LOCA HASTE RESPI LIBRE',
        )

    def test_does_not_run_past_the_closing_quote_into_sequence(self):
        """02's greedy `'(.*)',` can over-match; `[^']*` cannot."""
        self.assertEqual(self.parse(_msr_100('T1 VIBE')), 'T1 VIBE')

    def test_handles_names_containing_commas(self):
        """14 real protocols contain a comma, e.g. 'TOF 0,5mm ISO'."""
        for name in ('TOF 0,5mm ISO',
                     'DWI RESOLVE  (B 50, 800, 1200)',
                     't1_vibe_dixon_tra_post 0, 30, 60, 90, 120'):
            self.assertEqual(self.parse(_msr_100(name)), name)

    def test_handles_accented_and_underscored_names(self):
        for name in ('CareBolus_départ aorte ascendante max',
                     'pd_tse_fs_tra_DRB_WARP',
                     'AAhead_scout'):
            self.assertEqual(self.parse(_msr_100(name)), name)

    def test_returns_empty_string_when_the_field_is_absent(self):
        self.assertEqual(self.parse("Measurement started. Sequence: '%SiemensSeq%\\tse'"), '')

    def test_non_string_message_returns_empty_string(self):
        for missing in (None, float('nan'), 42):
            self.assertEqual(self.parse(missing), '')

    def test_result_is_left_raw_apart_from_surrounding_whitespace(self):
        """Case and internal spacing must survive: the synthetic CSV's Protocol
        column has to match the real one exactly for Qlik to join them.
        Normalisation happens only when building the vocabulary."""
        self.assertEqual(self.parse(_msr_100('  SAG T1 DR gado  ')), 'SAG T1 DR gado')
        self.assertEqual(self.parse(_msr_100('T2_TSE_SAG')), 'T2_TSE_SAG')


if __name__ == '__main__':
    unittest.main()
