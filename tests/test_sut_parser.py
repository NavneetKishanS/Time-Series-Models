import ast
import importlib.util
import os
import re
import unittest


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PARSER_PATH = os.path.join(
    _REPO_ROOT, 'DatabricksPipeline', 'csv_pipeline_seqparams', '03_build_preprocessed_pkl.py',
)
_CONFIG_PATH = os.path.join(
    _REPO_ROOT, 'DatabricksPipeline', 'csv_pipeline_seqparams', 'config.py',
)


def _load_config():
    spec = importlib.util.spec_from_file_location(
        'csv_pipeline_seqparams_config', _CONFIG_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_sut_parsers():
    """Extract and exec just the SUT parsing helpers from the Databricks
    notebook source, without running the rest of the file — the rest
    unconditionally queries Spark at module level and can't be imported outside
    Databricks. Using the real source (via ast) rather than a hand-copied regex
    keeps this test tied to the actual implementation.
    """
    config = _load_config()
    with open(_PARSER_PATH) as f:
        source = f.read()
    tree = ast.parse(source)
    wanted = {
        '_SUT_TOKEN_RE', '_parse_sut_message',
        '_parse_sut_raw', '_parse_sut_categoricals',
    }
    namespace = {
        're': re,
        'SUT_FIELD_MAP': config.SUT_FIELD_MAP,
        'SUT_CATEGORICAL_FIELD_MAP': config.SUT_CATEGORICAL_FIELD_MAP,
    }
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            segment = ast.get_source_segment(source, node)
        elif (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in wanted
        ):
            segment = ast.get_source_segment(source, node)
        else:
            continue
        exec(compile(segment, _PARSER_PATH, 'exec'), namespace)
    missing = wanted - set(namespace)
    if missing:
        raise AssertionError(f"not found in {_PARSER_PATH}: {sorted(missing)}")
    return namespace


# Real MRI_SUT_1005 messages from sut_parameter_discovery.py's STEP 1 output
# (serial 176148, 2024-01-02), with the CSV-export quoting stripped — the
# actual Spark `Message` column contains the bare text without those quotes.
_HASTE_MSG_1 = (
    "Protocol: ( VER:62010002 DLL:%SiemensSeq%\\haste CS:BY1-3.SP3-5 OR:CT SG:2 "
    "SLC:9 SLT:10 FOV:300 PFOV:300 TR:866 TE:101 TEU:101000 POS:30 AVG:1 CONC:1 "
    "SP:0 TP:-1813 FA:110 FWC:1 WRP:1 REP:0 RCM:1 BR:320 PEL:333 PPF:1 SPF:16 "
    "I2D:0 PAT:2 PATP:2 PRS:512 PSN:2 FLTD:2 AAR:1 AAM:1 RSAT:0 ACC:2 CMM:1 "
    "CDM:1 B0:1 BCE:2 BCM:4 B1:1 ADJM:1 UT:655 GAI:1 PHYS:1 PAC:4 DYMO:1 DIM:2 "
    "CO:1 FC:1 DW:2200 BW:710 TF:256 LPS:1 RFT:2 GM:1 MP:30 HYP:0 PHAPS:1 "
    "MSM:4 ES:5920 ST:8 TST:9 SNR:87 B1S:40 B1SM:9 UI0:145 UI1:110 MUID:17 !s!)"
)
_HASTE_MSG_2 = (
    "Protocol: ( VER:62010002 DLL:%SiemensSeq%\\haste CS:BY1-3.SP4-6 OR:SCT SG:3 "
    "SLC:15 SLT:8 FOV:300 PFOV:300 TR:1000 TE:101 TEU:101000 POS:30 AVG:1 "
    "CONC:1 SP:-58 TP:-1871 FA:110 FWC:1 WRP:1 REP:0 RCM:1 BR:320 PEL:333 "
    "PPF:1 SPF:16 I2D:0 PAT:2 PATP:2 PRS:512 PSN:2 FLTD:2 AAR:1 AAM:1 RSAT:0 "
    "ACC:2 CMM:1 CDM:1 B0:1 BCE:2 BCM:4 B1:1 ADJM:1 UT:683 GAI:1 PHYS:1 PAC:4 "
    "DYMO:1 DIM:2 CO:1 FC:1 DW:2200 BW:710 TF:256 LPS:1 RFT:2 GM:1 MP:30 "
    "HYP:0 PHAPS:1 MSM:4 ES:5920 ST:15 TST:17 SNR:69 B1S:37 B1SM:13 UI0:145 "
    "UI1:110 MUID:25 !s!)"
)
_EP2D_DIFF_MSG = (
    "Protocol: ( VER:62010002 DLL:%SiemensSeq%\\ep2d_diff CS:BY2,3.SP4,5 OR:T "
    "SG:1 SLC:18 SLT:5 FOV:200 PFOV:101 TR:4300 TE:70 TEU:70000 POS:20 AVG:1 "
    "CONC:1 SP:-58 TP:-1871 FA:90 FWC:4 FSM:2 FSO:1 REP:0 RCM:1 BR:130 PEL:80 "
    "PPF:8 SPF:16 I2D:0 PAT:2 PATP:2 PRS:256 PSN:2 FLTD:2 FLTR:1 AAR:1 AAM:1 "
    "RSAT:0 ACC:2 CMM:1 CDM:1 B0:2 BCE:2 BCM:4 B1:1 ADJM:1 UT:683 GAI:1 "
    "PHYS:1 PAC:4 DYMO:1 DIM:2 DW:2500 BW:1538 TOM:2 EF:128 LPS:1 RFT:2 "
    "GM:17 GSRT:7 DIFF:1024 MSM:2 EXP:64 ES:960 ST:400 TST:401 SNR:39 NDW:1 "
    "BV0:2000 BVM:2000 DSH:2 B1S:7 B1SM:7 MUID:30 !s!)"
)


class ParseSutMessageTests(unittest.TestCase):
    def setUp(self):
        self.parse = _load_sut_parsers()['_parse_sut_message']

    def test_parses_the_mapped_fields_from_haste_message(self):
        result = self.parse(_HASTE_MSG_1)
        self.assertEqual(result, {
            'TR': '866', 'num_slices': '9', 'ST': '8', 'TST': '9',
            'slice_thickness': '10', 'averages': '1', 'concatenations': '1',
            'phase_encoding_lines': '333', 'parallel_imaging_factor': '2',
            'parallel_imaging_phase': '2', 'acceleration_factor': '2',
            'field_of_view': '300', 'turbo_factor': '256', 'TE': '101',
            'flip_angle': '110', 'base_resolution': '320', 'repetitions': '0',
            'phase_partial_fourier': '1', 'slice_partial_fourier': '16',
            'echo_spacing': '5920', 'bandwidth': '710',
        })

    def test_parses_different_values_from_a_second_haste_message(self):
        result = self.parse(_HASTE_MSG_2)
        self.assertEqual(result, {
            'TR': '1000', 'num_slices': '15', 'ST': '15', 'TST': '17',
            'slice_thickness': '8', 'averages': '1', 'concatenations': '1',
            'phase_encoding_lines': '333', 'parallel_imaging_factor': '2',
            'parallel_imaging_phase': '2', 'acceleration_factor': '2',
            'field_of_view': '300', 'turbo_factor': '256', 'TE': '101',
            'flip_angle': '110', 'base_resolution': '320', 'repetitions': '0',
            'phase_partial_fourier': '1', 'slice_partial_fourier': '16',
            'echo_spacing': '5920', 'bandwidth': '710',
        })

    def test_parses_a_different_sequence_type(self):
        # ep2d_diff carries a different field set entirely (DIFF, BV0, EF, ...)
        # from haste — every mapped key present must still resolve by name.
        result = self.parse(_EP2D_DIFF_MSG)
        self.assertEqual(result, {
            'TR': '4300', 'num_slices': '18', 'ST': '400', 'TST': '401',
            'slice_thickness': '5', 'averages': '1', 'concatenations': '1',
            'phase_encoding_lines': '80', 'parallel_imaging_factor': '2',
            'parallel_imaging_phase': '2', 'acceleration_factor': '2',
            'field_of_view': '200', 'TE': '70', 'flip_angle': '90',
            'base_resolution': '130', 'repetitions': '0',
            'phase_partial_fourier': '8', 'slice_partial_fourier': '16',
            'echo_spacing': '960', 'bandwidth': '1538',
        })

    def test_turbo_factor_is_sequence_scoped_not_universal(self):
        """TF is on the TSE-family (haste) messages and absent from ep2d_diff.

        EPI has no turbo factor — it uses an echo factor (EF) instead. This
        matters because _safe_float(sut_values.get('turbo_factor')) defaults a
        missing key to 0.0, and a turbo factor of 0 is not a real value, it
        means "does not apply to this sequence". Harmless while turbo_factor
        stays out of EXAMINATION_SEQPARAM_FEATURES; promoting it needs an
        explicit presence flag or scoping by sequence family. Pinned so the
        asymmetry is visible rather than discovered as a silent zero column.
        """
        self.assertEqual(self.parse(_HASTE_MSG_1)['turbo_factor'], '256')
        self.assertNotIn('turbo_factor', self.parse(_EP2D_DIFF_MSG))
        self.assertIn('EF:', _EP2D_DIFF_MSG)

    def test_st_is_a_computed_acquisition_time_for_haste(self):
        """SLC x TR reproduces ST on both haste messages.

        ST is CONFIRMED (2026-07-31) to be decided before the measurement — a
        planned value, not an observed outcome — so it is not the SD58 leak and
        is admissible as a model feature. This test keeps the supporting
        arithmetic visible: a measured elapsed time would not land on the
        computed product to the second.
        """
        for message in (_HASTE_MSG_1, _HASTE_MSG_2):
            parsed = self.parse(message)
            computed = int(parsed['num_slices']) * int(parsed['TR']) / 1000.0
            self.assertAlmostEqual(computed, float(parsed['ST']), delta=0.25)

    def test_st_is_not_merely_slices_times_tr(self):
        """The haste identity does NOT generalise — ep2d_diff breaks it.

        18 slices x 4300ms = 77.4s against ST:400. Diffusion covers all slices
        per TR and repeats over directions/b-values, so ST is a real
        per-sequence-family computation rather than a redundant product of two
        fields already in the feature set. That is why ST is worth having at
        all, and why nobody should "simplify" it away as derivable.
        """
        parsed = self.parse(_EP2D_DIFF_MSG)
        computed = int(parsed['num_slices']) * int(parsed['TR']) / 1000.0
        self.assertLess(computed, float(parsed['ST']) / 4)

    def test_ignores_unmapped_tokens(self):
        result = self.parse(_HASTE_MSG_1)
        for unmapped in ('MUID', 'VER', 'SNR', 'POS', 'DLL', 'OR'):
            self.assertNotIn(unmapped, result)

    def test_non_string_message_returns_empty_dict(self):
        self.assertEqual(self.parse(None), {})
        self.assertEqual(self.parse(float('nan')), {})


class ParseSutRawTests(unittest.TestCase):
    """The unfiltered capture — what makes the parameter route testable offline.

    Görtler (2026-07-31) wants the general duration model on sequence +
    sequence parameters rather than the customer-authored protocol name. Each
    candidate field would otherwise cost a Spark rebuild to evaluate, so step
    03 keeps every token.
    """

    def setUp(self):
        self.raw = _load_sut_parsers()['_parse_sut_raw']

    def test_captures_far_more_than_the_mapped_subset(self):
        for message in (_HASTE_MSG_1, _HASTE_MSG_2, _EP2D_DIFF_MSG):
            self.assertGreaterEqual(len(self.raw(message)), 50)

    def test_captures_the_fields_the_mapped_parse_drops(self):
        result = self.raw(_HASTE_MSG_1)
        # DLL/OR are the customer-agnostic categoricals; SNR/MUID/VER are
        # unmapped on purpose but must still survive the raw capture.
        for key in ('DLL', 'OR', 'SNR', 'MUID', 'VER', 'ES', 'TF'):
            self.assertIn(key, result)

    def test_does_not_capture_the_wrapper_word_protocol(self):
        # The tokens are wrapped in "Protocol: ( ... !s!)" — "Protocol" is
        # followed by a space, so it must not be read as a KEY:VALUE pair.
        self.assertNotIn('Protocol', self.raw(_HASTE_MSG_1))

    def test_non_string_message_returns_empty_dict(self):
        self.assertEqual(self.raw(None), {})
        self.assertEqual(self.raw(float('nan')), {})


class ParseSutCategoricalsTests(unittest.TestCase):
    def setUp(self):
        self.cats = _load_sut_parsers()['_parse_sut_categoricals']

    def test_extracts_the_sequence_binary_leaf_from_the_dll_path(self):
        self.assertEqual(
            self.cats(_HASTE_MSG_1)['sequence_binary'], 'haste')
        self.assertEqual(
            self.cats(_EP2D_DIFF_MSG)['sequence_binary'], 'ep2d_diff')

    def test_extracts_orientation(self):
        self.assertEqual(self.cats(_HASTE_MSG_1)['orientation'], 'CT')
        self.assertEqual(self.cats(_HASTE_MSG_2)['orientation'], 'SCT')
        self.assertEqual(self.cats(_EP2D_DIFF_MSG)['orientation'], 'T')

    def test_returns_strings_not_ids(self):
        """Ids come from a vocab frozen at train time, never from step 03.

        Same rule as protocol_name: if the pkl carried ids, a vocabulary change
        would silently remap a checkpoint's embedding rows.
        """
        for value in self.cats(_HASTE_MSG_1).values():
            self.assertIsInstance(value, str)

    def test_non_string_message_returns_empty_dict(self):
        self.assertEqual(self.cats(None), {})
        self.assertEqual(self.cats(float('nan')), {})


if __name__ == '__main__':
    unittest.main()
