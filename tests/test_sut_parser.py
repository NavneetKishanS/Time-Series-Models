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


def _load_parse_sut_message():
    """Extract and exec just `_SUT_TOKEN_RE` / `_parse_sut_message` from the
    Databricks notebook source, without running the rest of the file — the
    rest unconditionally queries Spark at module level and can't be imported
    outside Databricks. Using the real source (via ast) rather than a
    hand-copied regex keeps this test tied to the actual implementation.
    """
    config = _load_config()
    with open(_PARSER_PATH) as f:
        source = f.read()
    tree = ast.parse(source)
    wanted = {'_SUT_TOKEN_RE', '_parse_sut_message'}
    namespace = {'re': re, 'SUT_FIELD_MAP': config.SUT_FIELD_MAP}
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
    return namespace['_parse_sut_message']


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
        self.parse = _load_parse_sut_message()

    def test_parses_tr_and_num_slices_from_haste_message(self):
        result = self.parse(_HASTE_MSG_1)
        self.assertEqual(result, {'TR': '866', 'num_slices': '9'})

    def test_parses_different_values_from_a_second_haste_message(self):
        result = self.parse(_HASTE_MSG_2)
        self.assertEqual(result, {'TR': '1000', 'num_slices': '15'})

    def test_parses_tr_and_num_slices_from_a_different_sequence_type(self):
        # ep2d_diff carries a different field set entirely (DIFF, BV0, EF, ...)
        # from haste — TR/SLC must still resolve correctly by name.
        result = self.parse(_EP2D_DIFF_MSG)
        self.assertEqual(result, {'TR': '4300', 'num_slices': '18'})

    def test_ignores_unmapped_tokens(self):
        result = self.parse(_HASTE_MSG_1)
        self.assertNotIn('FOV', result)
        self.assertNotIn('MUID', result)
        self.assertEqual(set(result), {'TR', 'num_slices'})

    def test_non_string_message_returns_empty_dict(self):
        self.assertEqual(self.parse(None), {})
        self.assertEqual(self.parse(float('nan')), {})


if __name__ == '__main__':
    unittest.main()
