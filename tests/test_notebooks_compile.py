"""Every csv_pipeline_seqparams notebook must at least PARSE.

These files cannot be imported outside Databricks — they query Spark at module
level and rely on `%run ./config` to populate their namespace — so nothing in the
suite executes them, and a syntax error can sit in one until a cluster picks it
up twenty minutes into a run.

Compiling is cheap and catches the whole class. The specific slip that motivated
this: `{int(x.sum():,}` instead of `{int(x.sum()):,}` in an f-string inside a
report gate, which is invisible to every text-matching guard in
test_seqparams_config_guard.py because the string it looks for is still there.

SCOPED TO csv_pipeline_seqparams ON PURPOSE. `csv_pipeline/` and
`spark_pipeline/` each carry notebooks that have not parsed for a long time —
Databricks cells pasted in with a leading indent, and a `%sql` cell that was
never commented out. Those are real, but they are not this pipeline's, and
widening the glob would either turn the suite red for unrelated reasons or
invite a drive-by edit of the notebooks that own the old checkpoint and the Qlik
comparison CSVs. Widen it when those are fixed deliberately.
"""

import glob
import os
import unittest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PIPELINE_GLOBS = (
    'DatabricksPipeline/csv_pipeline_seqparams/[0-9]*.py',
    'DatabricksPipeline/csv_pipeline_seqparams/config.py',
)


def _notebook_paths():
    paths = []
    for pattern in _PIPELINE_GLOBS:
        paths.extend(glob.glob(os.path.join(_REPO, pattern)))
    return sorted(paths)


class NotebookCompileTests(unittest.TestCase):

    def test_the_glob_actually_matches_something(self):
        # A typo in the pattern would make every assertion below vacuous, which
        # is worse than not having the test at all.
        self.assertGreater(len(_notebook_paths()), 4)

    def test_every_pipeline_notebook_parses(self):
        for path in _notebook_paths():
            with self.subTest(notebook=os.path.relpath(path, _REPO)):
                with open(path) as handle:
                    source = handle.read()
                try:
                    compile(source, path, 'exec')
                except SyntaxError as error:
                    self.fail(f"{os.path.relpath(path, _REPO)}:{error.lineno} "
                              f"{error.msg}")


if __name__ == '__main__':
    unittest.main()
