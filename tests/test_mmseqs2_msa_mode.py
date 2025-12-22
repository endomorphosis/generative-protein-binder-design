import os
import tempfile
import unittest
from dataclasses import dataclass
import types
import sys
import logging as _py_logging
from pathlib import Path

import numpy as np


class _DummyTemplateSearcher:
    # The mmseqs2 path requires A3M templates.
    input_format = "a3m"
    output_format = "hhr"

    def query(self, msa_for_templates: str) -> str:
        # Return a non-empty blob that downstream code can pass around.
        if not isinstance(msa_for_templates, str) or not msa_for_templates.strip():
            raise ValueError("expected non-empty msa_for_templates")
        return "dummy_hhr"

    def get_template_hits(self, *, output_string: str, input_sequence: str):
        # Feature pipeline only forwards this to the featurizer; structure is irrelevant for the smoke test.
        return ["dummy_hit"]


@dataclass
class _DummyTemplatesResult:
    features: dict


class _DummyTemplateFeaturizer:
    def get_templates(self, *, query_sequence: str, hits):
        # pipeline.py logs template_domain_names shape[0]
        feats = {
            "template_domain_names": np.array([b"dummy"], dtype=np.object_),
        }
        return _DummyTemplatesResult(features=feats)


class _DummyMMseqs2Runner:
    def __init__(self, a3m: str):
        self._a3m = a3m

    def query(self, input_fasta_path: str):
        # Must return list[dict] with 'a3m'
        return [{"a3m": self._a3m}]


class TestMMseqs2MSAModeSmoke(unittest.TestCase):
    def test_pipeline_mmseqs2_mode_produces_features(self):
        # Ensure the bundled AlphaFold package can be imported as top-level
        # module name `alphafold`.
        repo_root = Path(__file__).resolve().parents[1]
        alphafold_root = repo_root / "tools" / "alphafold2"
        if str(alphafold_root) not in sys.path:
            sys.path.insert(0, str(alphafold_root))

        # AlphaFold's residue_constants uses `from jax import tree` and then
        # `tree.map(...)`. For this smoke test we don't need real JAX.
        try:
            import jax  # noqa: F401
        except Exception:
            def _tree_map(fn, x):
                if isinstance(x, list):
                    return [_tree_map(fn, v) for v in x]
                if isinstance(x, tuple):
                    return tuple(_tree_map(fn, v) for v in x)
                if isinstance(x, dict):
                    return {k: _tree_map(fn, v) for k, v in x.items()}
                return fn(x)

            jax_mod = types.ModuleType("jax")
            tree_mod = types.ModuleType("jax.tree")
            tree_mod.map = _tree_map
            jax_mod.tree = tree_mod
            sys.modules["jax"] = jax_mod
            sys.modules["jax.tree"] = tree_mod

        # AlphaFold's real templates module pulls in Biopython. For this
        # MMseqs2 MSA pipeline smoke test we can stub templates entirely.
        if "alphafold.data.templates" not in sys.modules:
            templates_mod = types.ModuleType("alphafold.data.templates")

            class TemplateHitFeaturizer:  # noqa: N801
                pass

            class HmmsearchHitFeaturizer(TemplateHitFeaturizer):  # noqa: N801
                pass

            class HhsearchHitFeaturizer(TemplateHitFeaturizer):  # noqa: N801
                pass

            templates_mod.TemplateHitFeaturizer = TemplateHitFeaturizer
            templates_mod.HmmsearchHitFeaturizer = HmmsearchHitFeaturizer
            templates_mod.HhsearchHitFeaturizer = HhsearchHitFeaturizer
            sys.modules["alphafold.data.templates"] = templates_mod

        # AlphaFold's code uses `absl.logging`. In minimal environments (like this
        # repo's base requirements), `absl-py` may not be installed.
        # For a pipeline smoke test, a tiny stub is sufficient.
        try:
            import absl  # noqa: F401
        except Exception:
            absl_mod = types.ModuleType("absl")

            class _AbslLoggingShim:
                def set_verbosity(self, *_args, **_kwargs):
                    return None

                def info(self, msg, *args, **kwargs):
                    _py_logging.getLogger("absl").info(msg, *args, **kwargs)

                def warning(self, msg, *args, **kwargs):
                    _py_logging.getLogger("absl").warning(msg, *args, **kwargs)

                def error(self, msg, *args, **kwargs):
                    _py_logging.getLogger("absl").error(msg, *args, **kwargs)

            absl_logging_mod = _AbslLoggingShim()
            absl_mod.logging = absl_logging_mod
            sys.modules["absl"] = absl_mod
            sys.modules["absl.logging"] = absl_logging_mod

        # Import inside the test so it’s easy to run from repo root.
        from alphafold.data import pipeline as af_pipeline

        # Minimal valid A3M with query + one hit.
        a3m = ">query\nACDE\n>hit1\nAC-E\n"

        dp = af_pipeline.DataPipeline(
            jackhmmer_binary_path="/bin/false",
            hhblits_binary_path="/bin/false",
            uniref90_database_path="/dev/null",
            mgnify_database_path="/dev/null",
            bfd_database_path=None,
            uniref30_database_path=None,
            small_bfd_database_path=None,
            template_searcher=_DummyTemplateSearcher(),
            template_featurizer=_DummyTemplateFeaturizer(),
            use_small_bfd=True,
            use_precomputed_msas=False,
            msa_tools_n_cpu=1,
            msa_mode="mmseqs2",
            mmseqs2_binary_path="mmseqs",
            mmseqs2_database_path="/tmp/dummy_mmseqs_db",
            mmseqs2_max_seqs=10,
        )

        # Inject the dummy runner to avoid requiring a real mmseqs install.
        dp.mmseqs2_runner = _DummyMMseqs2Runner(a3m)

        with tempfile.TemporaryDirectory() as tmp:
            fasta_path = os.path.join(tmp, "q.fasta")
            msa_dir = os.path.join(tmp, "msas")
            os.makedirs(msa_dir, exist_ok=True)
            with open(fasta_path, "w", encoding="utf-8") as f:
                f.write(">q\nACDE\n")

            features = dp.process(fasta_path, msa_dir)

        # Sanity checks: these are standard AlphaFold feature keys.
        self.assertIn("msa", features)
        self.assertIn("deletion_matrix_int", features)
        self.assertIn("num_alignments", features)
        self.assertIn("template_domain_names", features)

        # Ensure we got at least the query alignment.
        self.assertGreaterEqual(int(features["num_alignments"][0]), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
