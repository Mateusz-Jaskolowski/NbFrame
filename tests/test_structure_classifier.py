"""
Test suite for NbFrame structure classifier.

Tests the structure-based CDR3 conformation classification functionality
including model loading, feature-based prediction, PDB classification,
and CLI commands.
"""
import unittest
from pathlib import Path

from typer.testing import CliRunner

import nbframe
from nbframe.structure_classifier import (
    StructureModelMetadata,
    classify_structure,
    load_structure_classifier,
    predict_structure_from_features,
)
from nbframe.cli import app as cli_app


# Path to test data directory
TEST_DATA_DIR = Path(__file__).parent / "data"
TEST_PDBS_DIR = TEST_DATA_DIR / "pdbs"


class TestStructureClassifier(unittest.TestCase):
    """Tests for the structure-based classifier runtime API and CLI."""

    def test_load_structure_classifier_and_metadata(self):
        """Test loading the structure classifier model and metadata."""
        clf, meta = load_structure_classifier()

        # Basic sanity checks on metadata
        self.assertIsInstance(meta, StructureModelMetadata)
        self.assertIsInstance(meta.feature_cols, list)
        self.assertGreater(len(meta.feature_cols), 0)
        self.assertIn("kinked", meta.label_mapping)
        self.assertIn("extended", meta.label_mapping)

        # Classifier should support predict_proba
        self.assertTrue(hasattr(clf, "predict_proba"))

    def test_predict_from_synthetic_features(self):
        """Test prediction from synthetic feature dictionary."""
        # Build a simple synthetic feature dict using metadata-defined columns.
        _, meta = load_structure_classifier()
        base_features = {name: 0.0 for name in meta.feature_cols}

        result = predict_structure_from_features(base_features)

        self.assertIn(result["label"], ("kinked", "extended"))
        p_kink = result["prob_kinked"]
        p_ext = result["prob_extended"]
        self.assertIsInstance(p_kink, float)
        self.assertIsInstance(p_ext, float)
        self.assertGreaterEqual(p_kink, 0.0)
        self.assertGreaterEqual(p_ext, 0.0)
        self.assertLessEqual(p_kink, 1.0)
        self.assertLessEqual(p_ext, 1.0)
        self.assertAlmostEqual(p_kink + p_ext, 1.0, places=5)

        # Missing or invalid features should raise a clear error.
        incomplete = dict(base_features)
        incomplete.pop(meta.feature_cols[0])
        with self.assertRaises(ValueError):
            predict_structure_from_features(incomplete)

    def test_public_api_exposes_classify_structure(self):
        """Test that classify_structure is exposed in the public API."""
        # The high-level API should be available from the package root.
        self.assertTrue(hasattr(nbframe, "classify_structure"))
        self.assertIs(nbframe.classify_structure, classify_structure)

    def test_cli_classify_structure_help(self):
        """Test CLI classify-structure command help."""
        # Smoke test: ensure the CLI command is registered and has help text.
        runner = CliRunner()
        result = runner.invoke(cli_app, ["classify-structure", "--help"])

        self.assertEqual(result.exit_code, 0)
        self.assertIn("classify-structure", result.output)
        # Check for key options
        self.assertIn("--pdb", result.output)
        self.assertIn("--pdb-dir", result.output)


class TestStructureClassifierWithPDBs(unittest.TestCase):
    """Tests for structure classifier using actual PDB files."""

    @classmethod
    def setUpClass(cls):
        """Check if test PDB files are available."""
        cls.kinked_pdbs_dir = TEST_PDBS_DIR / "kinked"
        cls.extended_pdbs_dir = TEST_PDBS_DIR / "extended"
        cls.edge_cases_dir = TEST_PDBS_DIR / "edge_cases"
        cls.mmcif_dir = TEST_PDBS_DIR / "mmcif"

        # Check for required test files
        cls.has_kinked_pdbs = cls.kinked_pdbs_dir.exists() and any(cls.kinked_pdbs_dir.glob("*.pdb"))
        cls.has_extended_pdbs = cls.extended_pdbs_dir.exists() and any(cls.extended_pdbs_dir.glob("*.pdb"))
        cls.has_edge_cases = cls.edge_cases_dir.exists()
        cls.has_mmcif = cls.mmcif_dir.exists() and any(cls.mmcif_dir.glob("*.cif"))

    def test_classify_kinked_pdb(self):
        """Test classification of known kinked PDB structures."""
        if not self.has_kinked_pdbs:
            self.skipTest("Kinked PDB test files not available")

        # Test with first available kinked PDB
        kinked_pdbs = list(self.kinked_pdbs_dir.glob("*.pdb"))
        pdb_path = kinked_pdbs[0]

        result = classify_structure(str(pdb_path))

        # Should return a valid result
        self.assertIsNotNone(result)
        self.assertIn("label", result)
        self.assertIn("prob_kinked", result)

        # Kinked structures should have high P(kinked)
        self.assertGreater(result["prob_kinked"], 0.5,
                          f"Expected kinked PDB {pdb_path.name} to have P(kinked) > 0.5")

    def test_classify_extended_pdb(self):
        """Test classification of known extended PDB structures."""
        if not self.has_extended_pdbs:
            self.skipTest("Extended PDB test files not available")

        # Test with first available extended PDB
        extended_pdbs = list(self.extended_pdbs_dir.glob("*.pdb"))
        pdb_path = extended_pdbs[0]

        result = classify_structure(str(pdb_path))

        # Should return a valid result
        self.assertIsNotNone(result)
        self.assertIn("label", result)
        self.assertIn("prob_kinked", result)

        # Extended structures should have low P(kinked)
        self.assertLess(result["prob_kinked"], 0.5,
                       f"Expected extended PDB {pdb_path.name} to have P(kinked) < 0.5")

    def test_classify_pdb_no_vhh_chains(self):
        """Test that PDB with no VHH chains raises appropriate error."""
        if not self.has_edge_cases:
            self.skipTest("Edge case PDB files not available")

        no_vhh_pdb = self.edge_cases_dir / "missing_VHH_chains_5GRY.pdb"
        if not no_vhh_pdb.exists():
            self.skipTest("missing_VHH_chains_5GRY.pdb not available")

        # Should raise an error or return None when no VHH chains found
        with self.assertRaises((ValueError, RuntimeError)):
            classify_structure(str(no_vhh_pdb))

    def test_classify_mmcif_format(self):
        """Test classification of mmCIF format files."""
        if not self.has_mmcif:
            self.skipTest("mmCIF test files not available")

        # Test with first available mmCIF file
        mmcif_files = list(self.mmcif_dir.glob("*.cif"))
        cif_path = mmcif_files[0]

        result = classify_structure(str(cif_path))

        # Should return a valid result
        self.assertIsNotNone(result)
        self.assertIn("label", result)
        self.assertIn("prob_kinked", result)
        self.assertIn(result["label"], ("kinked", "extended", "uncertain"))


if __name__ == "__main__":
    unittest.main()
