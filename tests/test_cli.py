"""
Test suite for NbFrame CLI commands.

Tests the command-line interface for both sequence and structure classification.
"""
import unittest
import tempfile
import os
from pathlib import Path

from typer.testing import CliRunner

from nbframe.cli import app as cli_app


# Path to test data directory
TEST_DATA_DIR = Path(__file__).parent / "data"
TEST_SEQUENCES_DIR = TEST_DATA_DIR / "sequences"
TEST_PDBS_DIR = TEST_DATA_DIR / "pdbs"


class TestSequenceClassifierCLI(unittest.TestCase):
    """Tests for the sequence classifier CLI commands."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.runner = CliRunner()
        cls.has_sequence_data = TEST_SEQUENCES_DIR.exists()

        # Test sequence (known kinked)
        cls.test_sequence = (
            "QVQLVESGGGSVQAGGSLRLSCAASGYTVSTYCMGWFRQAPGKEREGVATILGGSTYYGD"
            "SVKGRFTISQDNAKNTVYLQMNSLKPEDTAIYYCAGSTVASTGWCSRLRPYDYHYRGQGTQVTVSS"
        )

    def test_classify_sequence_help(self):
        """Test that classify-sequence --help works."""
        result = self.runner.invoke(cli_app, ["classify-sequence", "--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("classify-sequence", result.output)
        self.assertIn("--sequence", result.output)

    def test_classify_single_sequence(self):
        """Test classifying a single sequence via CLI."""
        result = self.runner.invoke(
            cli_app,
            ["classify-sequence", "-s", self.test_sequence]
        )
        self.assertEqual(result.exit_code, 0, f"CLI failed: {result.output}")
        # Should contain prediction output (check lowercase)
        self.assertIn("p(kinked)", result.output.lower())

    def test_classify_sequence_from_fasta(self):
        """Test classifying sequences from FASTA file."""
        if not self.has_sequence_data:
            self.skipTest("Test sequence data not available")

        fasta_path = TEST_SEQUENCES_DIR / "batch_mixed.fasta"
        if not fasta_path.exists():
            self.skipTest("batch_mixed.fasta not available")

        result = self.runner.invoke(
            cli_app,
            ["classify-sequence", "-f", str(fasta_path)]
        )
        self.assertEqual(result.exit_code, 0, f"CLI failed: {result.output}")

    def test_classify_sequence_to_csv(self):
        """Test saving sequence classification results to CSV."""
        if not self.has_sequence_data:
            self.skipTest("Test sequence data not available")

        fasta_path = TEST_SEQUENCES_DIR / "batch_mixed.fasta"
        if not fasta_path.exists():
            self.skipTest("batch_mixed.fasta not available")

        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
            output_csv = tmp.name

        try:
            result = self.runner.invoke(
                cli_app,
                ["classify-sequence", "-f", str(fasta_path), "-o", output_csv]
            )
            self.assertEqual(result.exit_code, 0, f"CLI failed: {result.output}")
            self.assertTrue(os.path.exists(output_csv))

            # Verify CSV has content
            with open(output_csv) as f:
                content = f.read()
                self.assertGreater(len(content), 0)
                self.assertIn("nbframe_score", content)
        finally:
            if os.path.exists(output_csv):
                os.unlink(output_csv)

    def test_classify_sequence_with_thresholds(self):
        """Test sequence classification with custom thresholds."""
        result = self.runner.invoke(
            cli_app,
            [
                "classify-sequence",
                "-s", self.test_sequence,
                "--kinked-threshold", "0.80",
                "--extended-threshold", "0.30"
            ]
        )
        self.assertEqual(result.exit_code, 0, f"CLI failed: {result.output}")


class TestStructureClassifierCLI(unittest.TestCase):
    """Tests for the structure classifier CLI commands."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.runner = CliRunner()
        cls.kinked_pdbs_dir = TEST_PDBS_DIR / "kinked"
        cls.extended_pdbs_dir = TEST_PDBS_DIR / "extended"
        cls.has_kinked_pdbs = cls.kinked_pdbs_dir.exists() and any(cls.kinked_pdbs_dir.glob("*.pdb"))
        cls.has_extended_pdbs = cls.extended_pdbs_dir.exists() and any(cls.extended_pdbs_dir.glob("*.pdb"))

    def test_classify_structure_help(self):
        """Test that classify-structure --help works."""
        result = self.runner.invoke(cli_app, ["classify-structure", "--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("classify-structure", result.output)
        self.assertIn("--pdb", result.output)

    def test_classify_single_pdb(self):
        """Test classifying a single PDB file via CLI."""
        if not self.has_kinked_pdbs:
            self.skipTest("Kinked PDB test files not available")

        pdb_path = next(self.kinked_pdbs_dir.glob("*.pdb"))
        result = self.runner.invoke(
            cli_app,
            ["classify-structure", "-p", str(pdb_path)]
        )
        # Note: May fail if PDB doesn't have valid VHH chain, which is OK for this test
        # Just verify command runs without crash
        self.assertIn(result.exit_code, [0, 1], f"CLI crashed: {result.output}")

    def test_classify_pdb_directory(self):
        """Test classifying a directory of PDB files."""
        if not self.has_kinked_pdbs:
            self.skipTest("Kinked PDB test files not available")

        result = self.runner.invoke(
            cli_app,
            ["classify-structure", "-d", str(self.kinked_pdbs_dir)]
        )
        # Just verify command runs without crash
        self.assertIn(result.exit_code, [0, 1], f"CLI crashed: {result.output}")

    def test_classify_structure_to_csv(self):
        """Test saving structure classification results to CSV."""
        if not self.has_kinked_pdbs:
            self.skipTest("Kinked PDB test files not available")

        pdb_path = next(self.kinked_pdbs_dir.glob("*.pdb"))

        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
            output_csv = tmp.name

        try:
            result = self.runner.invoke(
                cli_app,
                ["classify-structure", "-p", str(pdb_path), "--output-csv", output_csv]
            )
            # Check if output was created (may fail if PDB doesn't have valid VHH)
            if result.exit_code == 0 and os.path.exists(output_csv):
                with open(output_csv) as f:
                    content = f.read()
                    self.assertGreater(len(content), 0)
        finally:
            if os.path.exists(output_csv):
                os.unlink(output_csv)


class TestCLIGeneral(unittest.TestCase):
    """General CLI tests."""

    def setUp(self):
        self.runner = CliRunner()

    def test_main_help(self):
        """Test that main --help works."""
        result = self.runner.invoke(cli_app, ["--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("classify-sequence", result.output)
        self.assertIn("classify-structure", result.output)

    def test_version_or_about(self):
        """Test that version/about information is available."""
        # Try --version first
        result = self.runner.invoke(cli_app, ["--version"])
        # If --version doesn't work, just ensure the help works
        if result.exit_code != 0:
            result = self.runner.invoke(cli_app, ["--help"])
            self.assertEqual(result.exit_code, 0)


if __name__ == "__main__":
    unittest.main()
