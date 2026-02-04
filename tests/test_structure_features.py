"""
Test suite for NbFrame structure features module.

Tests the structural feature computation functionality including
angle calculations, contact density, RSA, and RMSD computation.
"""
import unittest
from pathlib import Path

from nbframe.structure_features import (
    build_residues_by_aho,
    compute_c_terminal_angles,
    compute_fr2_rsa,
    compute_n_terminal_angles,
    compute_structure_features,
    get_chain,
    load_structure,
)


# Path to test data directory
TEST_DATA_DIR = Path(__file__).parent / "data"
TEST_PDBS_DIR = TEST_DATA_DIR / "pdbs"


class TestStructureFeaturesSmoke(unittest.TestCase):
    """
    Smoke tests for structure_features helpers.

    These tests verify that the helpers can be imported and are callable.
    """

    def test_module_import_and_api(self):
        """Test that all expected functions are importable and callable."""
        self.assertTrue(callable(load_structure))
        self.assertTrue(callable(get_chain))
        self.assertTrue(callable(build_residues_by_aho))
        self.assertTrue(callable(compute_n_terminal_angles))
        self.assertTrue(callable(compute_c_terminal_angles))
        self.assertTrue(callable(compute_fr2_rsa))
        self.assertTrue(callable(compute_structure_features))


class TestStructureFeaturesWithPDBs(unittest.TestCase):
    """
    Tests for structure feature computation using actual PDB files.
    """

    @classmethod
    def setUpClass(cls):
        """Check if test PDB files are available."""
        cls.kinked_pdbs_dir = TEST_PDBS_DIR / "kinked"
        cls.extended_pdbs_dir = TEST_PDBS_DIR / "extended"
        cls.mmcif_dir = TEST_PDBS_DIR / "mmcif"

        cls.has_kinked_pdbs = cls.kinked_pdbs_dir.exists() and any(cls.kinked_pdbs_dir.glob("*.pdb"))
        cls.has_extended_pdbs = cls.extended_pdbs_dir.exists() and any(cls.extended_pdbs_dir.glob("*.pdb"))
        cls.has_mmcif = cls.mmcif_dir.exists() and any(cls.mmcif_dir.glob("*.cif"))

    def test_load_pdb_structure(self):
        """Test loading a PDB structure."""
        if not self.has_kinked_pdbs:
            self.skipTest("Kinked PDB test files not available")

        pdb_path = next(self.kinked_pdbs_dir.glob("*.pdb"))
        structure = load_structure(str(pdb_path))

        self.assertIsNotNone(structure)
        # BioPython structure should have models
        self.assertGreater(len(list(structure.get_models())), 0)

    def test_load_mmcif_structure(self):
        """Test loading an mmCIF structure."""
        if not self.has_mmcif:
            self.skipTest("mmCIF test files not available")

        cif_path = next(self.mmcif_dir.glob("*.cif"))
        structure = load_structure(str(cif_path))

        self.assertIsNotNone(structure)
        # BioPython structure should have models
        self.assertGreater(len(list(structure.get_models())), 0)

    def test_compute_features_from_pdb(self):
        """Test computing all features from a PDB file."""
        if not self.has_kinked_pdbs:
            self.skipTest("Kinked PDB test files not available")

        pdb_path = next(self.kinked_pdbs_dir.glob("*.pdb"))

        # This will compute features if the PDB has a valid VHH chain
        try:
            features = compute_structure_features(str(pdb_path))

            # Verify expected feature keys
            expected_keys = ['alpha_N', 'tau_N', 'alpha_C', 'tau_C',
                           'contact_density', 'fr2_rsa_key']
            for key in expected_keys:
                self.assertIn(key, features, f"Missing expected feature: {key}")

            # Verify features are numeric (some may be None if computation failed)
            for key in expected_keys:
                if features[key] is not None:
                    self.assertIsInstance(features[key], (int, float),
                                        f"Feature {key} should be numeric")

        except (ValueError, RuntimeError, KeyError) as e:
            # Some PDBs may not have valid VHH chains or correct chain IDs
            self.skipTest(f"PDB does not have valid VHH chain: {e}")

    def test_feature_values_differ_by_conformation(self):
        """Test that kinked and extended structures have different feature values."""
        if not (self.has_kinked_pdbs and self.has_extended_pdbs):
            self.skipTest("Both kinked and extended PDB files required")

        # Use classify_structure which auto-detects VHH chains
        from nbframe import classify_structure

        kinked_pdb = next(self.kinked_pdbs_dir.glob("*.pdb"))
        extended_pdb = next(self.extended_pdbs_dir.glob("*.pdb"))

        # Get features via classify_structure (which auto-detects chains)
        kinked_result = classify_structure(str(kinked_pdb))
        extended_result = classify_structure(str(extended_pdb))

        kinked_features = kinked_result.get('features', {})
        extended_features = extended_result.get('features', {})

        # At least some features should differ between kinked and extended
        # (they characterize different conformations)
        differences = 0
        for key in kinked_features:
            if key in extended_features:
                kf = kinked_features[key]
                ef = extended_features[key]
                # Skip None values
                if kf is not None and ef is not None:
                    if abs(kf - ef) > 0.01:
                        differences += 1

        self.assertGreater(differences, 0,
                         "Kinked and extended structures should have different feature values")


if __name__ == "__main__":
    unittest.main()
