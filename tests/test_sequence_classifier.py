"""
Test suite for NbFrame sequence classifier.

Tests the sequence-based CDR3 conformation prediction functionality including
single sequence prediction, batch processing, FASTA file handling, and
blueprint scoring.
"""
import unittest
import os
import tempfile
import pandas as pd
import numpy as np
from pathlib import Path

# Import nbframe functions to test
from nbframe import (
    predict_kink_probability,
    predict_kink_probabilities,
    predict_from_fasta,
    read_fasta,
    results_to_dataframe,
    save_results_to_csv,
    batch_align_sequences,
)


class TestSequenceClassifier(unittest.TestCase):
    """Test suite for NbFrame sequence classifier"""

    # Known sequences with structural labels (PDB IDs -> sequence).
    # Extended (non-kinking) VHH structures:
    EXTENDED_PDB_SEQUENCES = {
        "4cdg_D": (
            "QVQLQESGGGLVQAGGSLRLSCAASGIWFSINNMAWYRQTPGKQRERIAIITSAGTTNYVDSVKGRFTISRDDAKNTMYLQMNSLIPEDTAVYYCNLVADYDMGFQSFWGRGTQVTVSS"
        ),
        "5m2w_B": (
            "QVQLVESGGGLVQAGGTLKLSCAASGSISGIVVMAWYRQAPGKQRELVASITSGGTTNYADSVKGRFTISKDNAENTLYLRMNSLKPEDTAVYYCKAFFRRDYVGYDYWGQGTQVTVSS"
        ),
        "5h8d_A": (
            "QVQLQESGGGLVQPGGSLKLSCAASGFTFSRYAMSWYRQAPGKERESVARISSGGGTIYYADSVKGRFTISREDAKNTVYLQMNSLKPEDTAVYYCYVGGFWGQGTQVTVSS"
        ),
    }

    # Kinked VHH structures:
    KINKED_PDB_SEQUENCES = {
        "1f2x_L": (
            "QVQLVESGGGSVQAGGSLRLSCAASGYTVSTYCMGWFRQAPGKEREGVATILGGSTYYGDSVKGRFTISQDNAKNTVYLQMNSLKPEDTAIYYCAGSTVASTGWCSRLRPYDYHYRGQGTQVTVSS"
        ),
        "1mvf_B": (
            "QVQLVESGGGSVQAGGSLRLSCAASGFTYSRKYMGWFRQAPGKEREGVAAIFIDNGNTIYADSVQGRFTISQDNAKNTVYLQMNSLKPEDTAMYYCAASSRWMDYSALTAKAYNSWGQGTQVTVSS"
        ),
        "8im0_B": (
            "QVQLVESGGGLVQAGGSLRLSCAVSGRPFSEYNLGWFRQAPGKEREFVARIRSSGTTVYTDSVKGRFSASRDNAKNMGYLQLNSLEPEDTAVYYCAMSRVDTDSPAFYDYWGQGTQVTVST"
        ),
    }

    # We expose simple lists of sequences and a combined test set. The ordering is:
    #   - all kinked sequences first
    #   - all extended (non-kinking) sequences afterwards
    KINKING_SEQUENCES = list(KINKED_PDB_SEQUENCES.values())
    NON_KINKING_SEQUENCES = list(EXTENDED_PDB_SEQUENCES.values())

    TEST_SEQUENCE_NAMES = list(KINKED_PDB_SEQUENCES.keys()) + list(EXTENDED_PDB_SEQUENCES.keys())
    TEST_SEQUENCES = KINKING_SEQUENCES + NON_KINKING_SEQUENCES

    def setUp(self):
        """Set up test case - create temporary files if needed"""
        # Create a temporary FASTA file with test sequences
        self.temp_fasta = tempfile.NamedTemporaryFile(suffix='.fasta', delete=False)
        with open(self.temp_fasta.name, 'w') as f:
            # Use PDB-based names so test data can be traced back to structures
            for name, seq in zip(self.TEST_SEQUENCE_NAMES, self.TEST_SEQUENCES):
                f.write(f">{name}\n{seq}\n")

        # Create a temporary output file for CSV test
        self.temp_csv = tempfile.NamedTemporaryFile(suffix='.csv', delete=False)
        self.temp_csv.close()

    def tearDown(self):
        """Clean up after test case"""
        # Remove temporary files
        try:
            os.unlink(self.temp_fasta.name)
            os.unlink(self.temp_csv.name)
        except:
            pass

    def test_read_fasta(self):
        """Test reading sequences from FASTA file"""
        sequences = read_fasta(self.temp_fasta.name)

        # Verify correct number of sequences read
        self.assertEqual(len(sequences), len(self.TEST_SEQUENCES))

        # Verify sequence contents match
        for i, (name, seq) in enumerate(sequences):
            self.assertEqual(name, self.TEST_SEQUENCE_NAMES[i])
            self.assertEqual(seq, self.TEST_SEQUENCES[i])

    def test_single_prediction(self):
        """Test prediction on a single sequence"""
        # Test with a known kinking sequence
        kink_result = predict_kink_probability(self.KINKING_SEQUENCES[0])

        # Verify result structure
        self.assertIsInstance(kink_result, dict)
        self.assertIn('input_sequence', kink_result)
        self.assertIn('aligned_sequence', kink_result)
        self.assertIn('raw_score', kink_result)
        self.assertIn('probability', kink_result)

        # Test with a known non-kinking sequence
        non_kink_result = predict_kink_probability(self.NON_KINKING_SEQUENCES[0])

        # Both predictions should produce numeric probabilities
        self.assertIsInstance(kink_result['probability'], (float, np.floating))
        self.assertIsInstance(non_kink_result['probability'], (float, np.floating))

        # Kinking sequence should score higher than non-kinking sequence,
        # but we do not assume a fixed 0.5 threshold for the calibrated model.
        self.assertGreater(kink_result['probability'], non_kink_result['probability'])

    def test_batch_prediction(self):
        """Test batch prediction with batch API"""
        results = predict_kink_probabilities(
            self.TEST_SEQUENCES,
            verbose=False,
            batch_size=2
        )

        # Verify correct number of results returned
        self.assertEqual(len(results), len(self.TEST_SEQUENCES))

        # Verify all results have probability scores
        for result in results:
            self.assertIsNotNone(result['probability'])
            self.assertIsInstance(result['probability'], float)

        # Verify kinking sequences have higher probabilities than non-kinking
        kinking_scores = [results[i]['probability'] for i in range(len(self.KINKING_SEQUENCES))]
        non_kinking_scores = [results[i+len(self.KINKING_SEQUENCES)]['probability']
                              for i in range(len(self.NON_KINKING_SEQUENCES))]

        self.assertGreater(min(kinking_scores), max(non_kinking_scores))

    def test_batch_alignment(self):
        """Test batch alignment function"""
        aligned_sequences = batch_align_sequences(
            self.TEST_SEQUENCES,
            fix_cdr1_gaps=True,
            verbose=False
        )

        # Verify all sequences were aligned
        self.assertEqual(len(aligned_sequences), len(self.TEST_SEQUENCES))

        # Verify alignments are valid
        for aligned_seq in aligned_sequences:
            self.assertIsNotNone(aligned_seq)
            # Check that aligned sequences have gaps (expected in AHo alignment)
            self.assertIn('-', aligned_seq)
            # Check alignment length consistency: all aligned VHH should be
            # similar length, but the absolute length can vary with ANARCI.
            # We therefore only require that lengths are within a reasonable range.

        lengths = [len(seq) for seq in aligned_sequences]
        self.assertGreater(min(lengths), 0)
        # Ensure the aligned lengths do not vary wildly across this small test set
        self.assertLessEqual(max(lengths) - min(lengths), 20)

    def test_results_to_dataframe(self):
        """Test conversion of results to dataframe"""
        results = predict_kink_probabilities(
            self.TEST_SEQUENCES,
            verbose=False,
            batch_size=2
        )

        df = results_to_dataframe(results)

        # Verify DataFrame structure
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), len(self.TEST_SEQUENCES))
        self.assertIn('nbframe_score', df.columns)
        self.assertIn('raw_score', df.columns)
        self.assertIn('aligned_sequence', df.columns)

    def test_csv_output(self):
        """Test saving results to CSV"""
        results = predict_kink_probabilities(
            self.TEST_SEQUENCES,
            verbose=False,
            batch_size=2
        )

        # Test saving to CSV
        save_results_to_csv(results, self.temp_csv.name)

        # Verify file exists and can be read as CSV
        self.assertTrue(os.path.exists(self.temp_csv.name))
        df = pd.read_csv(self.temp_csv.name)
        self.assertEqual(len(df), len(self.TEST_SEQUENCES))

    def test_predict_from_fasta(self):
        """Test prediction from FASTA file"""
        df = predict_from_fasta(
            self.temp_fasta.name,
            verbose=False,
            batch_size=2
        )

        # The optimized FASTA prediction returns a DataFrame
        self.assertIsInstance(df, pd.DataFrame)

        # Verify correct number of results (one row per input sequence)
        self.assertEqual(len(df), len(self.TEST_SEQUENCES))

        # Verify expected columns are present
        expected_columns = {'name', 'sequence', 'nbframe_score', 'raw_score', 'aligned_sequence'}
        self.assertTrue(expected_columns.issubset(set(df.columns)))

    def test_error_handling(self):
        """Test error handling for invalid inputs"""
        # Test with an invalid sequence (not amino acids)
        result = predict_kink_probability("INVALID123SEQUENCE")
        self.assertIsNotNone(result['error'])

        # Test with an empty sequence
        result = predict_kink_probability("")
        self.assertIsNotNone(result['error'])

        # Test with non-existent FASTA file
        with self.assertRaises(FileNotFoundError):
            read_fasta("nonexistent_file.fasta")

    def test_large_batch_simulation(self):
        """Simulate working with a larger batch by repeating test sequences"""
        # Create a larger test set (100 sequences) by repeating test sequences
        large_test_set = self.TEST_SEQUENCES * 25

        # Test batch processing with larger set
        results = predict_kink_probabilities(
            large_test_set,
            verbose=False,
            batch_size=50  # Test a larger batch size
        )

        # Verify all results processed
        self.assertEqual(len(results), len(large_test_set))

        # Verify no errors in batch processing
        errors = [r for r in results if r['error'] is not None]
        self.assertEqual(len(errors), 0)


if __name__ == '__main__':
    unittest.main()
