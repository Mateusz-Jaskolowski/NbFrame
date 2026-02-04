"""
Pytest configuration and shared fixtures for NbFrame tests.

This module provides fixtures for test data paths, temporary directories,
and common test utilities.
"""
import pytest
import tempfile
import shutil
from pathlib import Path


# =============================================================================
# Path Fixtures
# =============================================================================

@pytest.fixture
def test_data_dir():
    """Path to the test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture
def sequences_dir(test_data_dir):
    """Path to the test sequences directory."""
    return test_data_dir / "sequences"


@pytest.fixture
def pdbs_dir(test_data_dir):
    """Path to the test PDBs directory."""
    return test_data_dir / "pdbs"


@pytest.fixture
def kinked_pdbs_dir(pdbs_dir):
    """Path to kinked PDB files."""
    return pdbs_dir / "kinked"


@pytest.fixture
def extended_pdbs_dir(pdbs_dir):
    """Path to extended PDB files."""
    return pdbs_dir / "extended"


@pytest.fixture
def edge_cases_dir(pdbs_dir):
    """Path to edge case PDB files."""
    return pdbs_dir / "edge_cases"


@pytest.fixture
def mmcif_dir(pdbs_dir):
    """Path to mmCIF test files."""
    return pdbs_dir / "mmcif"


# =============================================================================
# Test Sequences
# =============================================================================

@pytest.fixture
def kinked_sequences():
    """Known kinked VHH sequences from TestSequences.md."""
    return {
        "1f2x_L": (
            "QVQLVESGGGSVQAGGSLRLSCAASGYTVSTYCMGWFRQAPGKEREGVATILGGSTYYGD"
            "SVKGRFTISQDNAKNTVYLQMNSLKPEDTAIYYCAGSTVASTGWCSRLRPYDYHYRGQGTQVTVSS"
        ),
        "1mvf_B": (
            "QVQLVESGGGSVQAGGSLRLSCAASGFTYSRKYMGWFRQAPGKEREGVAAIFIDNGNTIY"
            "ADSVQGRFTISQDNAKNTVYLQMNSLKPEDTAMYYCAASSRWMDYSALTAKAYNSWGQGTQVTVSS"
        ),
        "8im0_B": (
            "QVQLVESGGGLVQAGGSLRLSCAVSGRPFSEYNLGWFRQAPGKEREFVARIRSSGTTVYT"
            "DSVKGRFSASRDNAKNMGYLQLNSLEPEDTAVYYCAMSRVDTDSPAFYDYWGQGTQVTVST"
        ),
    }


@pytest.fixture
def extended_sequences():
    """Known extended VHH sequences from TestSequences.md."""
    return {
        "4cdg_D": (
            "QVQLQESGGGLVQAGGSLRLSCAASGIWFSINNMAWYRQTPGKQRERIAIITSAGTTNYV"
            "DSVKGRFTISRDDAKNTMYLQMNSLIPEDTAVYYCNLVADYDMGFQSFWGRGTQVTVSS"
        ),
        "5m2w_B": (
            "QVQLVESGGGLVQAGGTLKLSCAASGSISGIVVMAWYRQAPGKQRELVASITSGGTTNYADSVKGR"
            "FTISKDNAENTLYLRMNSLKPEDTAVYYCKAFFRRDYVGYDYWGQGTQVTVSS"
        ),
        "5h8d_A": (
            "QVQLQESGGGLVQPGGSLKLSCAASGFTFSRYAMSWYRQAPGKERESVARISSGGGTIYYADSVKGR"
            "FTISREDAKNTVYLQMNSLKPEDTAVYYCYVGGFWGQGTQVTVSS"
        ),
    }


@pytest.fixture
def all_test_sequences(kinked_sequences, extended_sequences):
    """All test sequences combined."""
    return {**kinked_sequences, **extended_sequences}


# =============================================================================
# Temporary Directory Fixtures
# =============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory that is cleaned up after the test."""
    dirpath = tempfile.mkdtemp()
    yield Path(dirpath)
    shutil.rmtree(dirpath, ignore_errors=True)


@pytest.fixture
def temp_fasta(temp_dir, all_test_sequences):
    """Create a temporary FASTA file with test sequences."""
    fasta_path = temp_dir / "test_sequences.fasta"
    with open(fasta_path, 'w') as f:
        for name, seq in all_test_sequences.items():
            f.write(f">{name}\n{seq}\n")
    return fasta_path


@pytest.fixture
def temp_csv(temp_dir):
    """Create a temporary CSV file path."""
    return temp_dir / "output.csv"


# =============================================================================
# Skip Conditions
# =============================================================================

@pytest.fixture
def skip_if_no_kinked_pdbs(kinked_pdbs_dir):
    """Skip test if kinked PDB files are not available."""
    if not kinked_pdbs_dir.exists() or not any(kinked_pdbs_dir.glob("*.pdb")):
        pytest.skip("Kinked PDB test files not available")


@pytest.fixture
def skip_if_no_extended_pdbs(extended_pdbs_dir):
    """Skip test if extended PDB files are not available."""
    if not extended_pdbs_dir.exists() or not any(extended_pdbs_dir.glob("*.pdb")):
        pytest.skip("Extended PDB test files not available")


@pytest.fixture
def skip_if_no_mmcif(mmcif_dir):
    """Skip test if mmCIF files are not available."""
    if not mmcif_dir.exists() or not any(mmcif_dir.glob("*.cif")):
        pytest.skip("mmCIF test files not available")


# =============================================================================
# Sample PDB Paths
# =============================================================================

@pytest.fixture
def sample_kinked_pdb(kinked_pdbs_dir):
    """Path to a sample kinked PDB file."""
    pdbs = list(kinked_pdbs_dir.glob("*.pdb"))
    if not pdbs:
        pytest.skip("No kinked PDB files available")
    return pdbs[0]


@pytest.fixture
def sample_extended_pdb(extended_pdbs_dir):
    """Path to a sample extended PDB file."""
    pdbs = list(extended_pdbs_dir.glob("*.pdb"))
    if not pdbs:
        pytest.skip("No extended PDB files available")
    return pdbs[0]


@pytest.fixture
def sample_mmcif(mmcif_dir):
    """Path to a sample mmCIF file."""
    cifs = list(mmcif_dir.glob("*.cif"))
    if not cifs:
        pytest.skip("No mmCIF files available")
    return cifs[0]


@pytest.fixture
def no_vhh_pdb(edge_cases_dir):
    """Path to a PDB with no VHH chains."""
    pdb_path = edge_cases_dir / "missing_VHH_chains_5GRY.pdb"
    if not pdb_path.exists():
        pytest.skip("No VHH chains PDB not available")
    return pdb_path
