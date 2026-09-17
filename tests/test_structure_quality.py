"""Coordinate perturbations must not masquerade as conformational changes."""
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from Bio.PDB import PDBIO
from typer.testing import CliRunner

import nbframe
from nbframe import structure_features as sf, structure_numbering as sn, sequence_align as sa
from nbframe.structure_classifier import predict_structure_from_features, load_structure_classifier
from nbframe.structure_quality import assess_structure_quality
from nbframe.cli import app

PDB = Path(__file__).parent / 'data/pdbs/kinked/9bsv.pdb'


@pytest.fixture(scope='module')
def aho(tmp_path_factory):
    info = sn.parse_pdb_chains(str(PDB))['D']
    mapping = sn._build_pdb_to_aho_mapping(info, sa.number_chain_to_aho(info.sequence))
    return sn.renumber_structure_to_aho(sf.load_structure(str(PDB)), 'D', mapping,
        temp_dir=tmp_path_factory.mktemp('quality_aho'))[0]


def save(structure, path):
    io = PDBIO(); io.set_structure(structure); io.save(str(path))
    return str(path)


def measure(structure, tmp_path):
    return sf.compute_structure_features(save(structure, tmp_path/'aho.pdb'), chain_id='D')


def test_complete_structure_and_legal_numbering_gaps(aho, tmp_path):
    features = measure(aho, tmp_path)
    assert features['quality']['status'] == 'passed'
    assert features['quality']['sequence_coverage'] == 'not_assessed'
    chain = aho[0]['D']
    numbers = [r.id[1] for r in chain if 108 <= r.id[1] <= 138]
    assert len(numbers) < 31  # Expected AHo gaps in a complete loop.
    result = predict_structure_from_features(features)
    assert result['status'] == 'classified' and result['label'] == 'kinked'
    assert result['error'] is None
    assert result['quality']['regions']['cdr3']['atom_completeness'] == 1


@pytest.mark.parametrize('region,position,atom', [('fr2', 44, 'CA'), ('fr2', 51, 'CB'), ('cdr3', 110, 'O')])
def test_missing_atoms_withhold_prediction(aho, tmp_path, region, position, atom):
    damaged = aho.copy(); chain = damaged[0]['D']
    assert atom in chain[position]
    chain[position].detach_child(atom)
    features = measure(damaged, tmp_path)
    assert features['quality']['regions'][region]['atom_completeness'] < 1
    assert features['contact_nres'] is None
    clf, _ = load_structure_classifier()
    with patch.object(clf, 'predict_proba', side_effect=AssertionError('Must not score missing data')):
        result = predict_structure_from_features(features)
    assert result['status'] == 'insufficient_quality'
    assert result['label'] is None and result['prob_kinked'] is None
    assert result['error'] and result['warnings']
    json.dumps(result, allow_nan=False)


def test_removed_sidechains_are_not_zero_contact_evidence(aho, tmp_path):
    damaged = aho.copy()
    for residue in damaged[0]['D']:
        if 108 <= residue.id[1] <= 138:
            for atom in list(residue):
                if atom.name not in {'N', 'CA', 'C', 'O'}:
                    residue.detach_child(atom.id)
    result = predict_structure_from_features(measure(damaged, tmp_path))
    assert result['status'] == 'insufficient_quality'
    assert result['features']['contact_nres'] is None
    assert result['features']['alpha_N'] is not None
    assert result['probabilities'] == {'kinked': None, 'extended': None}


def test_internal_deleted_residue_detected_away_from_angles(aho, tmp_path):
    damaged = aho.copy(); chain = damaged[0]['D']
    chain.detach_child(chain[113].id)
    features = measure(damaged, tmp_path)
    assert features['alpha_N'] is not None and features['alpha_C'] is not None
    assert features['quality']['regions']['cdr3']['backbone_breaks']
    assert features['contact_nres'] is None


def test_missing_fr2_base_is_not_replaced_by_insertion(aho, tmp_path):
    damaged = aho.copy(); chain = damaged[0]['D']
    chain[50].id = (' ', 50, 'A')
    features = measure(damaged, tmp_path)
    assert features['quality']['regions']['fr2']['missing_required_positions'] == [50]
    assert features['quality']['status'] == 'insufficient_quality'


def test_zero_occupancy_is_unusable(aho, tmp_path):
    damaged = aho.copy(); damaged[0]['D'][44]['CB'].set_occupancy(0)
    features = measure(damaged, tmp_path)
    assert features['fr2_rsa_key'] is None
    assert 'CB' in next(i for i in features['quality']['issues'] if i.get('residue') == '44')['atoms']


def test_nonfinite_coordinates_do_not_reach_sasa(aho):
    damaged = aho.copy(); damaged[0]['D'][44]['CB'].coord[:] = np.nan
    report = assess_structure_quality(damaged[0]['D'], {'alpha_N': 0, 'tau_N': 0, 'alpha_C': 0, 'tau_C': 0})
    assert report['status'] == 'insufficient_quality'
    assert 'fr2_rsa_key' in report['invalid_features']


def test_feature_only_inputs_do_not_claim_coordinate_validation():
    _, meta = load_structure_classifier()
    result = predict_structure_from_features({name: 0 for name in meta.feature_cols})
    assert result['quality']['status'] == 'not_assessed'


@pytest.mark.parametrize('value', [float('inf'), float('-inf'), float('nan')])
def test_nonfinite_model_features_rejected(value):
    _, meta = load_structure_classifier()
    features = {name: 0 for name in meta.feature_cols}; features[meta.feature_cols[0]] = value
    with pytest.raises(ValueError, match='missing or invalid'):
        predict_structure_from_features(features)


@pytest.fixture
def mixed_raw(tmp_path):
    structure = sf.load_structure(str(PDB)); model = structure[0]
    for chain in list(model):
        if chain.id != 'D': model.detach_child(chain.id)
    damaged = model['D'].copy(); damaged.id = 'Q'
    # Remove all sidechains from the second copy; preserve the sequence/backbone.
    for residue in damaged:
        for atom in list(residue):
            if atom.name not in {'N','CA','C','O'}: residue.detach_child(atom.id)
    model.add(damaged)
    return save(structure, tmp_path/'mixed.pdb')


def test_mixed_quality_chains_retain_good_result_and_failure(mixed_raw):
    results = nbframe.classify_all_nanobodies_in_pdb(mixed_raw, chain_ids=['Q','D'], filter_by_rmsd=False)
    assert list(results) == ['Q','D']
    assert results['D']['status'] == 'classified'
    assert results['Q']['status'] == 'insufficient_quality'
    assert results['Q']['chain_id_used'] == 'Q' and results['Q']['model_id_used'] == 0
    auto = nbframe.classify_structure(str(PDB), filter_by_rmsd=False)
    assert auto['chain_id_used'] in {'D','E','F'}


def test_cli_retains_withheld_result_in_summary_exports(mixed_raw, tmp_path):
    output = tmp_path/'results.json'; csv = tmp_path/'results.csv'
    result = CliRunner().invoke(app, ['classify-structure', '--pdb', mixed_raw, '--chain','Q',
        '--output-json',str(output), '--output-csv',str(csv), '--summary-only'])
    assert result.exit_code == 1, result.output
    assert 'prediction withheld' in result.output
    payload = json.loads(output.read_text())
    assert payload['status'] == 'insufficient_quality' and payload['prob_kinked'] is None
    assert 'features' not in payload and payload['quality']['issues']
    row = pd.read_csv(csv).iloc[0]
    assert row.status == 'insufficient_quality' and pd.isna(row.prob_kinked)
    assert json.loads(row.quality)['status'] == 'insufficient_quality'


def test_table_quality_round_trip_for_incomplete_structure(tmp_path):
    source = Path(__file__).parent/'data/pdbs/kinked/1mvf.pdb'
    directory = tmp_path/'input'; directory.mkdir()
    (directory/'sample.pdb').write_bytes(source.read_bytes())
    csv = tmp_path/'features.csv'
    nbframe.compute_features_for_pdb_directory(directory, output_csv=csv)
    result = predict_structure_from_features(pd.read_csv(csv).iloc[0].to_dict())
    assert result['status'] == 'insufficient_quality'
    assert result['label'] is None
    # Stripping metadata still cannot turn invalidated measurements into scores.
    features = pd.read_csv(csv).iloc[0].to_dict(); features.pop('quality')
    with pytest.raises(ValueError, match='missing or invalid'):
        predict_structure_from_features(features)


def test_complete_prediction_matches_saved_baseline():
    result = nbframe.classify_structure(str(PDB), chain_id='D', filter_by_rmsd=False)
    # Geometry starts from float32 coordinates; allow platform-level rounding.
    assert result['prob_kinked'] == pytest.approx(0.9934712575233963, abs=1e-8, rel=0)


def test_zero_occupancy_angle_anchor_is_unmeasurable(aho, tmp_path):
    damaged = aho.copy(); damaged[0]['D'][107]['CA'].set_occupancy(0)
    features = measure(damaged, tmp_path)
    assert features['alpha_N'] is None and features['tau_N'] is None
    assert not features['quality']['regions']['angle_N']['measurable']
    assert '107' in features['quality']['regions']['angle_N']['anchor_residues']


def test_missing_key_residue_and_entire_loop_are_reported(aho, tmp_path):
    damaged = aho.copy(); chain = damaged[0]['D']
    chain.detach_child(chain[44].id)
    for residue in list(chain):
        if 108 <= residue.id[1] <= 138: chain.detach_child(residue.id)
    features = measure(damaged, tmp_path)
    assert features['quality']['regions']['rsa_key']['missing_required_positions'] == [44]
    assert features['quality']['regions']['cdr3']['resolved_residues'] == 0
    assert features['contact_nres'] is None and features['fr2_rsa_key'] is None


def test_insertion_residue_is_checked(aho, tmp_path):
    damaged = aho.copy(); chain = damaged[0]['D']
    residue = chain[113]
    residue.id = (' ', 112, 'A')
    residue.detach_child('O')
    features = measure(damaged, tmp_path)
    assert '112A' in features['quality']['regions']['cdr3']['incomplete_residues']


def test_optional_terminal_oxygen_and_hydrogens_not_required(aho, tmp_path):
    features = measure(aho, tmp_path)
    assert all(not atom.name.startswith('H') for atom in aho.get_atoms())
    assert features['quality']['status'] == 'passed'


def test_cli_mixed_results_and_directory_withheld_export(mixed_raw, tmp_path):
    output = tmp_path/'mixed.json'
    result = CliRunner().invoke(app, ['classify-structure', '--pdb',mixed_raw,
        '--chain','Q,D', '--output-json',str(output)])
    assert result.exit_code == 0, result.output
    payload = json.loads(output.read_text())
    assert payload['D']['status'] == 'classified' and payload['Q']['status'] == 'insufficient_quality'
    structure = sf.load_structure(mixed_raw); structure[0].detach_child('D')
    directory=tmp_path/'bad_only'; directory.mkdir()
    save(structure, directory/'bad.pdb')
    output=tmp_path/'directory.json'; csv=tmp_path/'directory.csv'
    result=CliRunner().invoke(app, ['classify-structure','--pdb-dir',str(directory),
        '--output-json',str(output),'--output-csv',str(csv),'--summary-only'])
    assert result.exit_code == 1, result.output
    assert 'successes=0' in result.output
    payload=json.loads(output.read_text())
    assert next(iter(payload.values()))['Q']['status'] == 'insufficient_quality'
    assert pd.read_csv(csv).iloc[0].status == 'insufficient_quality'


def test_csv_withheld_result_remains_strict_json_serializable(tmp_path):
    source = Path(__file__).parent/'data/pdbs/kinked/1mvf.pdb'
    directory=tmp_path/'input'; directory.mkdir()
    (directory/'sample.pdb').write_bytes(source.read_bytes())
    csv=tmp_path/'features.csv'
    nbframe.compute_features_for_pdb_directory(directory, output_csv=csv)
    result=predict_structure_from_features(pd.read_csv(csv).iloc[0].to_dict())
    json.dumps(result, allow_nan=False)


def test_quality_check_does_not_mutate_coordinates(aho):
    before=[atom.coord.copy() for atom in aho.get_atoms()]
    assess_structure_quality(aho[0]['D'],dict(alpha_N=0, tau_N=0, alpha_C=0, tau_C=0))
    for atom, coordinate in zip(aho.get_atoms(),before):
        np.testing.assert_array_equal(atom.coord, coordinate)
