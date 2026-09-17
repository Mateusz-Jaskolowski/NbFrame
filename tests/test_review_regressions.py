"""Regression cases for review findings F01–F13 (September 2026)."""
from pathlib import Path
import shutil
from unittest.mock import patch

import anarci
import numpy as np
import pandas as pd
import pytest
from Bio.PDB import MMCIFIO, PDBIO
from Bio.PDB.Atom import Atom
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import Residue
from click import unstyle
from typer.testing import CliRunner

import nbframe
from nbframe import sequence_align as sa, sequence_predictor as sp
from nbframe import structure_features as sf, structure_numbering as sn
from nbframe.structure_classifier import load_structure_classifier, predict_structure_from_features
from nbframe.cli import app

DATA = Path(__file__).parent / 'data'
PDB = DATA / 'pdbs/kinked/1mvf.pdb'


@pytest.fixture(scope='module')
def sequence():
    return nbframe.read_fasta(DATA / 'sequences/single_kinked.fasta')[0][1]


@pytest.fixture(scope='module')
def aligned(sequence):
    return sa.get_aho_aligned_vhh_string(sequence)


@pytest.fixture(scope='module')
def aho(tmp_path_factory):
    info = sn.parse_pdb_chains(str(PDB))['B']
    mapping = sn._build_pdb_to_aho_mapping(info, sa.number_chain_to_aho(info.sequence))
    return sn.renumber_structure_to_aho(sf.load_structure(str(PDB)), 'B', mapping,
                                       temp_dir=tmp_path_factory.mktemp('aho'))[0]


def test_f01_insertions_do_not_shift_hallmarks(sequence):
    variant = sequence[:29] + 'G' * 7 + sequence[29:]
    numbering, _ = anarci.number(variant, scheme='aho')
    assert ((36, 'A'), 'G') in numbering
    canonical = ''.join(dict(numbering).get((i, ' '), '-') for i in range(1, 150))
    expected = sp.predict_kink_probability(canonical, do_alignment=False)['probability']
    single = nbframe.classify_sequence(variant)
    batch = nbframe.classify_sequences([variant], ncpu=1)[0]
    assert len(single['aligned_sequence']) == 149
    assert single['error'] is None and batch['error'] is None
    assert single['probability'] == pytest.approx(expected, abs=1e-8)
    assert batch['probability'] == pytest.approx(expected, abs=1e-8)
    assert single['label'] == batch['label'] == 'kinked'


def test_f01_cdr1_correction_cannot_move_framework_columns():
    numbering = [((i, ' '), '-' if 24 <= i <= 34 else 'A') for i in range(1,150)]
    result = sa._canonical_aho_string(numbering, fix_cdr1_gaps=True)
    assert result[:23] == 'A'*23
    assert result[42:] == 'A'*107


@pytest.mark.parametrize('value', ['', '123', '-'*149, 'X'*149, 'A'*148, 'A'*150,
                                    'A'*79 + '-'*70, None, 123])
def test_f02_invalid_prealigned_inputs_never_score(value):
    single = nbframe.classify_sequence(value, do_alignment=False)
    batch = nbframe.classify_sequences([value], do_alignment=False)[0]
    for result in (single, batch):
        assert result['error']
        assert result['probability'] is None and result['label'] is None


def test_f02_insufficient_hallmark_coverage(aligned):
    chars = list(aligned)
    for p in (12, 15, 17):
        chars[p - 1] = '-'
    result = nbframe.classify_sequence(''.join(chars), do_alignment=False)
    assert result['probability'] is None
    assert 'hallmark' in result['error']


def test_f02_fasta_and_dataframe_validation(tmp_path, aligned):
    fasta = tmp_path / 'aligned.fasta'
    fasta.write_text(f'>valid\n{aligned}\n>invalid\n123\n')
    result = sp.predict_from_fasta(fasta, do_alignment=False, batch_size=1)
    assert pd.notna(result.iloc[0].nbframe_score)
    assert pd.isna(result.iloc[1].nbframe_score) and result.iloc[1].error
    result = sp.predict_dataframe(pd.DataFrame({'seq': [aligned, '123']}), 'seq', do_alignment=False)
    assert pd.notna(result.iloc[0].nbframe_score)
    assert pd.isna(result.iloc[1].nbframe_score) and result.iloc[1].error


def test_f07_mixed_batch_preserves_valid_records(sequence):
    values = [sequence, 'QVQL123', None, '', sequence.lower()]
    results = sp.predict_kink_probabilities(values, batch_size=2, ncpu=1)
    assert [r['input_sequence'] for r in results] == values
    assert results[0]['probability'] == pytest.approx(results[-1]['probability'])
    assert all(r['error'] and r['probability'] is None for r in results[1:4])
    aligned = sa.batch_align_sequences(values, chunk_size=2, ncpu=1)
    assert aligned[0] == aligned[-1]
    assert aligned[1:4] == [None, None, None]


def test_f08_alignment_and_scoring_are_bounded(sequence):
    with patch.object(sa.anarci, 'run_anarci', wraps=sa.anarci.run_anarci) as align_spy, \
         patch.object(sp, '_predict_batch_with_lr_model', wraps=sp._predict_batch_with_lr_model) as score_spy:
        results = sp.predict_kink_probabilities([sequence]*5, batch_size=2, ncpu=1)
    assert all(r['error'] is None for r in results)
    assert [len(call.args[0]) for call in align_spy.call_args_list] == [2, 2, 1]
    assert [len(call.args[0]) for call in score_spy.call_args_list] == [2, 2, 1]
    assert all(call.kwargs['ncpu'] == 1 for call in align_spy.call_args_list)


def test_f08_direct_alignment_chunks_and_worker_cap(sequence):
    with patch.object(sa.anarci, 'run_anarci', wraps=sa.anarci.run_anarci) as spy:
        sa.batch_align_sequences([sequence]*3, chunk_size=1, ncpu=4)
    assert [len(c.args[0]) for c in spy.call_args_list] == [1, 1, 1]
    assert all(c.kwargs['ncpu'] == 1 for c in spy.call_args_list)


@pytest.mark.parametrize('size', [0, -1, False, 1.5])
def test_f08_invalid_batch_sizes(size):
    with pytest.raises(ValueError, match='positive integer'):
        sp.predict_kink_probabilities([], batch_size=size)
    with pytest.raises(ValueError, match='positive integer'):
        sa.batch_align_sequences([], chunk_size=size)


def test_f07_infrastructure_errors_are_not_record_validation_errors(sequence):
    with patch.object(sa.anarci, 'run_anarci', side_effect=FileNotFoundError('hmmscan')):
        with pytest.raises(FileNotFoundError, match='hmmscan'):
            sp.predict_kink_probabilities([sequence], ncpu=1)


def test_sequence_golden_scores_are_unchanged():
    table = pd.read_csv(Path(__file__).parents[1] / 'data/supplementary_table_all_929_structures.csv')
    results = sp.predict_kink_probabilities(table.AHo_aligned_sequence.tolist(),
                                           do_alignment=False, batch_size=37)
    assert all(r['error'] is None for r in results)
    np.testing.assert_allclose([r['probability'] for r in results], table['sequence_P(Kinked)'], atol=1e-12, rtol=0)


def test_f03_deleted_residue_is_not_replaced_by_later_neighbour(aho):
    chain = aho[0]['B'].copy()
    assert sf.compute_c_terminal_angles(chain)[0] is not None
    chain.detach_child((' ', 138, ' '))
    assert sf.compute_c_terminal_angles(chain) == (None, None)


@pytest.mark.parametrize('residue,atom', [(137, 'CA'), (138, 'N'), (137, 'C')])
def test_f03_missing_anchor_atoms(aho, residue, atom):
    chain = aho[0]['B'].copy()
    chain[residue].detach_child(atom)
    assert sf.compute_c_terminal_angles(chain) == (None, None)


def test_f03_numbering_gaps_are_allowed_for_connected_residues(aho):
    source = aho[0]['B']
    expected = sf.compute_n_terminal_angles(source)
    chain = Chain('H')
    for old, new in [(107,107), (108,108), (109,125), (110,126)]:
        residue = source[old].copy()
        residue.id = (' ',new,' ')
        chain.add(residue)
    assert sf.compute_n_terminal_angles(chain) == pytest.approx(expected)


def atom_residue(number, insertion=' ', distance=3.):
    residue = Residue((' ',number,insertion), 'ALA', ' ')
    residue.add(Atom('CA', np.array([distance,0.,0.]), 0., 1., ' ', ' CA ', number, element='C'))
    return residue


@pytest.mark.parametrize('code', ['A','H','K','L'])
def test_f04_decode_insertion_codes(code):
    assert sa._decode_aho_position((123, code)) == f'123{code}'
    assert sa._decode_aho_position(('H',123,code)) == f'123{code}'
    assert sa._decode_aho_position(('H',123)) == 123


def test_f04_insertions_contribute_independently_to_contacts_and_lengths():
    chain = Chain('H')
    chain.add(atom_residue(44, distance=0.))
    for insertion in (' ', 'A', 'B', 'H', 'K', 'L'):
        chain.add(atom_residue(123, insertion))
    residues = sf.build_residues_by_aho(chain)
    assert len(residues) == 7
    assert residues[123].id == (' ',123,' ')
    assert sf.compute_cdr3_length(residues) == 6
    assert sf.compute_cdr3_fr2_contact_nres(residues, iter([123]), soft=False) == 6
    assert sf.compute_cdr3_fr2_contacts(residues, iter([123])) == 1
    assert set(sf.cdr3_fr2_residue_min_distances(residues, [123])) == {123,'123A','123B','123H','123K','123L'}


def test_f04_framework_correspondence_uses_base_residue(aho):
    changed = aho.copy()
    changed[0]['B'].add(atom_residue(44, 'A', distance=100.))
    assert sf.calculate_framework_rmsd(changed, 'B', aho, 'B') == pytest.approx(0., abs=1e-6)


def test_f04_insertion_identity_survives_renumbering_and_file_roundtrip(aho, tmp_path):
    original = list(aho[0]['B'])[:4]
    mapping = {str(residue.id[1]): label for residue, label in
               zip(original, [123, '123H', '123K', '123L'])}
    _, path = sn.renumber_structure_to_aho(aho, 'B', mapping, temp_dir=tmp_path)
    residues = sf.build_residues_by_aho(sf.load_structure(str(path))[0]['B'])
    assert set(residues) == {123, '123H', '123K', '123L'}


def test_f05_mixed_fab_and_vhh_detection(tmp_path):
    path = DATA / 'pdbs/edge_cases/9bt8.pdb'
    assert sn.identify_nanobody_chains_from_pdb(str(path)) == ['A']
    result = nbframe.classify_all_nanobodies_in_pdb(str(path))
    assert set(result) == {'A'}
    assert result['A']['prob_kinked'] == pytest.approx(0.8151889824302802, abs=1e-8)
    # A genuine isolated VH/VL pair must still be excluded, independent of IDs.
    structure = sf.load_structure(str(path))
    for chain in list(structure[0]):
        if chain.id not in ('H','L'):
            structure[0].detach_child(chain.id)
    structure[0]['H'].id = 'X'
    structure[0]['L'].id = 'Y'
    writer = PDBIO(); writer.set_structure(structure); writer.save(str(tmp_path/'fab.pdb'))
    assert sn.identify_nanobody_chains_from_pdb(str(tmp_path/'fab.pdb')) == []


@pytest.mark.parametrize('name,element', [('H','H'), ('1HB','H'), ('2HD','D'), ('1HB',''), ('1HB',None)])
def test_f06_hydrogens_cannot_create_heavy_atom_contacts(name, element):
    residues = {44: atom_residue(44, distance=0.), 120: atom_residue(120, distance=6.)}
    expected = sf.compute_cdr3_fr2_contact_nres(residues,[120])
    for number, coordinate in [(44,1.),(120,5.)]:
        atom = Atom(name,np.array([coordinate,0.,0.]),0.,1.,' ',name,1000,element='H')
        atom.element = element
        residues[number].add(atom)
    assert sf.compute_cdr3_fr2_contact_nres(residues,[120]) == expected
    assert sf.compute_cdr3_fr2_contacts(residues,[120]) == 0.


def test_f06_rsa_ignores_hydrogens_without_mutating_input(aho):
    structure = aho.copy()
    expected = sf.compute_fr2_rsa(structure,'B')
    residue = structure[0]['B'][44]
    residue.add(Atom('1HB',residue['CA'].coord+np.array([1.,0.,0.]),0.,1.,' ','1HB ',10000,element='H'))
    assert sf.compute_fr2_rsa(structure,'B') == expected
    assert '1HB' in residue


def test_f09_long_mmcif_chain_and_export(tmp_path):
    source = sf.load_structure(str(PDB)); source[0]['B'].id = 'AA'
    io = MMCIFIO(); io.set_structure(source); io.save(str(tmp_path/'input.cif'))
    expected = nbframe.classify_structure(str(PDB),chain_id='B')
    actual = nbframe.classify_structure(str(tmp_path/'input.cif'),chain_id='AA',
                                        save_pdb=True,aho_output_dir=str(tmp_path/'export'))
    assert actual['prob_kinked'] == pytest.approx(expected['prob_kinked'],abs=1e-8)
    files = list((tmp_path/'export').glob('*.cif'))
    assert len(files) == 1
    assert 'AA' in sf.load_structure(str(files[0]))[0]


def test_f09_directory_cli_discovers_mmcif(tmp_path):
    result = CliRunner().invoke(app,['classify-structure','-d',str(DATA/'pdbs/mmcif'),
                                    '--output-csv',str(tmp_path/'results.csv')])
    assert result.exit_code == 0, result.output
    assert set(pd.read_csv(tmp_path/'results.csv').pdb_name) == {'9bsv.cif','9bdo.cif'}


def test_f10_directory_dataframe_and_csv_are_valid_classifier_inputs(tmp_path):
    inputs = tmp_path/'input'; inputs.mkdir()
    shutil.copyfile(DATA/'pdbs/kinked/9bsv.pdb',inputs/'sample.pdb')
    output = tmp_path/'features.csv'
    df = nbframe.compute_features_for_pdb_directory(inputs,output_csv=output)
    _, meta = load_structure_classifier()
    assert set(meta.feature_cols) <= set(df.columns)
    assert set(df.columns) == set(pd.read_csv(output).columns)
    assert predict_structure_from_features(df.iloc[0].to_dict())['label'] == 'kinked'
    assert predict_structure_from_features(pd.read_csv(output).iloc[0].to_dict())['label'] == 'kinked'
    empty = tmp_path/'empty'; empty.mkdir()
    assert list(nbframe.compute_features_for_pdb_directory(empty,as_dataframe=True).columns) == list(df.columns)


def test_f11_recursive_duplicate_names_and_exports(tmp_path):
    inputs=tmp_path/'inputs'; exports=tmp_path/'exports'
    paths=[]
    for name in ('first','second'):
        directory=inputs/name; directory.mkdir(parents=True)
        path=directory/'sample.pdb'; shutil.copyfile(PDB,path); paths.append(str(path))
    results=nbframe.compute_features_for_pdb_directory(inputs,recursive=True,save_pdb=True,save_pdb_dir=exports)
    assert set(results) == {'first/sample.pdb','second/sample.pdb'}
    assert len(list(exports.glob('*.pdb'))) == 2
    batch_exports=tmp_path/'batch'
    sn.compute_features_for_pdbs(paths,chain_ids=['B','B'],save_pdb=True,aho_output_dir=batch_exports)
    assert len(list(batch_exports.glob('*.pdb'))) == 2


def test_f12_all_chains_filtered_cli_has_actionable_message():
    with pytest.warns(UserWarning, match='Filtered'):
        result=CliRunner().invoke(app,['classify-structure','-p',str(PDB),'--rmsd-threshold','0'])
    assert result.exit_code == 1
    assert not isinstance(result.exception,StopIteration)
    assert 'prediction withheld (filtered)' in result.output
    assert 'framework RMSD' in result.output


def test_f12_zero_progress_is_rejected_before_work(tmp_path):
    result=CliRunner().invoke(app,['classify-structure','-d',str(tmp_path),'--progress-interval','0'])
    assert result.exit_code == 2
    assert not isinstance(result.exception,ZeroDivisionError)
    assert '--progress-interval' in unstyle(result.output)


def test_f12_insufficient_coverage_has_its_own_reason():
    with patch.object(sn, 'calculate_framework_rmsd', return_value=None):
        with pytest.warns(UserWarning, match='insufficient matching framework CA coverage') as caught:
            assert sn.compute_features_from_pdb(str(PDB), chain_id='B') is None
    assert all('exceeds' not in str(warning.message) for warning in caught)


def test_f13_default_temporary_output_is_owned_by_caller(aho):
    mapping={str(residue.id[1]):residue.id[1] for residue in aho[0]['B']}
    _, path=sn.renumber_structure_to_aho(aho,'B',mapping)
    try:
        assert path.is_file()
        assert 'B' in sf.load_structure(str(path))[0]
    finally:
        shutil.rmtree(path.parent)


def test_f13_batch_context_still_cleans_up(aho):
    mapping={str(residue.id[1]):residue.id[1] for residue in aho[0]['B']}
    with sn.RenumberingBatchContext() as context:
        _, path=sn.renumber_structure_to_aho(aho,'B',mapping,temp_dir=context.temp_dir)
        assert path.exists()
    assert not path.exists()
