"""Release regressions: complete reporting, copy selection, and configuration."""
import json
from pathlib import Path
import shutil

import pandas as pd
import pytest
from typer.testing import CliRunner

from nbframe.cli import app
from nbframe import structure_classifier as sc
from nbframe.sequence_predictor import classify_sequence, classify_sequences

DATA = Path(__file__).parent / 'data'
COPIES = DATA / 'pdbs/extended/4cdg.pdb'
GOOD = DATA / 'pdbs/kinked/9bsv.pdb'


def test_every_copy_retained_and_usable_representative():
    results = sc.classify_all_nanobodies_in_pdb(str(COPIES))
    assert set(results) == {'C', 'D'}
    assert results['C']['status'] == 'insufficient_quality'
    assert results['D']['status'] == 'classified'
    grouped = sc.classify_all_nanobodies_in_pdb(str(COPIES), unique_sequences=True)
    assert set(grouped) == {'D'}
    group = grouped['D']['sequence_group']
    assert group['chain_ids'] == ['C', 'D']
    assert [r['status'] for r in group['results']] == ['insufficient_quality', 'classified']
    assert group['prob_kinked_range'] == [results['D']['prob_kinked']] * 2
    selected = sc.classify_structure(str(COPIES))
    assert selected['chain_id_used'] == 'D'
    assert selected['prob_kinked'] == results['D']['prob_kinked']


def test_group_reports_disagreement_without_selecting_by_confidence(monkeypatch):
    def predict(path, chain_id, **kwargs):
        prob = 0.99 if chain_id == 'C' else 0.1
        return dict(status='classified', chain_id_used=chain_id,
                    label='kinked' if chain_id == 'C' else 'extended', prob_kinked=prob,
                    features={'framework_rmsd': 0.8 if chain_id == 'C' else 0.4})
    monkeypatch.setattr(sc, 'classify_structure', predict)
    group = sc.classify_all_nanobodies_in_pdb(str(COPIES), unique_sequences=True)['D']['sequence_group']
    assert group['label_disagreement'] is True
    assert group['prob_kinked_range'] == [0.1, 0.99]
    assert len(group['results']) == 2


def test_chain_error_does_not_drop_good_chain():
    results = sc.classify_all_nanobodies_in_pdb(str(GOOD), chain_ids=['missing', 'D'])
    assert results['missing']['status'] == 'error'
    assert 'not found' in results['missing']['error']
    assert results['D']['status'] == 'classified'


def test_input_failure_has_record_and_strict_mode_raises(tmp_path):
    bad = tmp_path / 'bad.pdb'
    bad.write_text('invalid input\n')
    result = sc.classify_all_nanobodies_in_pdb(str(bad), strict=False)['__input__']
    assert result['status'] == 'error' and result['chain_id_used'] is None
    assert result['prob_kinked'] is None and result['error']
    assert sc.classify_structure(str(bad), strict=False)['status'] == 'error'
    with pytest.raises(ValueError):
        sc.classify_all_nanobodies_in_pdb(str(bad))


def test_filtered_api_and_cli_retain_reason(tmp_path):
    result = sc.classify_structure(str(GOOD), 'D', rmsd_threshold=0)
    assert result['status'] == 'filtered' and result['prob_kinked'] is None
    assert result['features']['framework_rmsd'] > 0
    output = tmp_path / 'out.json'
    csv = tmp_path / 'out.csv'
    run = CliRunner().invoke(app, ['classify-structure', '-p', str(GOOD), '--chain', 'D',
        '--rmsd-threshold', '0', '--output-json', str(output), '--output-csv', str(csv)])
    assert run.exit_code == 1, run.output
    assert json.loads(output.read_text())['status'] == 'filtered'
    assert pd.read_csv(csv).iloc[0]['status'] == 'filtered'


def test_mixed_directory_preserves_invalid_file(tmp_path):
    inputs = tmp_path / 'inputs'; inputs.mkdir()
    shutil.copy(GOOD, inputs / 'good.pdb')
    (inputs / 'broken.pdb').write_text('invalid input\n')
    output = tmp_path / 'out.json'; csv = tmp_path / 'out.csv'
    run = CliRunner().invoke(app, ['classify-structure', '-d', str(inputs),
        '--output-json', str(output), '--output-csv', str(csv)])
    assert run.exit_code == 0, run.output
    payload = json.loads(output.read_text())
    assert len(payload) == 2
    assert payload[str(inputs / 'broken.pdb')]['__input__']['status'] == 'error'
    failed = pd.read_csv(csv).query("status == 'error'").iloc[0]
    assert pd.isna(failed['chain_id']) and pd.notna(failed['error'])


def test_grouped_summary_retains_copies_without_features(tmp_path):
    output = tmp_path / 'out.json'; csv = tmp_path / 'out.csv'
    run = CliRunner().invoke(app, ['classify-structure', '-p', str(COPIES),
        '--unique-sequences', '--summary-only', '--output-json', str(output), '--output-csv', str(csv)])
    assert run.exit_code == 0, run.output
    payload = json.loads(output.read_text())
    assert 'features' not in payload
    copies = payload['sequence_group']['results']
    assert len(copies) == 2 and all('features' not in r for r in copies)
    assert all('quality' in r for r in copies)
    assert len(json.loads(pd.read_csv(csv).iloc[0]['sequence_group'])['results']) == 2


@pytest.mark.parametrize('kinked,extended', [(float('nan'), .2), (.7, float('inf')), (1.1,.2), (.7,-.1), (.2,.7), (.5,.5), (True,.2)])
def test_thresholds_rejected_even_without_inputs(kinked, extended):
    for call in (lambda: classify_sequences([], kinked_threshold=kinked, extended_threshold=extended),
                 lambda: classify_sequence('bad', kinked_threshold=kinked, extended_threshold=extended),
                 lambda: sc.classify_all_nanobodies_in_pdb('missing', kinked_threshold=kinked, extended_threshold=extended)):
        with pytest.raises(ValueError, match='Thresholds'):
            call()


@pytest.mark.parametrize('value', [-1, float('nan'), float('inf'), True])
def test_invalid_rmsd_rejected(value):
    with pytest.raises(ValueError, match='rmsd_threshold'):
        sc.classify_structure('missing', rmsd_threshold=value)


def test_cli_validates_thresholds_without_label():
    run = CliRunner().invoke(app, ['classify-sequence', '-s', 'bad', '--no-label', '--kinked-threshold', 'nan'])
    assert run.exit_code == 2 and 'Thresholds' in run.output
    run = CliRunner().invoke(app, ['classify-structure', '-p', 'missing', '--rmsd-threshold', 'nan'])
    assert run.exit_code == 2 and 'rmsd_threshold' in run.output


def test_metadata_thresholds_used_in_api_and_cli(monkeypatch, tmp_path):
    from dataclasses import replace
    monkeypatch.setattr(sc, '_STRUCTURE_METADATA', replace(sc._load_metadata(), kinked_threshold=1., extended_threshold=0.))
    result = sc.classify_structure(str(GOOD), 'D')
    assert result['label'] == 'uncertain'
    output = tmp_path / 'out.json'
    run = CliRunner().invoke(app, ['classify-structure', '-p', str(GOOD), '--chain', 'D', '--output-json', str(output)])
    assert run.exit_code == 0, run.output
    payload = json.loads(output.read_text())
    assert payload['label'] == 'uncertain'
    assert payload['model_info']['thresholds'] == {'kinked': 1., 'extended': 0.}


def test_sequence_csv_has_labels_and_invalid_records(tmp_path):
    fasta = DATA / 'sequences/batch_mixed.fasta'
    from Bio import SeqIO
    sequence = str(next(SeqIO.parse(fasta, 'fasta')).seq)
    single_csv = tmp_path / 'single.csv'; batch_csv = tmp_path / 'batch.csv'
    inputs = tmp_path / 'inputs.fasta'; inputs.write_text(f'>good\n{sequence}\n>bad\nBAD!\n')
    runner = CliRunner()
    single = runner.invoke(app, ['classify-sequence', '-s', sequence, '-o', str(single_csv)])
    batch = runner.invoke(app, ['classify-sequence', '-f', str(inputs), '-o', str(batch_csv)])
    assert single.exit_code == batch.exit_code == 0
    a, b = pd.read_csv(single_csv), pd.read_csv(batch_csv)
    assert a.iloc[0]['label'] == b.iloc[0]['label']
    assert b.iloc[1]['status'] == 'error' and pd.notna(b.iloc[1]['error'])
    bad = runner.invoke(app, ['classify-sequence', '-s', 'BAD!', '-o', str(single_csv)])
    assert bad.exit_code == 1 and pd.read_csv(single_csv).iloc[0]['status'] == 'error'
    inputs.write_text('>bad\nBAD!\n')
    bad_batch = runner.invoke(app, ['classify-sequence', '-f', str(inputs), '-o', str(batch_csv)])
    assert bad_batch.exit_code == 1 and pd.read_csv(batch_csv).iloc[0]['status'] == 'error'
