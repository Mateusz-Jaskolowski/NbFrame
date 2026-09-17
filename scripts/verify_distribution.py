"""Exercise an installed wheel from outside the source checkout."""
from importlib.metadata import version
from pathlib import Path
import shutil
import subprocess
import sys

import nbframe
from nbframe.sequence_predictor import classify_sequence
from nbframe.structure_classifier import classify_structure, classify_all_nanobodies_in_pdb

source = Path(sys.argv[1]).resolve()
installed = Path(nbframe.__file__).resolve()
assert not installed.is_relative_to(source), f'Imported checkout instead of installed wheel: {installed}'
assert version('nbframe') == nbframe.__version__ == '0.3.0'
assert version('scikit-learn') == '1.7.2'
assert shutil.which('hmmscan'), 'HMMER executable is unavailable'
sequence = 'QVQLVESGGGSVQAGGSLRLSCAASGYTVSTYCMGWFRQAPGKEREGVATILGGSTYYGDSVKGRFTISQDNAKNTVYLQMNSLKPEDTAIYYCAGSTVASTGWCSRLRPYDYHYRGQGTQVTVSS'
result = classify_sequence(sequence)
assert result['status'] == 'classified' and 0 <= result['prob_kinked'] <= 1, result
for filename in ('pdbs/kinked/9bsv.pdb', 'pdbs/mmcif/9bsv.cif'):
    result = classify_structure(str(source / 'tests/data' / filename), 'D')
    assert result['status'] == 'classified' and result['label'] == 'kinked', result
copies = classify_all_nanobodies_in_pdb(str(source / 'tests/data/pdbs/extended/4cdg.pdb'))
assert copies['C']['status'] == 'insufficient_quality'
assert copies['D']['status'] == 'classified'
subprocess.run(['nbframe', '--help'], check=True, stdout=subprocess.DEVNULL)
print(f'Installed NbFrame {nbframe.__version__} verified at {installed}')
