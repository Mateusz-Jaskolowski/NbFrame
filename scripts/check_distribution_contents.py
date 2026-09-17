"""Check built distributions for required resources and accidental inclusions."""
from pathlib import Path
import sys
import tarfile
import zipfile

root = Path(sys.argv[1])
wheels = list(root.glob('*.whl'))
sdists = list(root.glob('*.tar.gz'))
assert len(wheels) == len(sdists) == 1, 'Expected one wheel and one source archive'
for artifact in [*wheels, *sdists]:
    if artifact.suffix == '.whl':
        with zipfile.ZipFile(artifact) as archive:
            names = archive.namelist()
        assert all(n.startswith('nbframe/') or '.dist-info/' in n for n in names), names
    else:
        with tarfile.open(artifact) as archive:
            names = [n.partition('/')[2] for n in archive.getnames()]
    assert not any('docs/review-' in n or '/archive/' in n for n in names), names
    for suffix in ('.joblib', '.json', '.pdb'):
        assert any(n.startswith('nbframe/data/') and n.endswith(suffix) for n in names), suffix
    print(f'Checked {artifact.name}: {len(names)} entries')
