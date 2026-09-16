"""Conservative coordinate-completeness checks for structure-model inputs.

These checks describe the resolved coordinates, not experimental confidence or
coverage of an unavailable full sequence. AHo numbering gaps alone are legal.
"""
from __future__ import annotations

import math
import numpy as np

from .structure_config import CDR3_AHOS, FR2_CONTACT_AHOS, FR2_KEY_RSA_AHOS

# Standard residue heavy atoms; terminal OXT and hydrogen atoms are optional.
_SIDECHAIN_ATOMS = {
    'ALA': 'CB', 'ARG': 'CB CG CD NE CZ NH1 NH2', 'ASN': 'CB CG OD1 ND2',
    'ASP': 'CB CG OD1 OD2', 'CYS': 'CB SG', 'GLN': 'CB CG CD OE1 NE2',
    'GLU': 'CB CG CD OE1 OE2', 'GLY': '', 'HIS': 'CB CG ND1 CD2 CE1 NE2',
    'ILE': 'CB CG1 CG2 CD1', 'LEU': 'CB CG CD1 CD2', 'LYS': 'CB CG CD CE NZ',
    'MET': 'CB CG SD CE', 'PHE': 'CB CG CD1 CD2 CE1 CE2 CZ', 'PRO': 'CB CG CD',
    'SER': 'CB OG', 'THR': 'CB OG1 CG2', 'TRP': 'CB CG CD1 CD2 NE1 CE2 CE3 CZ2 CZ3 CH2',
    'TYR': 'CB CG CD1 CD2 CE1 CE2 CZ OH', 'VAL': 'CB CG1 CG2',
}
CONTACT_FEATURES = ['contact_nres', 'contact_density']
RSA_FEATURES = ['fr2_rsa_key']


def _label(residue):
    return f'{residue.id[1]}{residue.id[2].strip()}'


def is_usable_atom(atom):
    occupancy = atom.get_occupancy()
    return (np.isfinite(atom.coord).all() and
            (occupancy is None or (math.isfinite(occupancy) and occupancy > 0)))


def add_quality_issue(report, code, region, message, features=(), **details):
    report['issues'].append(dict(code=code, region=region, message=message,
                                 affected_features=list(features), **details))
    if features:
        report['status'] = 'insufficient_quality'
        report['invalid_features'] = sorted(set(report['invalid_features']) | set(features))


def assess_structure_quality(chain, angles):
    """Assess selected conformers, preserving inserted residue identities.

    Completeness fractions refer to atoms expected for *observed* standard
    residues. Full sequence coverage cannot be inferred from an AHo gap.
    Missing FR2 base positions and observed peptide breaks are checked separately.
    """
    report = dict(version=1, status='passed', chain_id=chain.id,
                  model_id=chain.parent.id if chain.parent is not None else None,
                  sequence_coverage='not_assessed', regions={}, issues=[], invalid_features=[])
    residues = [r for r in chain if r.id[0] == ' ']
    for region, positions in [('cdr3', CDR3_AHOS), ('fr2', FR2_CONTACT_AHOS),
                              ('rsa_key', FR2_KEY_RSA_AHOS)]:
        selected = [r for r in residues if r.id[1] in positions]
        affected = CONTACT_FEATURES + RSA_FEATURES if region != 'rsa_key' else RSA_FEATURES
        expected_count = observed_count = 0
        incomplete = []
        for residue in selected:
            sidechain = _SIDECHAIN_ATOMS.get(residue.resname)
            if sidechain is None:
                add_quality_issue(report, 'unsupported_residue', region,
                                  f'{region}: unsupported residue {residue.resname} at {_label(residue)}.',
                                  affected, residue=_label(residue))
                continue
            expected = set(('N CA C O ' + sidechain).split())
            missing = sorted(a for a in expected if a not in residue or not is_usable_atom(residue[a]))
            expected_count += len(expected)
            observed_count += len(expected) - len(missing)
            if missing:
                incomplete.append(_label(residue))
                add_quality_issue(report, 'incomplete_heavy_atoms', region,
                                  f'{region}: residue {_label(residue)} has missing or unusable atoms: {", ".join(missing)}.',
                                  affected, residue=_label(residue), atoms=missing)
        report['regions'][region] = dict(resolved_residues=len(selected),
            expected_heavy_atoms=expected_count, usable_heavy_atoms=observed_count,
            atom_completeness=observed_count / expected_count if expected_count else None,
            incomplete_residues=incomplete)
        # These framework base positions are required by the fixed feature
        # definitions. Variable-loop numbering gaps are not missing-residue counts.
        required = positions if region in ('fr2', 'rsa_key') else []
        present = {r.id[1] for r in selected if not r.id[2].strip()}
        missing_positions = sorted(set(required) - present)
        report['regions'][region]['missing_required_positions'] = missing_positions
        if missing_positions or not selected:
            add_quality_issue(report, 'missing_region_residues', region,
                              f'{region}: required residues are absent ({missing_positions or "empty region"}).',
                              affected, positions=missing_positions)

    for region, lo, hi in [('cdr3', 107, 139), ('fr2', 43, 56)]:
        segment = [r for r in residues if lo <= r.id[1] <= hi]
        breaks = []
        for left, right in zip(segment, segment[1:]):
            pairs = [(left, 'C', right, 'N', 0.8, 2.0),
                     (left, 'CA', right, 'CA', 2.5, 4.5)]
            connected = all(a in l and b in r and is_usable_atom(l[a]) and is_usable_atom(r[b])
                            and low <= float(l[a] - r[b]) <= high
                            for l, a, r, b, low, high in pairs)
            if not connected:
                breaks.append([_label(left), _label(right)])
        report['regions'][region]['backbone_breaks'] = breaks
        if breaks:
            add_quality_issue(report, 'backbone_discontinuity', region,
                              f'{region}: backbone continuity cannot be established at {breaks}.',
                              CONTACT_FEATURES + RSA_FEATURES, residue_pairs=breaks)

    for end in ('N', 'C'):
        names = [f'alpha_{end}', f'tau_{end}', f'cos_alpha_{end}']
        valid = all(angles.get(name) is not None and math.isfinite(angles[name]) for name in names[:2])
        breakpoint = 108 if end == 'N' else 137
        index = next((i for i, r in enumerate(residues) if r.id == (' ', breakpoint, ' ')), None)
        anchors = residues[max(0, index - 1):index + 3] if index is not None else []
        report['regions'][f'angle_{end}'] = {
            'measurable': valid, 'anchor_residues': [_label(r) for r in anchors],
        }
        if not valid:
            add_quality_issue(report, 'unmeasurable_angle', f'angle_{end}',
                              f'CDR3 {end}-terminal angle anchors are missing, disconnected, or invalid.', names)

    # SASA uses the full selected chain. Non-finite coordinates anywhere would
    # make the spatial calculation invalid, even outside the measured regions.
    if any(not np.isfinite(atom.coord).all() for atom in chain.get_atoms()):
        add_quality_issue(report, 'nonfinite_coordinates', 'chain',
                          'The selected chain contains non-finite coordinates.', RSA_FEATURES)
    if any(atom.is_disordered() for atom in chain.get_atoms()):
        add_quality_issue(report, 'alternate_locations', 'chain',
                          'Alternate atom locations are present; checks and features use the selected conformers.')
    return report
