#
# Copyright The NOMAD Authors.
#
# This file is part of NOMAD. See https://nomad-lab.eu for further info.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import datetime
import numpy as np
import ase.io
import os
import re

from nomad.datamodel import EntryArchive
from nomad.units import ureg as units
from nomad.utils import extract_section
from nomad.parsing.parsers import _compressions
from runschema.run import Run, Program, TimeRun
from runschema.system import System, Atoms
from runschema.method import (
    Method,
    Electronic,
    BasisSet,
    BasisSetContainer,
)
from simulationworkflowschema import SinglePoint, SimulationWorkflow, SerialSimulation
from nomad.datamodel.metainfo.workflow import TaskReference, Link
from runschema.calculation import Calculation, Dos, DosValues, Charges

from nomad.parsing.file_parser import TextParser, Quantity
from glob import glob

from .metainfo.lobster import (
    x_lobster_section_cohp,
    x_lobster_section_coop,
    x_lobster_section_cobi,
    x_lobster_section_cohp_orbital_label,
    x_lobster_section_coop_orbital_label,
    x_lobster_section_cobi_orbital_label,
    x_lobster_section_cohp_orbital_pair,
    x_lobster_section_coop_orbital_pair,
    x_lobster_section_cobi_orbital_pair,
)
# from .workflow import LOBSTERWorkflow

"""
This is a LOBSTER code parser.
"""

e = (1 * units.e).to_base_units().magnitude
eV = (1 * units.eV).to_base_units().magnitude
re_float = r'[-+]?\d+\.\d*(?:[Ee][-+]\d+)?'


def _normalize_label(label):
    return str(label).strip()


def _normalize_orbital_label(label):
    text = str(label).strip()
    if '[' in text and ']' in text:
        text = text.replace('[', '_').replace(']', '')
    return text


def _normalize_fermi_orbital_values(values):
    """
    Normalize fermi orbital values by flattening if needed.
    Uses numpy operations to avoid list overhead for large data.

    Args:
        values: Input values (list or array)

    Returns:
        numpy array with appropriate shape

    Raises:
        ValueError: If values cannot be converted to numeric array
    """
    try:
        # Convert to numpy array if not already
        values_array = np.asarray(values)

        # Check if we have a 2D array with shape (n, 1) - equivalent to [[x], [y], [z]]
        # Flatten to 1D in this case
        if values_array.ndim == 2 and values_array.shape[1] == 1:
            return values_array.flatten()

        return values_array
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f'Failed to normalize Fermi orbital values. '
            f'Expected numeric array, got: {type(values).__name__}'
        ) from exc

_LABEL_SECTION_ATTR = {
    'hp': 'x_lobster_cohp_orbital_per_label',
    'op': 'x_lobster_coop_orbital_per_label',
    'bi': 'x_lobster_cobi_orbital_per_label',
}

_LABEL_SECTION_CLASS = {
    'hp': x_lobster_section_cohp_orbital_label,
    'op': x_lobster_section_coop_orbital_label,
    'bi': x_lobster_section_cobi_orbital_label,
}

_ORBITAL_PAIR_CLASS = {
    'hp': x_lobster_section_cohp_orbital_pair,
    'op': x_lobster_section_coop_orbital_pair,
    'bi': x_lobster_section_cobi_orbital_pair,
}


def get_lobster_file(filename):
    compressions = [''] + [v[0] for v in _compressions.values()]
    for compression in compressions:
        name = f'{filename}.{compression}'
        if os.path.isfile(name):
            return name
    return filename


def orb_coxp_icoxp_to_joule(icoxp_pairs, conversion_factor, data_type: str):
    """
    Convert the COXP/ICOXP values to joules

    Args:
        icoxp_pairs: list of COXP/ICOXP pairs
        conversion_factor: conversion factor to joules
        data_type: type of input data (hp or op or bi)
    """
    if data_type == 'hp' and isinstance(icoxp_pairs[0][0], np.ndarray):
        return [
            [[orb * conversion_factor for orb in orb_pair] for orb_pair in bond]
            for bond in icoxp_pairs
        ]
    elif data_type == 'hp' and isinstance(icoxp_pairs[0][0], float):
        return [
            [[orb_pair * conversion_factor] for orb_pair in bond]
            for bond in icoxp_pairs
        ]
    elif (
        data_type == 'hp'
        and isinstance(icoxp_pairs[0][0], tuple)
        and len(icoxp_pairs[0][0]) != 2
    ):
        return [
            [[orb * conversion_factor for orb in orb_pair] for orb_pair in bond]
            for bond in icoxp_pairs
        ]
    elif (
        data_type == 'hp'
        and isinstance(icoxp_pairs[0][0], list)
        and len(icoxp_pairs[0][0]) == 2
    ):
        return [
            [
                [[spin * conversion_factor for spin in orb] for orb in orb_pair]
                for orb_pair in bond
            ]
            for bond in icoxp_pairs
        ]
    else:
        return icoxp_pairs


def parse_ICOXPLIST(fname, scc, method, version):
    def icoxp_line_split(string):
        tmp = string.split()
        # LOBSTER version 3 and above
        if len(tmp) == 8:
            return [
                tmp[0],
                tmp[1],
                tmp[2],
                float(tmp[3]),
                [int(tmp[4]), int(tmp[5]), int(tmp[6])],
                float(tmp[7]),
            ]
        elif len(tmp) == 9 and not tmp[-1].isdigit():
            # Spin polarized data LOBSTER version 5.1 and above
            return [
                tmp[0],
                tmp[1],
                tmp[2],
                float(tmp[3]),
                [int(tmp[4]), int(tmp[5]), int(tmp[6])],
                [float(tmp[7]), float(tmp[8])],
            ]
        elif len(tmp) == 9 and tmp[-1].isdigit():
            # Non-Spin polarized data LOBSTER version 5.1 and above
            return [
                tmp[0],
                tmp[1],
                tmp[2],
                float(tmp[3]),
                [int(tmp[4]), int(tmp[5]), int(tmp[6])],
                float(tmp[7]),
            ]
        # LOBSTER versions below 3
        elif len(tmp) == 6:
            return [tmp[0], tmp[1], tmp[2], float(tmp[3]), float(tmp[4]), int(tmp[5])]

    float_version = float(version.split('.')[0] + '.' + version.split('.')[1])
    if 5 > float_version > 2:
        icoxplist_parser = TextParser(
            quantities=[
                Quantity(
                    'icoxpslist_for_spin',
                    r' *(CO[O,H,B,I,P]).*spin *\d *([^#]+[-\d\.]+)',
                    repeats=True,
                    sub_parser=TextParser(
                        quantities=[
                            Quantity(
                                'line',
                                # LOBSTER version 3 and above
                                r'( *\d+ +[^ ]+ +[^ ]+ +[\.\d]+ +[-\d]+ +[-\d]+ +[-\d]+ +[-\.\d]+ *)',
                                repeats=True,
                                str_operation=icoxp_line_split,
                            )
                        ]
                    ),
                )
            ]
        )
    elif float_version >= 5:
        icoxplist_parser = TextParser(
            quantities=[
                Quantity(
                    'icoxpslist_for_spin',
                    r' *(CO[O,H,B,I,P]).*\n .*([^#]+[-\d\.]+)',
                    repeats=True,
                    sub_parser=TextParser(
                        quantities=[
                            Quantity(
                                'line',
                                # LOBSTER version 5.1 and above
                                r'( *\d+ +[^ ]+ +[^ ]+ +[\.\d]+ +[-\d]+ +[-\d]+ +[-\d]+ +[-\.\d]+ +[-\.\d]+)',
                                repeats=True,
                                str_operation=icoxp_line_split,
                            )
                        ]
                    ),
                ),
                Quantity(
                    'icoxpslist_for_nsp',
                    r' *(CO[O,H,B,I,P]).*\n .*([^#]+[-\d\.]+)',
                    repeats=True,
                    sub_parser=TextParser(
                        quantities=[
                            Quantity(
                                'line',
                                # LOBSTER version 5.1 and above
                                r'( *\d+ +[^ ]+ +[^ ]+ +[\.\d]+ +[-\d]+ +[-\d]+ +[-\d]+ +[-\.\d]+ *)',
                                repeats=True,
                                str_operation=icoxp_line_split,
                            )
                        ]
                    ),
                ),
            ]
        )
    else:
        icoxplist_parser = TextParser(
            quantities=[
                Quantity(
                    'icoxpslist_for_spin',
                    r' *(CO[OH]P).*spin *\d *([^#]+[-\d\.]+)',
                    repeats=True,
                    sub_parser=TextParser(
                        quantities=[
                            Quantity(
                                'line',
                                # LOBSTER versions below 3
                                r'( *\d+ +[^ ]+ +[^ ]+ +[\.\d]+ +[-\.\d]+ +[\d]+ *)',
                                repeats=True,
                                str_operation=icoxp_line_split,
                            ),
                        ]
                    ),
                )
            ]
        )

    if not os.path.isfile(fname):
        return
    icoxplist_parser.mainfile = fname
    icoxplist_parser.parse()

    icoxp = []
    label_attr = _LABEL_SECTION_ATTR.get(method)
    label_class = _LABEL_SECTION_CLASS.get(method)
    for spin, icoxplist in enumerate(icoxplist_parser.get('icoxpslist_for_spin')):
        if spin == 0:
            if method == 'op':
                section = x_lobster_section_coop()
                scc.x_lobster_section_coop = section
            elif method == 'hp':
                section = x_lobster_section_cohp()
                scc.x_lobster_section_cohp = section
            elif method == 'bi':
                section = x_lobster_section_cobi()
                scc.x_lobster_section_cobi = section
        else:
            if method == 'op':
                section = scc.x_lobster_section_coop
            elif method == 'hp':
                section = scc.x_lobster_section_cohp
            elif method == 'bi':
                section = scc.x_lobster_section_cobi
        if icoxplist.get('line') is not None:
            raw_lines_sp = icoxplist.get('line')
        else:
            raw_lines_sp = []

        if icoxplist_parser.get('icoxpslist_for_nsp') is not None:
            raw_lines_nsp = icoxplist_parser.get('icoxpslist_for_nsp')[spin].get('line')
        else:
            raw_lines_nsp = []

        if len(raw_lines_nsp) > len(raw_lines_sp):
            raw_lines = raw_lines_nsp
        else:
            raw_lines = raw_lines_sp

        if raw_lines is None:
            break

        lines = [x for x in raw_lines if x[1].count('_') == 0]

        orb_data = {_normalize_label(line[0]): [] for line in lines}
        atom_orb_pairs = {_normalize_label(line[0]): [] for line in lines}
        atom_orb_icoxps = {_normalize_label(line[0]): [] for line in lines}


        for line in raw_lines:
            if line[1].count('_') > 0:
                key = _normalize_label(line[0])
                orb_data[key].append(line[1:])
                atom_orb_pairs[key].append(
                    [_normalize_orbital_label(line[1]), _normalize_orbital_label(line[2])]
                )
                atom_orb_icoxps[key].append(line[-1])


        if atom_orb_icoxps:
            orb_labels = list(atom_orb_icoxps.keys())
            orb_icoxps = list(atom_orb_icoxps.values())
            if len(orb_icoxps[0]) > 0:
                orb_icoxps = orb_coxp_icoxp_to_joule(
                    icoxp_pairs=list(atom_orb_icoxps.values()),
                    conversion_factor=eV,
                    data_type=method,
                )

            if label_attr and label_class:
                orbital_pair_class = _ORBITAL_PAIR_CLASS.get(method)
                # Get existing label sections from the parent section
                existing_label_sections = getattr(section, label_attr) or []
                # Create lookup dictionary for O(1) access
                label_section_map = {sec.x_lobster_pair_label: sec for sec in existing_label_sections}

                for label, values in zip(orb_labels, orb_icoxps):
                    # Find existing label section or create new one
                    label_section = label_section_map.get(label)

                    if label_section is None:
                        label_section = label_class()
                        label_section.x_lobster_pair_label = label
                        section.m_add_sub_section(label_attr, label_section)

                    orbital_pairs_list = atom_orb_pairs.get(label, [])

                    # Get existing orbital pairs or create new ones
                    existing_orbital_pairs = label_section.x_lobster_orbital_pairs or []

                    # Ensure we have enough orbital pair subsections
                    for pair_idx in range(len(existing_orbital_pairs), len(orbital_pairs_list)):
                        pair_section = orbital_pair_class()
                        pair_section.x_lobster_atom1_orbital = orbital_pairs_list[pair_idx][0]
                        pair_section.x_lobster_atom2_orbital = orbital_pairs_list[pair_idx][1]
                        label_section.m_add_sub_section('x_lobster_orbital_pairs', pair_section)

                    # Set fermi level values for each orbital pair
                    fermi_attr = 'x_lobster_integrated_orbital_co{}_at_fermi_level'.format(method)
                    normalized_values = _normalize_fermi_orbital_values(values)

                    orbital_pairs = label_section.x_lobster_orbital_pairs or []
                    for pair_idx, fermi_val in enumerate(normalized_values):
                        if pair_idx < len(orbital_pairs):
                            pair_section = orbital_pairs[pair_idx]
                            per_spin = getattr(pair_section, fermi_attr)

                            # Handle different fermi value formats
                            if isinstance(fermi_val, (list, tuple, np.ndarray)):
                                # Already have values for multiple spins [spin0, spin1, ...]
                                # This happens with LOBSTER 5.1+ format
                                per_spin = np.array(fermi_val)
                            else:
                                # Single value for current spin channel
                                if per_spin is None:
                                    per_spin = np.zeros(spin + 1)
                                elif len(per_spin) <= spin:
                                    new_per_spin = np.zeros(spin + 1)
                                    new_per_spin[:len(per_spin)] = per_spin
                                    per_spin = new_per_spin
                                per_spin[spin] = fermi_val

                            setattr(pair_section, fermi_attr, per_spin)

        if isinstance(lines[0][5], int):
            label, a1, a2, distances, tmp, bonds = zip(*lines)
        else:
            label, a1, a2, distances, v, tmp = zip(*lines)
        icoxp.append(0)
        icoxp[-1] = np.asarray(tmp)
        if spin == 0:
            setattr(
                section, 'x_lobster_number_of_co{}_pairs'.format(method), len(list(a1))
            )
            setattr(section, 'x_lobster_co{}_atom1_labels'.format(method), list(a1))
            setattr(section, 'x_lobster_co{}_atom2_labels'.format(method), list(a2))
            setattr(
                section,
                'x_lobster_co{}_distances'.format(method),
                np.array(distances) * units.angstrom,
            )

            # version specific entries
            if 'v' in locals():
                setattr(
                    section,
                    'x_lobster_co{}_translations'.format(method),
                    np.asarray(v),
                )
            if 'bonds' in locals():
                setattr(
                    section,
                    'x_lobster_co{}_number_of_bonds'.format(method),
                    np.asarray(bonds),
                )

    if len(icoxp) > 0 and not isinstance(icoxp[0][0], np.ndarray):
        setattr(
            section,
            'x_lobster_integrated_co{}_at_fermi_level'.format(method),
            np.array(icoxp) * units.eV if method == 'hp' else np.array(icoxp),
        )
    elif len(icoxp) > 0 and isinstance(icoxp[0][0], np.ndarray):
        setattr(
            section,
            'x_lobster_integrated_co{}_at_fermi_level'.format(method),
            np.array(icoxp[0]).T * units.eV if method == 'hp' else np.array(icoxp[0]).T,
        )

    icoxplist_parser.close()

def _split_coxp_pair_line(line):
    """Parse a raw pair line into [label, atom1, atom2, distance]."""
    match = re.match(r'\s*No\.(\d+)\:(.+?)\->(.+?)\(([-+]?\d+\.\d+(?:[Ee][-+]?\d+)?)\)\s*$', str(line))
    if match is None:
        return None

    label, atom1, atom2, distance = match.groups()
    return [label.strip(), atom1.strip(), atom2.strip(), float(distance)]


class COXPCARParser(TextParser):
    def init_quantities(self):
        self._quantities = [
            Quantity(
                'coxp_pairs',
                [r'(No\.\d+[^\n]*)'],  # Extract raw pair line text
                repeats=True,
            ),
            Quantity(
                'coxp_lines',
                [r'\s*(-*\d+\.\d+(?:[ \t]+-*\d+\.\d+)+)\s*'],
                repeats=True,
                dtype=np.float64,
            ),
        ]


def _coxp_exceeds_uncompressed_limit(fname, logger, limit_bytes):
    try:
        with open(fname, 'rb') as handle:
            compression, open_compressed = _compressions.get(handle.read(3), (None, open))
    except OSError as exc:
        logger.warning(f'Unable to read COXPCAR file size: {exc}. Skipping parsing.')
        return True

    if compression is None:
        try:
            return os.path.getsize(fname) > limit_bytes
        except OSError as exc:
            logger.warning(f'Unable to stat COXPCAR file: {exc}. Skipping parsing.')
            return True

    try:
        total = 0
        with open_compressed(fname) as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > limit_bytes:
                    return True
    except OSError as exc:
        logger.warning(f'Unable to read compressed COXPCAR file: {exc}. Skipping parsing.')
        return True

    return False


def parse_COXPCAR(fname, scc, method, logger):
    def _separate_orbital_data(pairs_list, coxp_lines_list):
        """
        Separate the data for atom pairs cohps and atom orbitals cohps

        Args:
            pairs_list: list of pairs
            coxp_lines_list: list of COXP lines
        """
        # Filter average coxp/icoxp for spin up and spin down
        filtered_coxp_lines = (
            coxp_lines_list[3 : (len(pairs_list) * 2) + 3]
            + coxp_lines_list[(len(pairs_list) * 2) + 5 :]
        )

        atom_pair_cohp = {pair[0]: [] for pair in pairs_list}
        atom_orb_cohp = {pair[0]: [] for pair in pairs_list}

        if len(pairs_list) * 4 + 5 == len(coxp_lines_list):  # spin polarized data
            pairs_list = [pair for pair in pairs_list for _ in range(2)] * 2
            for ix, (p, c) in enumerate(zip(pairs_list, filtered_coxp_lines)):
                if '[' not in p[1]:
                    atom_pair_cohp[p[0]].append(filtered_coxp_lines[ix])
                else:
                    atom_orb_cohp[p[0]].append(filtered_coxp_lines[ix])

        elif len(pairs_list) * 2 + 3 == len(coxp_lines_list):  # non spin polarized data
            pairs_list = [pair for pair in pairs_list for _ in range(2)]
            for ix, (p, c) in enumerate(zip(pairs_list, coxp_lines_list[3:])):
                if '[' not in p[1]:
                    atom_pair_cohp[p[0]].append(coxp_lines_list[3:][ix])
                else:
                    atom_orb_cohp[p[0]].append(coxp_lines_list[3:][ix])
        else:
            logger.warning(
                'Unexpected number of columns {} in CO{}CAR.lobster.'.format(
                    len(coxp_lines), method.upper()
                )
            )
            return

        return atom_pair_cohp, atom_orb_cohp

    def _get_pair_label_distance(pairs_list: list[list]):
        """
        Get the atom pair label and distance seperately for
        orbital and non-orbital pairs.

        Args:
            pairs_list: nested list of pairs
        """
        label = []
        atom1 = []
        atom2 = []
        distance = []
        label_orb = []
        atom1_orb = []
        atom2_orb = []
        distance_orb = []
        for lab, at1, at2, dist in pairs_list:
            if '[' not in at1:
                atom1.append(at1)
                atom2.append(at2)
                distance.append(dist)
                label.append(lab)
            else:
                atom1_orb.append(at1)
                atom2_orb.append(at2)
                distance_orb.append(dist)
                label_orb.append(lab)

        return (
            label,
            atom1,
            atom2,
            distance,
            label_orb,
            atom1_orb,
            atom2_orb,
            distance_orb,
        )

    def _group_orb_coxp_spin_data(orb_cohp_data: dict, spin_polarized: bool = False):
        """
        Group the orbital COXP/iCOXP data for spin polarized calculations

        Args:
            orb_cohp_data: dictionary of atom orbital COHP data
            spin_polarized: bool, whether the data is spin polarized or not
        """
        orb_coxp = []
        orb_icoxp = []
        for values in orb_cohp_data.values():
            if spin_polarized:
                num_pairs = len(values) // 2
                spin_up = values[:num_pairs]
                spin_dn = values[num_pairs:]

                # Separate COXP and iCOXP for spin-up and spin-down
                coxps_up = spin_up[0::2]
                coxps_dn = spin_dn[0::2]
                icoxps_up = spin_up[1::2]
                icoxps_dn = spin_dn[1::2]
                # Group spin-polarized data as [spin_up, spin_dn] for each pair
                coxps = [[up, dn] for up, dn in zip(coxps_up, coxps_dn)]
                icoxps = [[up, dn] for up, dn in zip(icoxps_up, icoxps_dn)]
                orb_coxp.append(coxps)
                orb_icoxp.append(icoxps)
            else:
                # Separate COXP and iCOXP for non-spin-polarized data
                coxps = values[0::2]
                icoxps = values[1::2]
                orb_coxp.append(coxps)
                orb_icoxp.append(icoxps)

        return orb_coxp, orb_icoxp

    def _get_orbital_pairs_by_label(pairs_list):
        pairs_by_label = {}
        for lab, at1, at2, _dist in pairs_list:
            if '[' not in at1:
                continue
            pairs_by_label.setdefault(lab, []).append(
                [_normalize_orbital_label(at1), _normalize_orbital_label(at2)]
            )
        return pairs_by_label

    coxpcar_parser = COXPCARParser()

    if not os.path.isfile(fname):
        return
    if _coxp_exceeds_uncompressed_limit(fname, logger, 2 * 1024**3):
        logger.warning(
            'Skipping CO{}CAR parsing because uncompressed size exceeds 2 GB.'.format(
                method.upper()
            )
        )
        return
    coxpcar_parser.line_parsing = True  # Enable streaming mode for memory efficiency
    coxpcar_parser.mainfile = fname
    coxpcar_parser.parse()

    if method == 'op':
        if not scc.x_lobster_section_coop:
            section = x_lobster_section_coop()
            scc.x_lobster_section_coop = section
        else:
            section = scc.x_lobster_section_coop
    elif method == 'hp':
        if not scc.x_lobster_section_cohp:
            section = x_lobster_section_cohp()
            scc.x_lobster_section_cohp = section
        else:
            section = scc.x_lobster_section_cohp
    elif method == 'bi':
        if not scc.x_lobster_section_cobi:
            section = x_lobster_section_cobi()
            scc.x_lobster_section_cobi = section
        else:
            section = scc.x_lobster_section_cobi

    # Extract raw pair lines and process them outside the parser
    raw_pair_lines = coxpcar_parser.pop('coxp_pairs')
    if raw_pair_lines:
        pairs = [_split_coxp_pair_line(line) for line in raw_pair_lines]
        pairs = [p for p in pairs if p is not None]  # Filter invalid parses
    else:
        pairs = []

    if not pairs:
        coxpcar_parser.close()
        logger.warning(
            'No CO{}P values detected in CO{}CAR.lobster.'.format(
                method.upper(), method.upper()
            )
        )
        return

    coxp_lines = coxpcar_parser.pop('coxp_lines')
    if coxp_lines is None:
        coxpcar_parser.close()
        logger.warning(
            'No CO{} values detected in CO{}CAR.lobster.'
            'The file is likely incomplete'.format(method.upper(), method.upper())
        )
        return
    coxp_lines = list(np.asarray(coxp_lines).T)

    spin_polarized = len(coxp_lines) == 4 * len(pairs) + 5

    _lab, a1, a2, distances, _lab_orb, _a1_orb, _a2_orb, _distances_orb = (
        _get_pair_label_distance(pairs)
    )
    atom_pair_cohp, atom_orb_cohp = _separate_orbital_data(pairs, coxp_lines)
    atom_orb_cohp = {
        _normalize_label(key): value for key, value in atom_orb_cohp.items()
    }
    orb_pairs_by_label = {
        _normalize_label(key): value
        for key, value in _get_orbital_pairs_by_label(pairs).items()
    }

    number_of_pairs = len(list(a1))  # excluding orbital pairs (atom pairs only)
    tot_interactions = len(pairs)  # including orbital pairs

    if not spin_polarized:
        acoxp = [coxp_lines[1]]
        aicoxp = [coxp_lines[2]]
        coxp = [[coxp_icoxp[0]] for coxp_icoxp in list(atom_pair_cohp.values())]
        icoxp = [[coxp_icoxp[1]] for coxp_icoxp in list(atom_pair_cohp.values())]
        coxp_orb, icoxp_orb = _group_orb_coxp_spin_data(
            atom_orb_cohp, spin_polarized=spin_polarized
        )

    elif spin_polarized:
        spin_dn_coxp_icoxp = coxp_lines[
            (tot_interactions * 2) + 3 : (tot_interactions * 2) + 5
        ]
        acoxp = [coxp_lines[1], spin_dn_coxp_icoxp[0]]
        aicoxp = [coxp_lines[2], spin_dn_coxp_icoxp[1]]

        coxp = [
            [coxp_icoxp[0], coxp_icoxp[2]]
            for coxp_icoxp in list(atom_pair_cohp.values())
        ]
        icoxp = [
            [coxp_icoxp[1], coxp_icoxp[3]]
            for coxp_icoxp in list(atom_pair_cohp.values())
        ]
        coxp_orb, icoxp_orb = _group_orb_coxp_spin_data(
            atom_orb_cohp, spin_polarized=spin_polarized
        )
        if len(icoxp_orb[0]) > 0:
            icoxp_orb = orb_coxp_icoxp_to_joule(
                icoxp_pairs=icoxp_orb, conversion_factor=eV, data_type=method
            )
        if len(coxp_orb[0]) > 0 and method == 'hp':
            coxp_orb = orb_coxp_icoxp_to_joule(
                icoxp_pairs=coxp_orb, conversion_factor=eV, data_type=method
            )

    setattr(section, 'x_lobster_number_of_co{}_pairs'.format(method), number_of_pairs)
    setattr(section, 'x_lobster_co{}_atom1_labels'.format(method), list(a1))
    setattr(section, 'x_lobster_co{}_atom2_labels'.format(method), list(a2))
    setattr(
        section,
        'x_lobster_co{}_distances'.format(method),
        np.array(distances) * units.angstrom,
    )

    setattr(
        section, 'x_lobster_number_of_co{}_values'.format(method), len(coxp_lines[0])
    )
    setattr(
        section,
        'x_lobster_co{}_energies'.format(method),
        np.array(coxp_lines[0]) * units.eV,
    )

    setattr(section, 'x_lobster_average_co{}_values'.format(method), np.array(acoxp))
    setattr(
        section,
        'x_lobster_average_integrated_co{}_values'.format(method),
        np.array(aicoxp) * units.eV if method == 'hp' else np.array(aicoxp),
    )
    setattr(section, 'x_lobster_co{}_values'.format(method), np.array(coxp))
    setattr(
        section,
        'x_lobster_integrated_co{}_values'.format(method),
        np.array(icoxp) * units.eV if method == 'hp' else np.array(icoxp),
    )
    if len(coxp_orb) > 0:
        label_attr = _LABEL_SECTION_ATTR.get(method)
        label_class = _LABEL_SECTION_CLASS.get(method)
        orbital_pair_class = _ORBITAL_PAIR_CLASS.get(method)
        if label_attr and label_class and orbital_pair_class:
            # Get existing label sections
            existing_sections = getattr(section, label_attr) or []
            # Create lookup dictionary for O(1) access
            label_section_map = {sec.x_lobster_pair_label: sec for sec in existing_sections}

            orb_labels = list(atom_orb_cohp.keys())
            label_to_index = {lab: idx for idx, lab in enumerate(orb_labels)}
            label_order = [
                _normalize_label(lab) for lab in _lab if _normalize_label(lab) in label_to_index
            ]

            for label in label_order:
                if label not in orb_pairs_by_label:
                    continue

                # Find existing label section or create new one
                label_section = label_section_map.get(label)

                if label_section is None:
                    label_section = label_class()
                    label_section.x_lobster_pair_label = label
                    section.m_add_sub_section(label_attr, label_section)

                # Get the orbital pairs and values for this label
                orbital_pairs = orb_pairs_by_label[label]
                label_index = label_to_index[label]
                coxp_values_list = coxp_orb[label_index]
                icoxp_values_list = icoxp_orb[label_index]

                # Check if orbital pairs already exist (created by parse_ICOXPLIST)
                existing_orbital_pairs = label_section.x_lobster_orbital_pairs or []

                # Create or update subsections for each orbital pair
                for pair_idx, (atom1_orb, atom2_orb) in enumerate(orbital_pairs):
                    # Check if this orbital pair already exists
                    if pair_idx < len(existing_orbital_pairs):
                        # Reuse existing orbital pair subsection
                        pair_section = existing_orbital_pairs[pair_idx]
                    else:
                        # Create new orbital pair subsection
                        pair_section = orbital_pair_class()
                        pair_section.x_lobster_atom1_orbital = atom1_orb
                        pair_section.x_lobster_atom2_orbital = atom2_orb
                        label_section.m_add_sub_section('x_lobster_orbital_pairs', pair_section)

                    # Set the values - reshape for proper dimensions
                    # coxp_values_list[pair_idx] is either:
                    # - Non-spin-polarized: 1D array of [n_energy]
                    # - Spin-polarized: list of 2 arrays [up_array, dn_array]
                    coxp_val = coxp_values_list[pair_idx]
                    icoxp_val = icoxp_values_list[pair_idx]

                    # Convert to proper shape: [n_spin, n_energy]
                    # Check if spin-polarized by checking if it's a list/tuple with array-like elements
                    if isinstance(coxp_val, (list, tuple)) and len(coxp_val) == 2 and hasattr(coxp_val[0], '__len__'):
                        # Spin-polarized: [up_array, dn_array] -> already correct structure
                        coxp_array = np.array(coxp_val)
                        icoxp_array = np.array(icoxp_val)
                    else:
                        # Non-spin-polarized: array -> wrap in list to get [1, n_energy]
                        coxp_array = np.array([coxp_val])
                        icoxp_array = np.array([icoxp_val])

                    value_attr = 'x_lobster_co{}_orbital_values'.format(method)
                    ivalue_attr = 'x_lobster_integrated_co{}_orbital_values'.format(method)
                    setattr(pair_section, value_attr, coxp_array)
                    setattr(pair_section, ivalue_attr, icoxp_array)
    coxpcar_parser.close()


def parse_CHARGE(fname, scc):
    charge_parser = TextParser(
        quantities=[
            Quantity(
                'charges',
                r' *\d+ +[A-Za-z]{1,2} +([-\d\.]+) +([-\d\.]+) *',
                repeats=True,
            )
        ]
    )

    if not os.path.isfile(fname):
        return
    charge_parser.mainfile = fname
    charge_parser.parse()

    charges = charge_parser.get('charges')
    if charges is not None:
        sec_charges = Charges()
        scc.charges.append(sec_charges)
        sec_charges.analysis_method = 'mulliken'
        sec_charges.kind = 'integrated'
        sec_charges.value = np.array(list(zip(*charges))[0]) * units.elementary_charge
        sec_charges = Charges()
        scc.charges.append(sec_charges)
        sec_charges.analysis_method = 'loewdin'
        sec_charges.kind = 'integrated'
        sec_charges.value = np.array(list(zip(*charges))[1]) * units.elementary_charge
    charge_parser.close()


def parse_DOSCAR(fname, run, logger):
    def parse_species(run, atomic_numbers):
        """
        If we don't have any structure from the underlying DFT code, we can
        at least figure out what atoms we have in the structure. The best place
        to get this info from is the DOSCAR.lobster
        """

        if not run.system:
            system = System()
            run.system.append(system)
            system.atoms = Atoms(species=atomic_numbers, periodic=[True, True, True])

    def translate_lm(lm):
        lm_dictionary = {
            's': [0, 0],
            'p_z': [1, 0],
            'p_x': [1, 1],
            'p_y': [1, 2],
            'd_z^2': [2, 0],
            'd_xz': [2, 1],
            'd_yz': [2, 2],
            'd_xy': [2, 3],
            'd_x^2-y^2': [2, 4],
            'z^3': [3, 0],
            'xz^2': [3, 1],
            'yz^2': [3, 2],
            'xyz': [3, 3],
            'z(x^2-y^2)': [3, 4],
            'x(x^2-3y^2)': [3, 5],
            'y(3x^2-y^2)': [3, 6],
        }
        return lm_dictionary.get(lm[1:])

    if not os.path.isfile(fname):
        return

    energies = []
    dos_values = []
    integral_dos = []
    atom_projected_dos_values = []
    atom_index = 0
    n_atoms = 0
    n_dos = 0
    atomic_numbers = []
    lms = []
    with open(fname, 'rb') as f:
        compression, open_compressed = _compressions.get(f.read(3), (None, open))

    with open_compressed(fname) as f:
        for i, line in enumerate(f):
            if i == 0:
                n_atoms = int(line.split()[0])
            if i == 1:
                _ = float(line.split()[0]) * units.angstrom**3
            if i == 5:
                n_dos = int(line.split()[2])
            if compression:
                try:
                    line = line.decode('utf-8')
                except Exception:
                    break

            if 'Z=' in line:
                atom_index += 1
                atom_projected_dos_values.append([])
                lms.append((line.split(';')[-1]).split())
                atomic_numbers.append(int(line.split(';')[-2].split('=')[1]))
                continue
            if i > 5:
                line = [float(x) for x in line.split()]
                if atom_index == 0:
                    energies.append(line[0])
                    if len(line) == 3:
                        dos_values.append([line[1]])
                        integral_dos.append([line[2]])
                    elif len(line) == 5:
                        dos_values.append([line[1], line[2]])
                        integral_dos.append([line[3], line[4]])
                else:
                    atom_projected_dos_values[-1].append(line[1:])

    if len(atomic_numbers) > 0 and len(atomic_numbers) == n_atoms:
        parse_species(run, atomic_numbers)

    if n_dos == 0:
        return

    if len(dos_values) == n_dos:
        value = np.asarray(dos_values).T
        n_spin_channels = len(value)
        n_electrons = sum(atomic_numbers)
        index = (np.abs(energies)).argmin()
        # integrated dos at the Fermi level should be the number of electrons
        n_valence_electrons = int(round(sum(integral_dos[index])))
        n_core_electrons = n_electrons - n_valence_electrons
        value_integrated = np.asarray(integral_dos).T + n_core_electrons / len(
            integral_dos[0]
        )
        for spin_i in range(n_spin_channels):
            dos = Dos()
            run.calculation[0].dos_electronic.append(dos)
            dos.n_energies = n_dos
            dos.energies = energies * units.eV
            dos.spin_channel = spin_i if n_spin_channels == 2 else None
            dos_total = DosValues()
            dos.total.append(dos_total)
            dos_total.value = value[spin_i] * (1 / units.eV)
            dos_total.value_integrated = value_integrated[spin_i]
    else:
        logger.warning(
            "Unable to parse total dos from DOSCAR.lobster, \
                            it doesn't contain enough dos values"
        )
        return

    for atom_i, pdos in enumerate(atom_projected_dos_values):
        if len(pdos) != n_dos:
            logger.warning(
                "Unable to parse atom lm-projected dos from DOSCAR.lobster, \
                            it doesn't contain enough dos values"
            )
            continue

        if len(lms[atom_i]) == len(pdos[0]):
            # we have the same lm-projections for spin up and dn
            dos_values = np.array([[lmdos] for lmdos in zip(*pdos)]) / eV
        elif len(lms[atom_i]) * 2 == len(pdos[0]):
            transposed_pdos = np.asarray(pdos).T
            pdos_up = transposed_pdos[0::2]
            pdos_dn = transposed_pdos[1::2]
            dos_values = np.array([[a, b] for a, b in zip(pdos_up, pdos_dn)]) / eV
        else:
            logger.warning('Unexpected number of columns in DOSCAR.lobster')
            return
        for lm_i, lm in enumerate(lms[atom_i]):
            for spin_i in range(len(dos_values[lm_i])):
                dos = run.calculation[0].dos_electronic[spin_i]
                section_pdos = DosValues()
                dos.atom_projected.append(section_pdos)
                section_pdos.atom_index = atom_i
                section_pdos.m_kind = 'real_orbital'
                section_pdos.lm = translate_lm(lm)
                section_pdos.value = dos_values[lm_i][spin_i]


mainfile_parser = TextParser(
    quantities=[
        Quantity('program_version', r'^LOBSTER *v([\d\.]+) *', repeats=False),
        Quantity(
            'datetime',
            r'starting on host \S* on (\d{4}-\d\d-\d\d\sat\s\d\d:\d\d:\d\d)\s[A-Z]{3,4}',
            repeats=False,
        ),
        Quantity(
            'x_lobster_code',
            r'detecting used PAW program... (.*)',
            repeats=False,
            flatten=False,
        ),
        Quantity(
            'x_lobster_basis',
            r'setting up local basis functions\.\.\.\s*(?:WARNING.*\s*)*\s*((?:[a-zA-Z]{1,2}\s+\(.+\)(?:\s+\d\S+)+\s+)+)',
            repeats=False,
            sub_parser=TextParser(
                quantities=[
                    Quantity(
                        'x_lobster_basis_species',
                        r'([a-zA-Z]+){1,2}\s+\(([^)]+)\)((?:\s+\d\S+)+)\s+',
                        repeats=True,
                    )
                ]
            ),
        ),
        Quantity(
            'spilling',
            r'((?:spillings|abs. )[\s\S]*?charge\s*spilling:\s*\d+\.\d+%)',
            repeats=True,
            sub_parser=TextParser(
                quantities=[
                    Quantity(
                        'abs_total_spilling',
                        r'abs.\s*total\s*spilling:\s*(\d+\.\d+)%',
                        repeats=False,
                    ),
                    Quantity(
                        'abs_charge_spilling',
                        r'abs.\s*charge\s*spilling:\s*(\d+\.\d+)%',
                        repeats=False,
                    ),
                ]
            ),
        ),
        Quantity('finished', r'finished in (\d)', repeats=False),
    ]
)


class LobsterParser:
    def __init__(self):
        self._child_archives = {}

    def get_files(self, pattern, filepath):
        for _ in range(10):
            filenames = glob(f'{os.path.dirname(filepath)}/{pattern}')
            pattern = os.path.join('**', pattern)
            if filenames:
                break

        if len(filenames) > 1:
            # filter files that match
            suffix = os.path.basename(filepath).strip('lobsterout')
            matches = [f for f in filenames if suffix in f]
            filenames = matches if matches else filenames

        filenames = [f for f in filenames if os.access(f, os.F_OK)]
        return filenames

    def get_mainfile_keys(self, **kwargs):
        filepath = kwargs.get('filename', '')
        for filetype in ['OUTCAR', 'vasp']:
            files = self.get_files(f'{filetype}*', filepath)
            if files:
                return ['workflow']
        return True

    @staticmethod
    def capitalize_positions(string, positions):
        """
        Capitalizes the letters in a string at the specified positions.

        Args:
            string (str): The input string.
            positions (list of int): List of positions (0-indexed) to capitalize.

        Returns:
            str: The modified string with specified positions capitalized.
        """
        # Convert the string to a list to make it mutable
        char_list = list(string)

        # Loop through the positions and capitalize them if within bounds
        for pos in positions:
            if 0 <= pos < len(char_list):
                char_list[pos] = char_list[pos].upper()

        # Join the list back into a string
        return ''.join(char_list)

    @staticmethod
    def get_basis_function_dict(lobster_basis_species: list):
        """
        Returns a dict with specie symbol as key and projection basis functions as values.

        Args:
            lobster_basis_species (list): The input list.

        Returns:
            dict: The dictionary with per element projection basis functions.
        """
        species_basis_function = {}
        for specie_basis in lobster_basis_species:
            species_basis_function[specie_basis[0]] = specie_basis[2:]

        return species_basis_function

    def parse(self, mainfile: str, archive: EntryArchive, logger=None):
        mainfile_parser.mainfile = mainfile
        mainfile_path = os.path.dirname(mainfile)
        mainfile_parser.parse()

        run = Run()
        archive.run.append(run)

        run.program = Program(
            name='LOBSTER', version=str(mainfile_parser.get('program_version'))
        )
        # FIXME: There is a timezone info present as well, but datetime support for timezones
        # is bad and it doesn't support some timezones (for example CEST).
        # That leads to test failures, so ignore it for now.
        date = datetime.datetime.strptime(
            ' '.join(mainfile_parser.get('datetime')), '%Y-%m-%d at %H:%M:%S'
        ) - datetime.datetime(1970, 1, 1)
        run.time_run = TimeRun(wall_start=date.total_seconds())
        code = mainfile_parser.get('x_lobster_code')

        # parse structure
        structure = None
        if code is not None:
            if code == 'VASP':
                try:
                    contcar_path = get_lobster_file(
                        os.path.join(mainfile_path, 'CONTCAR')
                    )
                    structure = ase.io.read(contcar_path, format='vasp')
                except FileNotFoundError:
                    logger.warning(
                        'Unable to parse structure info, no CONTCAR detected'
                    )
            elif code == 'Quantum Espresso':
                for file in os.listdir(mainfile_path):
                    # lobster requires the QE input to have *.scf.in suffix
                    if file.endswith('.scf.in'):
                        qe_input_file = os.path.join(mainfile_path, file)
                        try:
                            structure = ase.io.read(qe_input_file, format='espresso-in')
                        except Exception:
                            pass
                if structure is None:
                    logger.warning(
                        'Unable to parse structure info, no Quantum Espresso input'
                        'detected.'
                    )
            else:
                logger.warning('Parsing of {} structure is not supported'.format(code))

        if isinstance(structure, ase.Atoms):
            system = System()
            run.system.append(system)
            system.atoms = Atoms(
                lattice_vectors=structure.get_cell() * units.angstrom,
                labels=structure.get_chemical_symbols(),
                periodic=structure.get_pbc(),
                positions=structure.get_positions() * units.angstrom,
            )

        if mainfile_parser.get('finished') is not None:
            run.clean_end = True
        else:
            run.clean_end = False

        scc = Calculation()
        run.calculation.append(scc)
        method = Method()
        run.method.append(method)
        scc.method_ref = method

        spilling = mainfile_parser.get('spilling')
        if spilling is not None:
            method.electronic = Electronic(n_spin_channels=len(spilling))
            total_spilling = []
            charge_spilling = []
            for s in spilling:
                total_spilling.append(s.get('abs_total_spilling'))
                charge_spilling.append(s.get('abs_charge_spilling'))
            if total_spilling[0] is not None:
                scc.x_lobster_abs_total_spilling = np.array(total_spilling)
            scc.x_lobster_abs_charge_spilling = np.array(charge_spilling)

        method.x_lobster_code = code

        if (basis := mainfile_parser.get('x_lobster_basis')) is not None:
            if (species := basis.get('x_lobster_basis_species')) is not None:
                # store projection basis used for calc (useful for filtering)
                method.x_lobster_basis_functions = self.get_basis_function_dict(
                    lobster_basis_species=species
                )

                basis_used = species[0][1]
                # checks necessary as LOBSTER 5.1.1 writes basis names now in lower case
                if basis_used == 'pbevaspfit2015':
                    basis_used = self.capitalize_positions(
                        string=species[0][1], positions=[3, 7]
                    )
                elif basis_used in ['bunge', 'koga']:
                    basis_used = self.capitalize_positions(
                        string=species[0][1], positions=[0]
                    )
                method.electrons_representation = [
                    BasisSetContainer(
                        type='atom-centered orbitals',  # https://pubs.acs.org/doi/pdf/10.1021/j100135a014
                        scope=[
                            'wavefunction'
                        ],  # https://pubs.acs.org/doi/pdf/10.1021/jp202489s
                        basis_set=[
                            BasisSet(
                                type=basis_used,
                                # https://www.nature.com/articles/s41524-019-0208-x
                                scope=['full-electron'],
                            )
                        ],
                    )
                ]

        parse_ICOXPLIST(
            get_lobster_file(os.path.join(mainfile_path, 'ICOHPLIST.lobster')),
            scc,
            'hp',
            version=run.program.version,
        )
        parse_ICOXPLIST(
            get_lobster_file(os.path.join(mainfile_path, 'ICOOPLIST.lobster')),
            scc,
            'op',
            version=run.program.version,
        )
        parse_ICOXPLIST(
            get_lobster_file(os.path.join(mainfile_path, 'ICOBILIST.lobster')),
            scc,
            'bi',
            version=run.program.version,
        )
        parse_COXPCAR(
            get_lobster_file(os.path.join(mainfile_path, 'COHPCAR.lobster')),
            scc,
            'hp',
            logger,
        )
        parse_COXPCAR(
            get_lobster_file(os.path.join(mainfile_path, 'COOPCAR.lobster')),
            scc,
            'op',
            logger,
        )
        parse_COXPCAR(
            get_lobster_file(os.path.join(mainfile_path, 'COBICAR.lobster')),
            scc,
            'bi',
            logger,
        )
        parse_CHARGE(
            get_lobster_file(os.path.join(mainfile_path, 'CHARGE.lobster')), scc
        )
        doscar_lso = get_lobster_file(
            os.path.join(mainfile_path, 'DOSCAR.LSO.lobster')
        )
        doscar = get_lobster_file(os.path.join(mainfile_path, 'DOSCAR.lobster'))
        if os.path.isfile(doscar_lso):
            if os.path.isfile(doscar):
                logger.warning(
                    'Both DOSCAR.LSO.lobster and DOSCAR.lobster found; '
                    'parsing only DOSCAR.LSO.lobster to avoid duplicate DOS.'
                )
            parse_DOSCAR(doscar_lso, run, logger)
        elif os.path.isfile(doscar):
            parse_DOSCAR(doscar, run, logger)

        workflow = SinglePoint()
        archive.workflow2 = workflow

        if run.system:
            scc.system_ref = run.system[0]

        if self._child_archives:
            # link vasp entries to lobster in a generic workflow
            workflow_archive = self._child_archives.get('workflow')
            workflow_archive.workflow2 = SerialSimulation(name='LOBSTER Workflow')

            try:
                logger.info(
                    f'Underlying VASP calculation detected. Attempting to link VASP and LOBSTER entries.'
                )
                from nomad.search import search  # noqa
                from nomad.app.v1.models import MetadataRequired  # noqa

                parent_file = mainfile.split('raw/')[-1]  # noqa

                parent_dir = os.path.dirname(parent_file)

                upload_id = archive.metadata.upload_id
                metadata = search(
                    owner='visible',
                    user_id=archive.metadata.main_author.user_id,
                    query={'upload_id': upload_id},
                    required=MetadataRequired(
                        include=['entry_id', 'mainfile', 'parser_name']
                    ),
                ).data
                for result in metadata:
                    # skip non-vasp files
                    if 'vasp' not in result.get('parser_name', '').lower():
                        continue
                    entry_id = result.get('entry_id')
                    if not entry_id:
                        continue
                    entry_mainfile = result.get('mainfile')
                    # link only entries in the same directory or sub-directories
                    if entry_mainfile.startswith(parent_dir):
                        entry_archive = archive.m_context.load_archive(
                            entry_id, upload_id, None
                        )
                        # add DFT run to workflow tasks
                        dft_task = TaskReference(task=entry_archive.workflow2)

                        # Extract DFT Inputs and Outputs
                        input_structure = extract_section(
                            entry_archive, ['run', 'system']
                        )
                        dft_calculation = extract_section(
                            entry_archive, ['run', 'calculation']
                        )

                        dft_task.name = 'DFT run'
                        dft_task.inputs = [
                            Link(section=input_structure, name='Input Structure')
                        ]
                        dft_task.outputs = [
                            Link(section=dft_calculation, name='Output DFT calculation')
                        ]

                        # Set the DFT task as an input for the workflow
                        workflow_archive.workflow2.inputs = [
                            Link(section=input_structure, name='Structure')
                        ]

                        # add DFT task to the workflow tasks
                        workflow_archive.workflow2.tasks.append(dft_task)
            except Exception:
                logger.warning(f'Error setting workflow inputs, i.e., VASP entries.')

            # add lobster archive to the workflow tasks
            lobster_calculation = extract_section(archive, ['run', 'calculation'])
            lobster_task = TaskReference(task=archive.workflow2, name='LOBSTER run')

            try:
                lobster_task.inputs = [
                    Link(
                        section=dft_task.outputs[0].section,
                        name='Structure and PlaneWavefunctions',
                    )
                ]
            except UnboundLocalError:
                logger.warning(f'Error connecting VASP with LOBSTER entry.')

            lobster_task.outputs = [
                Link(section=lobster_calculation, name='Output LOBSTER calculation')
            ]

            workflow_archive.workflow2.tasks.append(lobster_task)

            # Set workflow outputs
            workflow_archive.workflow2.outputs = [
                Link(section=lobster_task.outputs[0].section, name='LOBSTER Outputs')
            ]

        mainfile_parser.close()
