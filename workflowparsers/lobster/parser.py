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

from nomad.datamodel import EntryArchive
from nomad.units import ureg as units
from nomad.parsing.parsers import _compressions
from runschema.run import Run, Program, TimeRun
from runschema.system import System, Atoms
from runschema.method import (
    Method,
    Electronic,
    BasisSet,
    BasisSetContainer,
)
from runschema.calculation import Calculation, Dos, DosValues, Charges

from nomad.parsing.file_parser import TextParser, Quantity

from .metainfo.lobster import x_lobster_section_cohp, x_lobster_section_coop, x_lobster_section_cobi

"""
This is a LOBSTER code parser.
"""

e = (1 * units.e).to_base_units().magnitude
eV = (1 * units.eV).to_base_units().magnitude


def get_lobster_file(filename):
    compressions = [''] + [v[0] for v in _compressions.values()]
    for compression in compressions:
        name = f'{filename}.{compression}'
        if os.path.isfile(name):
            return name
    return filename


def parse_ICOXPLIST(fname, scc, method, version):

    def icoxp_line_split(string):
        tmp = string.split()
        # LOBSTER version 3 and above
        if len(tmp) == 8:
            return [
                tmp[1],
                tmp[2],
                float(tmp[3]),
                [int(tmp[4]), int(tmp[5]), int(tmp[6])],
                float(tmp[7]),
            ]
        elif len(tmp) == 9:
            return [
                tmp[1],
                tmp[2],
                float(tmp[3]),
                [int(tmp[4]), int(tmp[5]), int(tmp[6])],
                float(tmp[7]), float(tmp[8]),
            ]
        # LOBSTER versions below 3
        elif len(tmp) == 6 and tmp[-2] != "spin":
            return [tmp[1], tmp[2], float(tmp[3]), float(tmp[4]), int(tmp[5])]

    float_version = float(version.split(".")[0]+"."+version.split(".")[1])
    if 5 > float_version > 2:
        icoxplist_parser = TextParser(
            quantities=[
                Quantity(
                    "icoxpslist_for_spin",
                    r"\s*(CO[O,H,B,I,P]).*spin\s*\d\s*([^#]+[-\d\.]+)",
                    repeats=True,
                    sub_parser=TextParser(
                        quantities=[
                            Quantity(
                                "line",
                                # LOBSTER version 3 and above
                                r"(\s*\d+\s+\w+\s+\w+\s+[\.\d]+\s+[-\d]+\s+[-\d]+\s+[-\d]+\s+[-\.\d]+\s*)",
                                repeats=True,
                                str_operation=icoxp_line_split,
                            )
                        ]
                    ),
                )
            ]
        )
    elif float_version >=5:
        icoxplist_parser = TextParser(
            quantities=[
                Quantity(
                    "icoxpslist_for_spin",
                    r"\s*(CO[O,H,B,I,P]).*\n\s.*([^#]+[-\d\.]+)",
                    repeats=True,
                    sub_parser=TextParser(
                        quantities=[
                            Quantity(
                                "line",
                                # LOBSTER version 5.1 and above
                                r"(\s*\d+\s+\w+\s+\w+\s+[\.\d]+\s+[-\d]+\s+[-\d]+\s+[-\d]+\s+[-\.\d]+\s*)",
                                repeats=True,
                                str_operation=icoxp_line_split,
                            )
                        ]
                    ),
                )
            ]
        )
    else:
        icoxplist_parser = TextParser(
            quantities=[
                Quantity(
                    "icoxpslist_for_spin",
                    r"\s*(CO[OH]P).*spin\s*\d\s*([^#]+[-\d\.]+)",
                    repeats=True,
                    sub_parser=TextParser(
                        quantities=[
                            Quantity(
                                "line",
                                # LOBSTER versions below 3
                                r"(\s*\d+\s+\w+\s+\w+\s+[\.\d]+\s+[-\.\d]+\s+[\d]+\s*)",
                                repeats=True,
                                str_operation=icoxp_line_split,
                            )
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
    for spin, icoxplist in enumerate(icoxplist_parser.get('icoxpslist_for_spin')):
        lines = icoxplist.get('line')
        if lines is None:
            break

        lines = [x for x in lines if x[0].count("_") == 0]

        if isinstance(lines[0][4], int):
            a1, a2, distances, tmp, bonds = zip(*lines)
        else:
            a1, a2, distances, v, tmp = zip(*lines)
        icoxp.append(0)
        icoxp[-1] = list(tmp)
        if spin == 0:
            if method == 'op':
                section = x_lobster_section_coop()
                scc.x_lobster_section_coop = section
            elif method == 'hp':
                section = x_lobster_section_cohp()
                scc.x_lobster_section_cohp = section
            elif method == "bi":
                section = x_lobster_section_cobi()
                scc.x_lobster_section_cobi = section

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
                setattr(section, 'x_lobster_co{}_translations'.format(method), list(v))
            if 'bonds' in locals():
                setattr(
                    section,
                    'x_lobster_co{}_number_of_bonds'.format(method),
                    list(bonds),
                )

    if len(icoxp) > 0:
        setattr(
            section,
            'x_lobster_integrated_co{}_at_fermi_level'.format(method),
            np.array(icoxp) * units.eV,
        )


def parse_COXPCAR(fname, scc, method, logger):

    def _separate_orbital_data(pairs_list, coxp_lines_list):
        """
        Separate the data for atom pairs cohps and atom orbitals cohps

        Args:
            pairs_list: list of pairs
            coxp_lines_list: list of COXP lines
        """
        atom_pair_cohp = []
        atom_orb_cohp = []
        if len(pairs_list) * 4  + 5 == len(coxp_lines_list): # spin polarized data
            pairs_list = pairs_list * 4
            for ix, (p, c) in enumerate(zip(pairs_list, coxp_lines_list[5:])):
                if "[" not in p[0]:
                    atom_pair_cohp.append(coxp_lines_list[5:][ix])
                else:
                    atom_orb_cohp.append(coxp_lines_list[5:][ix])
        elif len(pairs_list) * 2 + 3 == len(coxp_lines): # non spin polarized data
            pairs_list = pairs_list * 2
            for ix, (p, c) in enumerate(zip(pairs_list, coxp_lines_list[1:])):
                if "[" not in p[0]:
                    atom_pair_cohp.append(coxp_lines_list[3:][ix])
                else:
                    atom_orb_cohp.append(coxp_lines_list[3:][ix])
        else:
            logger.warning(
                'Unexpected number of columns {} ' 'in CO{}CAR.lobster.'.format(
                    len(coxp_lines), method.upper()
                )
            )
            return

        return atom_pair_cohp, atom_orb_cohp

    def _get_pair_label_distance(pairs_list:list[list]):
        """
        Get the atom pair label and distance seperately for
        orbital and non-orbital pairs.

        Args:
            pairs_list: nested list of pairs
        """
        atom1 = []
        atom2 = []
        distance = []
        atom1_orb = []
        atom2_orb = []
        distance_orb = []
        for at1, at2, dist in pairs_list:
            if "[" not in at1:
                atom1.append(at1)
                atom2.append(at2)
                distance.append(dist)
            else:
                atom1_orb.append(at1)
                atom2_orb.append(at2)
                distance_orb.append(dist)

        return atom1, atom2, distance, atom1_orb, atom2_orb, distance_orb


    coxpcar_parser = TextParser(
        quantities=[
            Quantity(
                "coxp_pairs",
                r"No\.\d+:(\w{1,2}\d+)->(\w{1,2}\d+)\(([\d\.]+)\)\s*?|No\.\d+:(\w{1,2}\d+\[[^\]]*\])->(\w{1,2}\d+\[[^\]]*\])\(([\d\.]+)\)\s*?",
                repeats=True,
            ),
            Quantity(
                "coxp_lines", r"\n\s*(-*\d+\.\d+(?:[ \t]+-*\d+\.\d+)+)", repeats=True
            ),
        ]
    )

    if not os.path.isfile(fname):
        return
    coxpcar_parser.mainfile = fname
    coxpcar_parser.parse()

    if method == 'op':
        if not scc.x_lobster_section_coop:
            section = x_lobster_section_coop()
            section.x_lobster_section_coop = section
        else:
            section = scc.x_lobster_section_coop
    elif method == 'hp':
        if not scc.x_lobster_section_cohp:
            section = x_lobster_section_cohp()
            scc.x_lobster_section_cohp = section
        else:
            section = scc.x_lobster_section_cohp
    elif method == "bi":
        if not scc.x_lobster_section_cobi:
            section = x_lobster_section_cobi()
            scc.x_lobster_section_cobi = section
        else:
            section = scc.x_lobster_section_cobi

    pairs = coxpcar_parser.get('coxp_pairs')

    if pairs is None:
        logger.warning(
            'No CO{}P values detected in CO{}CAR.lobster.'.format(
                method.upper(), method.upper()
            )
        )
        return

    coxp_lines = coxpcar_parser.get('coxp_lines')
    coxp_lines = list(zip(*coxp_lines))

    if coxp_lines is None:
        logger.warning(
            'No CO{} values detected in CO{}CAR.lobster.'
            'The file is likely incomplete'.format(method.upper(), method.upper())
        )
        return

    spin_polarized = len(coxp_lines) == 4 * len(pairs) + 5

    a1, a2, distances, a1_orb, a2_orb, distances_orb = _get_pair_label_distance(pairs)
    atom_pair_cohp, atom_orb_cohp = _separate_orbital_data(pairs, coxp_lines)

    number_of_pairs = len(list(a1))
    number_of_pairs_orb = len(list(a1_orb))

    if not spin_polarized:
        acoxp = [coxp_lines[1]]
        aicoxp = [coxp_lines[2]]
        coxp = [[x] for x in atom_pair_cohp[0::2]]
        coxp_orb = [[x] for x in atom_orb_cohp[0::2]]
        icoxp = [[x] for x in atom_pair_cohp[1::2]]
        icoxp_orb = [[x] for x in atom_orb_cohp[1::2]]
    elif spin_polarized:
        acoxp = [coxp_lines[1], coxp_lines[3]]
        aicoxp = [coxp_lines[2], coxp_lines[4]]
        coxp = [
            x
            for x in zip(
                atom_pair_cohp[0::2],
                atom_pair_cohp[number_of_pairs*2::2],
            )
        ]
        coxp_orb = [
            x
            for x in zip(
                atom_orb_cohp[0::2],
                atom_orb_cohp[number_of_pairs_orb*2::2],
            )
        ]
        icoxp = [
            x
            for x in zip(
                atom_pair_cohp[1::2],
                atom_pair_cohp[(number_of_pairs*2)+1::2],
            )
        ]
        icoxp_orb = [
            x
            for x in zip(
                atom_orb_cohp[1::2],
                atom_orb_cohp[(number_of_pairs_orb*2)+1::2],
            )
        ]

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

    # FIXME: correct magnitude?
    setattr(section, 'x_lobster_co{}_values'.format(method), np.array(coxp))
    setattr(section, 'x_lobster_average_co{}_values'.format(method), np.array(acoxp))
    setattr(
        section,
        'x_lobster_integrated_co{}_values'.format(method),
        np.array(icoxp) * units.eV,
    )
    setattr(
        section,
        'x_lobster_average_integrated_co{}_values'.format(method),
        np.array(aicoxp) * units.eV,
    )
    setattr(
        section,
        'x_lobster_integrated_co{}_values'.format(method),
        np.array(icoxp) * units.eV,
    )
    if number_of_pairs_orb>0:
        setattr(
            section,
            'x_lobster_number_of_co{}_orbital_pairs'.format(method),
            number_of_pairs_orb,
        )
        setattr(
            section,
            'x_lobster_co{}_atom1_orbital_labels'.format(method),
            list(a1_orb),
        )
        setattr(
            section,
            'x_lobster_co{}_atom2_orbital_labels'.format(method),
            list(a2_orb),
        )
        setattr(
            section,
            'x_lobster_co{}_orbital_values'.format(method),
            np.array(coxp_orb) * units.eV,
        )
        setattr(
            section,
            'x_lobster_integrated_co{}_orbital_values'.format(method),
            np.array(icoxp_orb) * units.eV,
        )


def parse_CHARGE(fname, scc):
    charge_parser = TextParser(
        quantities=[
            Quantity(
                'charges',
                r'\s*\d+\s+[A-Za-z]{1,2}\s+([-\d\.]+)\s+([-\d\.]+)\s*',
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
        value = list(zip(*dos_values))
        n_spin_channels = len(value)
        n_electrons = sum(atomic_numbers)
        index = (np.abs(energies)).argmin()
        # integrated dos at the Fermi level should be the number of electrons
        n_valence_electrons = int(round(sum(integral_dos[index])))
        n_core_electrons = n_electrons - n_valence_electrons
        value_integrated = np.array(list(zip(*integral_dos))) + n_core_electrons / len(
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
            pdos_up = list(zip(*pdos))[0::2]
            pdos_dn = list(zip(*pdos))[1::2]
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
        Quantity('program_version', r'^LOBSTER\s*v([\d\.]+)\s*', repeats=False),
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
                        r'([a-zA-Z]+){1,2}\s+\((.+)\)((?:\s+\d\S+)+)\s+',
                        repeats=True,
                    )
                ]
            ),
        ),
        Quantity(
            'spilling',
            r'((?:spillings|abs. tot)[\s\S]*?charge\s*spilling:\s*\d+\.\d+%)',
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
        pass

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
                        structure = ase.io.read(qe_input_file, format='espresso-in')
                if 'structure' not in locals():
                    logger.warning(
                        'Unable to parse structure info, no Quantum Espresso input detected'
                    )
            else:
                logger.warning('Parsing of {} structure is not supported'.format(code))

        if 'structure' in locals():
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
                basis_used = species[0][1]
                if basis_used == "pbevaspfit2015":
                    basis_used = self.capitalize_positions(string=species[0][1], positions=[3,7])
                elif basis_used == "bunge":
                    basis_used = self.capitalize_positions(string=species[0][1], positions=[0])
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
            get_lobster_file(mainfile_path + '/ICOHPLIST.lobster'), scc, 'hp',
            version=run.program.version
        )
        parse_ICOXPLIST(
            get_lobster_file(mainfile_path + '/ICOOPLIST.lobster'), scc, 'op',
            version=run.program.version
        )
        parse_ICOXPLIST(
            get_lobster_file(mainfile_path + '/ICOBILIST.lobster'), scc, "bi",
            version=run.program.version
        )
        parse_COXPCAR(
            get_lobster_file(mainfile_path + '/COHPCAR.lobster'), scc, 'hp', logger
        )
        parse_COXPCAR(
            get_lobster_file(mainfile_path + '/COOPCAR.lobster'), scc, 'op', logger
        )
        parse_COXPCAR(
            get_lobster_file(mainfile_path + "/COBICAR.lobster"), scc, "bi", logger
        )
        parse_CHARGE(get_lobster_file(mainfile_path + '/CHARGE.lobster'), scc)

        parse_DOSCAR(get_lobster_file(mainfile_path + '/DOSCAR.lobster'), run, logger)

        if run.system:
            scc.system_ref = run.system[0]
