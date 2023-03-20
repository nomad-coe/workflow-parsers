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
from nomad.datamodel.metainfo.simulation.run import Run, Program, TimeRun
from nomad.datamodel.metainfo.simulation.system import (
    System, Atoms)
from nomad.datamodel.metainfo.simulation.method import (
    Method, Electronic, BasisSet)
from nomad.datamodel.metainfo.simulation.calculation import (
    Calculation, Dos, DosValues, Charges)

from nomad.parsing.file_parser import TextParser, Quantity

from .metainfo.lobster import x_lobster_section_cohp, x_lobster_section_coop

'''
This is a LOBSTER code parser.
'''

e = (1 * units.e).to_base_units().magnitude
eV = (1 * units.eV).to_base_units().magnitude


def parse_ICOXPLIST(fname, scc, method):

    def icoxp_line_split(string):
        tmp = string.split()
        # LOBSTER version 3 and above
        if len(tmp) == 8:
            return [tmp[1], tmp[2], float(tmp[3]), [int(tmp[4]),
                    int(tmp[5]), int(tmp[6])], float(tmp[7])]
        # LOBSTER versions below 3
        elif len(tmp) == 6:
            return [tmp[1], tmp[2], float(tmp[3]), float(tmp[4]), int(tmp[5])]

    icoxplist_parser = TextParser(quantities=[
        Quantity('icoxpslist_for_spin', r'\s*CO[OH]P.*spin\s*\d\s*([^#]+[-\d\.]+)',
                 repeats=True,
                 sub_parser=TextParser(quantities=[
                     Quantity('line',
                              # LOBSTER version 3 and above
                              r'(\s*\d+\s+\w+\s+\w+\s+[\.\d]+\s+[-\d]+\s+[-\d]+\s+[-\d]+\s+[-\.\d]+\s*)|'
                              # LOBSTER versions below 3
                              r'(\s*\d+\s+\w+\s+\w+\s+[\.\d]+\s+[-\.\d]+\s+[\d]+\s*)',
                              repeats=True, str_operation=icoxp_line_split)])
                 )
    ])

    if not os.path.isfile(fname):
        return
    icoxplist_parser.mainfile = fname
    icoxplist_parser.parse()

    icoxp = []
    for spin, icoxplist in enumerate(icoxplist_parser.get('icoxpslist_for_spin')):

        lines = icoxplist.get('line')
        if lines is None:
            break
        if type(lines[0][4]) is int:
            a1, a2, distances, tmp, bonds = zip(*lines)
        else:
            a1, a2, distances, v, tmp = zip(*lines)
        icoxp.append(0)
        icoxp[-1] = list(tmp)
        if spin == 0:
            if method == 'o':
                section = scc.m_create(x_lobster_section_coop)
            elif method == 'h':
                section = scc.m_create(x_lobster_section_cohp)

            setattr(section, "x_lobster_number_of_co{}p_pairs".format(
                method), len(list(a1)))
            setattr(section, "x_lobster_co{}p_atom1_labels".format(
                method), list(a1))
            setattr(section, "x_lobster_co{}p_atom2_labels".format(
                method), list(a2))
            setattr(section, "x_lobster_co{}p_distances".format(
                method), np.array(distances) * units.angstrom)

            # version specific entries
            if 'v' in locals():
                setattr(section, "x_lobster_co{}p_translations".format(
                    method), list(v))
            if 'bonds' in locals():
                setattr(section, "x_lobster_co{}p_number_of_bonds".format(
                    method), list(bonds))

    if len(icoxp) > 0:
        setattr(section, "x_lobster_integrated_co{}p_at_fermi_level".format(
            method), np.array(icoxp) * units.eV)


def parse_COXPCAR(fname, scc, method, logger):
    coxpcar_parser = TextParser(quantities=[
        Quantity('coxp_pairs', r'No\.\d+:(\w{1,2}\d+)->(\w{1,2}\d+)\(([\d\.]+)\)\s*?',
                 repeats=True),
        Quantity('coxp_lines', r'\n\s*(-*\d+\.\d+(?:[ \t]+-*\d+\.\d+)+)',
                 repeats=True)
    ])

    if not os.path.isfile(fname):
        return
    coxpcar_parser.mainfile = fname
    coxpcar_parser.parse()

    if method == 'o':
        if not scc.x_lobster_section_coop:
            section = scc.m_create(x_lobster_section_coop)
        else:
            section = scc.x_lobster_section_coop
    elif method == 'h':
        if not scc.x_lobster_section_cohp:
            section = scc.m_create(x_lobster_section_cohp)
        else:
            section = scc.x_lobster_section_cohp

    pairs = coxpcar_parser.get('coxp_pairs')
    if pairs is None:
        logger.warning('No CO{}P values detected in CO{}PCAR.lobster.'.format(
            method.upper(), method.upper()))
        return
    a1, a2, distances = zip(*pairs)
    number_of_pairs = len(list(a1))

    setattr(section, "x_lobster_number_of_co{}p_pairs".format(
        method), number_of_pairs)
    setattr(section, "x_lobster_co{}p_atom1_labels".format(
        method), list(a1))
    setattr(section, "x_lobster_co{}p_atom2_labels".format(
        method), list(a2))
    setattr(section, "x_lobster_co{}p_distances".format(
        method), np.array(distances) * units.angstrom)

    coxp_lines = coxpcar_parser.get('coxp_lines')
    if coxp_lines is None:
        logger.warning('No CO{}P values detected in CO{}PCAR.lobster.'
                       'The file is likely incomplete'.format(
                           method.upper(), method.upper()))
        return
    coxp_lines = list(zip(*coxp_lines))

    setattr(section, "x_lobster_number_of_co{}p_values".format(
        method), len(coxp_lines[0]))
    setattr(section, "x_lobster_co{}p_energies".format(
        method), np.array(coxp_lines[0]) * units.eV)

    if len(coxp_lines) == 2 * number_of_pairs + 3:
        coxp = [[x] for x in coxp_lines[3::2]]
        icoxp = [[x] for x in coxp_lines[4::2]]
        acoxp = [coxp_lines[1]]
        aicoxp = [coxp_lines[2]]
    elif len(coxp_lines) == 4 * number_of_pairs + 5:
        coxp = [x for x in zip(coxp_lines[5:number_of_pairs * 2 + 4:2],
                coxp_lines[number_of_pairs * 2 + 5: 4 * number_of_pairs + 4:2])]
        icoxp = [x for x in zip(coxp_lines[6:number_of_pairs * 2 + 5:2],
                 coxp_lines[number_of_pairs * 2 + 6: 4 * number_of_pairs + 5:2])]
        acoxp = [coxp_lines[1], coxp_lines[3]]
        aicoxp = [coxp_lines[2], coxp_lines[4]]
    else:
        logger.warning('Unexpected number of columns {} '
                       'in CO{}PCAR.lobster.'.format(len(coxp_lines),
                                                     method.upper()))
        return

    # FIXME: correct magnitude?
    setattr(section, "x_lobster_co{}p_values".format(
        method), np.array(coxp))
    setattr(section, "x_lobster_average_co{}p_values".format(
        method), np.array(acoxp))
    setattr(section, "x_lobster_integrated_co{}p_values".format(
        method), np.array(icoxp) * units.eV)
    setattr(section, "x_lobster_average_integrated_co{}p_values".format(
        method), np.array(aicoxp) * units.eV)
    setattr(section, "x_lobster_integrated_co{}p_values".format(
        method), np.array(icoxp) * units.eV)


def parse_CHARGE(fname, scc):
    charge_parser = TextParser(quantities=[
        Quantity(
            'charges', r'\s*\d+\s+[A-Za-z]{1,2}\s+([-\d\.]+)\s+([-\d\.]+)\s*', repeats=True)
    ])

    if not os.path.isfile(fname):
        return
    charge_parser.mainfile = fname
    charge_parser.parse()

    charges = charge_parser.get('charges')
    if charges is not None:
        sec_charges = scc.m_create(Charges)
        sec_charges.analysis_method = "mulliken"
        sec_charges.kind = "integrated"
        sec_charges.value = np.array(list(zip(*charges))[0]) * units.elementary_charge
        sec_charges = scc.m_create(Charges)
        sec_charges.analysis_method = "loewdin"
        sec_charges.kind = "integrated"
        sec_charges.value = np.array(list(zip(*charges))[1]) * units.elementary_charge


def parse_DOSCAR(fname, run, logger):

    def parse_species(run, atomic_numbers):
        """
        If we don't have any structure from the underlying DFT code, we can
        at least figure out what atoms we have in the structure. The best place
        to get this info from is the DOSCAR.lobster
        """

        if not run.system:
            system = run.m_create(System)
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
    with open(fname) as f:
        for i, line in enumerate(f):
            if i == 0:
                n_atoms = int(line.split()[0])
            if i == 1:
                _ = float(line.split()[0]) * units.angstrom**3
            if i == 5:
                n_dos = int(line.split()[2])
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
        dos = run.calculation[0].m_create(Dos, Calculation.dos_electronic)
        dos.n_energies = n_dos
        dos.energies = energies * units.eV
        value = list(zip(*dos_values))
        n_electrons = sum(atomic_numbers)
        index = (np.abs(energies)).argmin()
        # integrated dos at the Fermi level should be the number of electrons
        n_valence_electrons = int(round(sum(integral_dos[index])))
        n_core_electrons = n_electrons - n_valence_electrons
        value_integrated = np.array(list(zip(*integral_dos))) + n_core_electrons / len(integral_dos[0])
        for spin_i in range(len(value)):
            dos_total = dos.m_create(DosValues, Dos.total)
            dos_total.spin = spin_i
            dos_total.value = value[spin_i] * (1 / units.eV)
            dos_total.value_integrated = value_integrated[spin_i]
    else:
        logger.warning('Unable to parse total dos from DOSCAR.lobster, \
                            it doesn\'t contain enough dos values')
        return

    for atom_i, pdos in enumerate(atom_projected_dos_values):
        if len(pdos) != n_dos:
            logger.warning('Unable to parse atom lm-projected dos from DOSCAR.lobster, \
                            it doesn\'t contain enough dos values')
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
                section_pdos = dos.m_create(DosValues, Dos.atom_projected)
                section_pdos.atom_index = atom_i
                section_pdos.spin = spin_i
                section_pdos.m_kind = 'real_orbital'
                section_pdos.lm = translate_lm(lm)
                section_pdos.value = dos_values[lm_i][spin_i]


mainfile_parser = TextParser(quantities=[
    Quantity('program_version', r'^LOBSTER\s*v([\d\.]+)\s*', repeats=False),
    Quantity('datetime', r'starting on host \S* on (\d{4}-\d\d-\d\d\sat\s\d\d:\d\d:\d\d)\s[A-Z]{3,4}',
             repeats=False),
    Quantity('x_lobster_code',
             r'detecting used PAW program... (.*)', repeats=False, flatten=False),
    Quantity('x_lobster_basis',
             r'setting up local basis functions...\s*(?:WARNING.*\s*)*\s*((?:[a-zA-Z]{1,2}\s+\(.+\)(?:\s+\d\S+)+\s+)+)',
             repeats=False,
             sub_parser=TextParser(quantities=[
                 Quantity('x_lobster_basis_species',
                          r'([a-zA-Z]+){1,2}\s+\((.+)\)((?:\s+\d\S+)+)\s+', repeats=True)
             ])),
    Quantity('spilling', r'((?:spillings|abs. tot)[\s\S]*?charge\s*spilling:\s*\d+\.\d+%)',
             repeats=True,
             sub_parser=TextParser(quantities=[
                 Quantity('abs_total_spilling',
                          r'abs.\s*total\s*spilling:\s*(\d+\.\d+)%', repeats=False),
                 Quantity('abs_charge_spilling',
                          r'abs.\s*charge\s*spilling:\s*(\d+\.\d+)%', repeats=False)
             ])),
    Quantity('finished', r'finished in (\d)', repeats=False),
])


class LobsterParser:
    def __init__(self):
        pass

    def parse(self, mainfile: str, archive: EntryArchive, logger=None):
        mainfile_parser.mainfile = mainfile
        mainfile_path = os.path.dirname(mainfile)
        mainfile_parser.parse()

        run = archive.m_create(Run)

        run.program = Program(
            name='LOBSTER',
            version=str(mainfile_parser.get('program_version')))
        # FIXME: There is a timezone info present as well, but datetime support for timezones
        # is bad and it doesn't support some timezones (for example CEST).
        # That leads to test failures, so ignore it for now.
        date = datetime.datetime.strptime(' '.join(mainfile_parser.get('datetime')),
                                          '%Y-%m-%d at %H:%M:%S') - datetime.datetime(1970, 1, 1)
        run.time_run = TimeRun(wall_start=date.total_seconds())
        code = mainfile_parser.get('x_lobster_code')

        # parse structure
        if code is not None:
            if code == 'VASP':
                try:
                    contcar_path = os.path.join(mainfile_path, 'CONTCAR')
                    structure = ase.io.read(contcar_path, format="vasp")
                except FileNotFoundError:
                    logger.warning('Unable to parse structure info, no CONTCAR detected')
            if code == 'Quantum Espresso':
                for file in os.listdir(mainfile_path):
                    # lobster requires the QE input to have *.scf.in suffix
                    if file.endswith(".scf.in"):
                        qe_input_file = os.path.join(mainfile_path, file)
                        structure = ase.io.read(qe_input_file, format="espresso-in")
                if 'structure' not in locals():
                    logger.warning('Unable to parse structure info, no Quantum Espresso input detected')
            else:
                logger.warning('Parsing of {} structure is not supported'.format(code))

        if 'structure' in locals():
            system = run.m_create(System)
            system.atoms = Atoms(
                lattice_vectors=structure.get_cell() * units.angstrom,
                labels=structure.get_chemical_symbols(),
                periodic=structure.get_pbc(),
                positions=structure.get_positions() * units.angstrom)

        if mainfile_parser.get('finished') is not None:
            run.clean_end = True
        else:
            run.clean_end = False

        scc = run.m_create(Calculation)
        method = run.m_create(Method)
        scc.method_ref = method

        spilling = mainfile_parser.get('spilling')
        if spilling is not None:
            method.electronic = Electronic(n_spin_channels=len(spilling))
            total_spilling = []
            charge_spilling = []
            for s in spilling:
                total_spilling.append(s.get('abs_total_spilling'))
                charge_spilling.append(s.get('abs_charge_spilling'))
            scc.x_lobster_abs_total_spilling = np.array(total_spilling)
            scc.x_lobster_abs_charge_spilling = np.array(charge_spilling)

        method.x_lobster_code = code

        basis = mainfile_parser.get('x_lobster_basis')
        if basis is not None:
            species = basis.get('x_lobster_basis_species')
            if species is not None:
                method.basis_set.append(BasisSet(name=species[0][1]))

        parse_ICOXPLIST(mainfile_path + '/ICOHPLIST.lobster', scc, 'h')
        parse_ICOXPLIST(mainfile_path + '/ICOOPLIST.lobster', scc, 'o')

        parse_COXPCAR(mainfile_path + '/COHPCAR.lobster', scc, 'h', logger)
        parse_COXPCAR(mainfile_path + '/COOPCAR.lobster', scc, 'o', logger)

        parse_CHARGE(mainfile_path + '/CHARGE.lobster', scc)

        parse_DOSCAR(mainfile_path + '/DOSCAR.lobster', run, logger)

        if run.system:
            scc.system_ref = run.system[0]
