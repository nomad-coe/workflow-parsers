#
# Copyright The NOMAD Authors.
#
# This file is part of NOMAD.
# See https://nomad-lab.eu for further info.
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
import os
import numpy as np
import logging
import json
import phonopy
from phonopy.units import THzToEv
from phonopy.structure.atoms import PhonopyAtoms

from .calculator import PhononProperties

from nomad import config  # type: ignore
from nomad.units import ureg
from nomad.parsing.file_parser import TextParser, Quantity
from runschema.run import Run, Program
from runschema.method import Method, Electronic
from runschema.system import System, Atoms
from runschema.calculation import (
    Calculation,
    BandStructure,
    BandEnergies,
    Dos,
    DosValues,
    Thermodynamics,
)
from simulationworkflowschema import Phonon, PhononMethod, PhononResults
from nomad.datamodel.metainfo.workflow import Link, Task
from nomad.datamodel import EntryArchive

from .metainfo import phonopy as phonopymetainfo  # pylint: disable=unused-import


def read_aims(filename):
    """Method to read FHI-aims geometry files in phonopy context."""
    cell = []
    positions = []
    fractional = []
    symbols = []
    magmoms = []
    if not os.path.isfile(filename):
        return
    with open(filename) as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = line.split()
            if len(line) == 0:
                continue
            if line[0] == 'lattice_vector':
                cell.append([float(x) for x in line[1:4]])
            elif line[0].startswith('atom'):
                fractional.append(line[0] == 'atom_frac')
                positions.append([float(x) for x in line[1:4]])
                symbols.append(line[4])
            elif line[0] == 'initial_moment':
                magmoms.append(float(line[1]))

    for n, pos in enumerate(positions):
        if fractional[n]:
            positions[n] = [
                sum([pos[j] * cell[j][i] for j in range(3)]) for i in range(3)
            ]
    if len(magmoms) == len(positions):
        return PhonopyAtoms(
            cell=cell, symbols=symbols, positions=positions, magmoms=magmoms
        )
    else:
        return PhonopyAtoms(cell=cell, symbols=symbols, positions=positions)


class Atoms_with_forces(PhonopyAtoms):
    """Hack to phonopy.atoms to maintain ASE compatibility also for forces."""

    def get_forces(self):
        return self.forces


def read_aims_output(filename):
    """Read FHI-aims output
    returns geometry with forces from last self-consistency iteration"""
    cell = []
    symbols = []
    positions = []
    forces = []
    N = 0

    with open(filename) as f:
        while True:
            line = f.readline()
            if not line:
                break
            if 'Number of atoms' in line:
                N = int(line.split()[5])
            elif '| Unit cell:' in line:
                cell = [[float(x) for x in f.readline().split()[1:4]] for _ in range(3)]
            elif 'Atomic structure:' in line or 'Updated atomic structure:' in line:
                positions = []
                symbols = []
                symbol_index = 3 if 'Atomic' in line else 4
                position_index = 4 if 'Atomic' in line else 1
                while len(positions) != N:
                    line = f.readline()
                    if 'Species' in line or 'atom ' in line:
                        line = line.split()
                        positions.append(
                            [
                                float(x)
                                for x in line[position_index : position_index + 3]
                            ]
                        )
                        symbols.append(line[symbol_index])
            elif 'Total atomic forces' in line:
                forces = [
                    [float(x) for x in f.readline().split()[2:5]] for _ in range(N)
                ]

    atoms = Atoms_with_forces(cell=cell, symbols=symbols, positions=positions)
    atoms.forces = forces

    return atoms


def read_forces_aims(reference_supercells, tolerance=1e-6, logger=None):
    """
    Collect the pre calculated forces for each of the supercells
    """

    def get_aims_output_file(directory):
        files = [f for f in os.listdir(directory) if f.endswith('.out')]
        output = None
        for f in files:
            try:
                output = read_aims_output(os.path.join(directory, f))
                break
            except Exception:
                pass
        return output, f

    def is_equal(reference, calculated):
        if len(reference) != len(calculated):
            logger.warning('Inconsistent number of atoms.')
            return False
        if (reference.get_atomic_numbers() != calculated.get_atomic_numbers()).any():
            logger.warning('Inconsistent species.')
            return False
        if (abs(reference.get_cell() - calculated.get_cell()) > tolerance).any():
            logger.warning('Inconsistent cell.')
            return False
        # get normalized positions, wrapped to the bounding cell
        ref_pos = reference.get_scaled_positions() % 1.0
        cal_pos = calculated.get_scaled_positions() % 1.0
        # resolve coordinates at the boundary
        ref_pos = np.where(ref_pos != 1.0, ref_pos, 0.0)
        cal_pos = np.where(cal_pos != 1.0, cal_pos, 0.0)
        if (abs(ref_pos - cal_pos) > tolerance).any():
            logger.warning('Inconsistent positions.')
            return False
        return True

    reference_paths, forces_sets = [], []

    n_pad = int(np.ceil(np.log10(len(reference_supercells) + 1))) + 1
    for n, reference_supercell in enumerate(reference_supercells):
        directory = 'phonopy-FHI-aims-displacement-%s' % (str(n + 1).zfill(n_pad))
        filename = os.path.join(directory, '%s.out' % directory)
        if os.path.isfile(filename):
            calculated_supercell = read_aims_output(filename)
        else:
            # try reading out files
            calculated_supercell, filename = get_aims_output_file(directory)
            filename = os.path.join(directory, filename)

        # compare if calculated cell really corresponds to supercell
        if not is_equal(reference_supercell, calculated_supercell):
            logger.error('Supercells do not match')

        forces = np.array(calculated_supercell.get_forces())
        drift_force = forces.sum(axis=0)
        for force in forces:
            force -= drift_force / forces.shape[0]
        forces_sets.append(forces)
        reference_paths.append(filename)
    return forces_sets, reference_paths


class ControlParser(TextParser):
    def __init__(self):
        super().__init__()

    def init_quantities(self):
        def str_to_nac(val_in):
            val = val_in.strip().split()
            nac = dict(file=val[0], method=val[1].lower())
            if len(val) > 2:
                nac['delta'] = [float(v) for v in val[3:6]]
            return nac

        def str_to_supercell(val_in):
            val = [int(v) for v in val_in.strip().split()]
            if len(val) == 3:
                return np.diag(val)
            else:
                return np.reshape(val, (3, 3))

        self._quantities = [
            Quantity(
                'displacement', r'\n *phonon displacement\s*([\d\.]+)', dtype=float
            ),
            Quantity(
                'symmetry_thresh',
                r'\n *phonon symmetry_thresh\s*([\d\.]+)',
                dtype=float,
            ),
            Quantity('frequency_unit', r'\n *phonon frequency_unit\s*(\S+)'),
            Quantity(
                'supercell',
                r'\n *phonon supercell\s*(.+)',
                str_operation=str_to_supercell,
            ),
            Quantity('nac', r'\n *phonon nac\s*(.+)', str_operation=str_to_nac),
        ]


def phonopy_obj_to_archive(
    phonopy_obj, references=[], archive=None, filename=None, logger=None, **kwargs
):
    """
    Executes Phonopy starting from a phonopy object and write the results on a nomad archive.
    """

    def parse_bandstructure():
        freqs, bands, bands_labels = properties.get_bandstructure()
        if freqs is None:
            return

        # convert THz to eV
        freqs = freqs * THzToEv

        # convert eV to J
        freqs = (freqs * ureg.eV).to('joules').magnitude

        sec_scc = archive.run[0].calculation[0]

        sec_k_band = BandStructure()
        sec_scc.band_structure_phonon.append(sec_k_band)

        for i in range(len(freqs)):
            sec_k_band_segment = BandEnergies()
            sec_k_band.segment.append(sec_k_band_segment)
            sec_k_band_segment.kpoints = bands[i]
            sec_k_band_segment.endpoints_labels = [
                str(label) for label in bands_labels[i]
            ]
            sec_k_band_segment.energies = [freqs[i]]

    def parse_dos():
        f, dos = properties.get_dos()

        # convert THz to eV to Joules
        f = f * THzToEv
        f = (f * ureg.eV).to('joules').magnitude

        sec_scc = archive.run[0].calculation[0]
        sec_dos = Dos()
        sec_scc.dos_phonon.append(sec_dos)
        sec_dos.energies = f
        sec_dos_values = DosValues()
        sec_dos.total.append(sec_dos_values)
        sec_dos_values.value = dos

    def parse_thermodynamical_properties():
        T, fe, _, cv = properties.get_thermodynamical_properties()

        n_atoms = len(phonopy_obj.unitcell)
        n_atoms_supercell = len(phonopy_obj.supercell)

        fe = fe / n_atoms

        # The thermodynamic properties are reported by phonopy for the base
        # system. Since the values in the metainfo are stored per the referenced
        # system, we need to multiple by the size factor between the base system
        # and the supersystem used in the calculations.
        cv = cv * (n_atoms_supercell / n_atoms)

        # convert to SI units
        fe = (fe * ureg.eV).to('joules').magnitude

        cv = (cv * ureg.eV / ureg.K).to('joules/K').magnitude

        sec_run = archive.run[0]
        sec_scc = sec_run.calculation[0]

        for n, Tn in enumerate(T):
            sec_thermo_prop = Thermodynamics()
            sec_scc.thermodynamics.append(sec_thermo_prop)
            sec_thermo_prop.temperature = Tn
            sec_thermo_prop.vibrational_free_energy_at_constant_volume = fe[n]
            sec_thermo_prop.heat_capacity_c_v = cv[n]

        # TODO create a taylor_expansion workflow?
        # sampling_method = 'taylor_expansion'
        # expansion_order = 2

    logger = logger if logger is not None else logging
    archive = archive if archive else EntryArchive()

    pbc = np.array((1, 1, 1), bool)

    unit_cell = phonopy_obj.unitcell.get_cell()
    unit_pos = phonopy_obj.unitcell.get_positions()
    unit_sym = np.array(phonopy_obj.unitcell.get_chemical_symbols())

    super_cell = phonopy_obj.supercell.get_cell()
    super_pos = phonopy_obj.supercell.get_positions()
    super_sym = np.array(phonopy_obj.supercell.get_chemical_symbols())

    unit_cell = (unit_cell * ureg.angstrom).to('meter').magnitude
    unit_pos = (unit_pos * ureg.angstrom).to('meter').magnitude

    super_cell = (super_cell * ureg.angstrom).to('meter').magnitude
    super_pos = (super_pos * ureg.angstrom).to('meter').magnitude

    try:
        displacement = np.linalg.norm(phonopy_obj.displacements[0][1:])
        displacement = displacement * ureg.angstrom
    except Exception:
        displacement = None

    supercell_matrix = phonopy_obj.supercell_matrix
    sym_tol = phonopy_obj.symmetry.tolerance

    sec_run = Run()
    archive.run.append(sec_run)
    sec_run.program = Program(name='Phonopy', version=phonopy.__version__)

    sec_system_unit = System()
    sec_run.system.append(sec_system_unit)
    sec_atoms = Atoms()
    sec_system_unit.atoms = sec_atoms
    sec_atoms.periodic = pbc
    sec_atoms.labels = unit_sym
    sec_atoms.positions = unit_pos
    sec_atoms.lattice_vectors = unit_cell

    sec_system = System()
    sec_run.system.append(sec_system)
    sec_system.sub_system_ref = sec_system_unit
    sec_system.systems_ref = [sec_system_unit]
    sec_atoms = Atoms()
    sec_system.atoms = sec_atoms
    sec_atoms.periodic = pbc
    sec_atoms.labels = super_sym
    sec_atoms.positions = super_pos
    sec_atoms.lattice_vectors = super_cell
    sec_atoms.supercell_matrix = supercell_matrix
    sec_system.x_phonopy_original_system_ref = sec_system_unit

    sec_method = Method()
    sec_run.method.append(sec_method)
    # TODO I put this so as to have a recognizable section method, but metainfo
    # should be expanded to include phonon related method parameters
    sec_method.electronic = Electronic(method='DFT')
    sec_method.x_phonopy_symprec = sym_tol
    if displacement is not None:
        sec_method.x_phonopy_displacement = displacement

    try:
        force_constants = phonopy_obj.get_force_constants()
        force_constants = (
            (force_constants * ureg.eV / ureg.angstrom**2).to('J/(m**2)').magnitude
        )
    except Exception:
        logger.error('Error producing force constants.')
        return

    sec_scc = Calculation()
    sec_run.calculation.append(sec_scc)
    sec_scc.system_ref = sec_system
    sec_scc.method_ref = sec_method
    sec_scc.hessian_matrix = force_constants

    # run Phonopy
    properties = PhononProperties(phonopy_obj, logger, **kwargs)

    parse_bandstructure()
    parse_dos()
    parse_thermodynamical_properties()

    # create workflow section
    vol = np.dot(unit_cell[0], np.cross(unit_cell[1], unit_cell[2]))
    n_imaginary = np.count_nonzero(properties.frequencies < 0)

    workflow = Phonon(method=PhononMethod(), results=PhononResults())
    workflow.method.force_calculator = phonopy_obj.calculator
    workflow.method.mesh_density = np.prod(properties.mesh) / vol
    workflow.results.n_imaginary_frequencies = n_imaginary
    if phonopy_obj.nac_params:
        workflow.method.with_non_analytic_correction = True
    inputs = []
    for path in references:
        try:
            archive = archive.m_context.resolve_archive(
                f'../upload/archive/mainfile/{path}'
            )
            section = archive.run[0].calculation[0]
        except Exception as e:
            section = None
            logger.error(
                'Could not resolve referenced calculations.', exc_info=e, path=path
            )
        if section is not None:
            inputs.append(Link(name='Input calculation', section=section))
    workflow.inputs = inputs
    workflow.outputs = [Link(name='Phonon results', section=f'/workflow2/results')]
    workflow.tasks = [
        Task(
            name='Phonon calculation', inputs=workflow.inputs, outputs=workflow.outputs
        )
    ]
    archive.workflow2 = workflow

    if filename:
        with open(filename, 'w') as f:
            json.dump(archive.m_to_dict(), f, indent=4)

    return archive


class PhonopyParser:
    level = 1

    def __init__(self, **kwargs):
        # super().__init__(
        #     name='parsers/phonopy', code_name='Phonopy', code_homepage='https://phonopy.github.io/phonopy/',
        #     mainfile_name_re=(r'(.*/phonopy-FHI-aims-displacement-0*1/control.in$)|(.*/phon.+yaml)')
        # )
        self._kwargs = kwargs
        self.control_parser = ControlParser()

    @property
    def mainfile(self):
        return self._filepath

    @mainfile.setter
    def mainfile(self, val):
        self._phonopy_obj = None
        self.references = []
        self._filepath = os.path.abspath(val)

    @property
    def phonopy_obj(self):
        if self._phonopy_obj is None:
            if 'control.in' in self.mainfile:
                self._build_phonopy_object_fhi_aims()
            elif self.mainfile.endswith('.yaml'):
                self._build_phonopy_object_yaml()
        return self._phonopy_obj

    def _build_phonopy_object_yaml(self):
        cwd = os.getcwd()
        os.chdir(os.path.dirname(self.mainfile))

        try:
            phonopy_obj = phonopy.load(self.mainfile)
        except Exception:
            self.logger.error('Error loading phonopy file.')
            phonopy_obj = None
        finally:
            os.chdir(cwd)

        self._phonopy_obj = phonopy_obj

    def _build_phonopy_object_fhi_aims(self):
        cwd = os.getcwd()
        os.chdir(os.path.dirname(os.path.dirname(self.mainfile)))
        try:
            cell_obj = read_aims('geometry.in')
            self.control_parser.mainfile = 'control.in'
            supercell_matrix = self.control_parser.get('supercell')
            displacement = self.control_parser.get('displacement', 0.001)
            sym = self.control_parser.get('symmetry_thresh', 1e-6)
            try:
                phonopy_obj = phonopy.Phonopy(
                    cell_obj, supercell_matrix, symprec=sym, calculator='fhi-aims'
                )
                phonopy_obj.generate_displacements(distance=displacement)
                supercells = phonopy_obj.get_supercells_with_displacements()
                set_of_forces, relative_paths = read_forces_aims(
                    supercells, logger=self.logger
                )
            except Exception:
                self.logger.error('Error generating phonopy object.')
                set_of_forces = []
                phonopy_obj = None
                relative_paths = []

            prep_path = self.mainfile.split('phonopy-FHI-aims-displacement-')
            # Try to resolve references as paths relative to the upload root.
            try:
                for path in relative_paths:
                    abs_path = '%s%s' % (prep_path[0], path)
                    rel_path = abs_path.split(config.fs.staging + '/')[1].split('/', 3)[
                        3
                    ]
                    self.references.append(rel_path)
            except Exception:
                self.logger.warning(
                    'Could not resolve path to a referenced calculation within the upload.'
                )

        finally:
            os.chdir(cwd)

        if set_of_forces:
            try:
                phonopy_obj.set_forces(set_of_forces)
                phonopy_obj.produce_force_constants()
            except Exception:
                self.logger.error('Error producing force constants.')
                pass

        self._phonopy_obj = phonopy_obj

    def parse(self, filepath, archive, logger, **kwargs):
        self.mainfile = os.path.abspath(filepath)
        self.archive = archive
        self.logger = logger if logger is not None else logging
        self._kwargs.update(kwargs)

        # get bandstructure configuration file
        maindir = os.path.dirname(self.mainfile)
        # TODO read band configuration from phonopy.yaml
        files = [f for f in os.listdir(maindir) if f.endswith('.conf')]
        self._kwargs.update(
            {'band_conf': os.path.join(maindir, files[0]) if files else None}
        )

        phonopy_obj = self.phonopy_obj
        if phonopy_obj is None:
            self.logger.error('Error running phonopy.')
            return

        phonopy_obj_to_archive(
            phonopy_obj, references=self.references, archive=archive, logger=logger
        )

    def after_normalization(self, archive, logger=None) -> None:
        # Overwrite the result method with method details taken from the first referenced
        # calculation. The program name and version are kept.
        self.logger = logger if logger is not None else logging
        try:
            first_referenced_calculation = archive.workflow2.results.calculations_ref[0]
            referenced_archive = first_referenced_calculation.m_root()
        except Exception:
            self.logger.warn('Error getting referenced calculation.')
            return

        new_method = referenced_archive.results.method.m_copy()
        new_method.simulation.program_name = (
            self.archive.results.method.simulation.program_name
        )
        new_method.simulation.program_version = (
            self.archive.results.method.simulation.program_version
        )
        archive.results.method = new_method
