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
import logging
import os
import numpy as np
import scipy.constants
from datetime import datetime

from nomad.units import ureg
from nomad.parsing.file_parser import TextParser, Quantity, DataTextParser
from nomad.datamodel.metainfo.simulation.run import Run, Program, TimeRun
from nomad.datamodel.metainfo.simulation.method import (
    Method, AtomParameters, Electronic, Smearing, Photon, CoreHole
)
from nomad.datamodel.metainfo.simulation.system import System, Atoms
from nomad.datamodel.metainfo.simulation.calculation import Calculation, Spectra
from nomad.datamodel.metainfo.workflow import Workflow
from nomad.datamodel.metainfo.simulation.workflow import SinglePoint as SinglePoint2

from .metainfo.quantum_espresso_xspectra import (
    x_qe_xspectra_input, x_qe_xspectra_n_parallel)


re_f = r'[-+]?\d+\.\d*(?:[Ee][-+]\d+)?'
re_n = r'[\n\r]'


class MainfileParser(TextParser):
    def init_quantities(self):
        def to_positions(block):
            labels, positions = [], []
            for line in block.strip().splitlines():
                line = line.split()
                if not line:
                    continue
                labels.append(line[1].strip(r'+|-'))
                positions.append(line[-4:-1])
            return labels, np.array(positions, np.dtype(np.float64))

        def to_val(val_in):
            val = [v.strip().lower() for v in val_in.strip().split(':')]
            val[1] = val[1] == 'true' if val[1] in ['true', 'false'] else val[1]
            return val

        self._quantities = [
            Quantity('program_version', r'Program XSpectra v.(\S+)', dtype=str),
            Quantity('start_time', r'starts on +(\w+) at (.+)', flatten=False, dtype=str),
            Quantity(
                'input',
                r'Reading input_file\s+\-+([\s\S]+?)\-{50}',
                sub_parser=TextParser(quantities=[
                    Quantity('x_qe_xspectra_calculation', r'calculation\: *(.+)', flatten=False, dtype=str),
                    Quantity(
                        'x_qe_xspectra_xepsilon',
                        rf'xepsilon +\[.+?\]\: +({re_f}) +({re_f}) +({re_f})',
                        dtype=np.dtype(np.float64)
                    ),
                    Quantity(
                        'x_qe_xspectra_xonly_plot', r'xonly_plot\: *(\S)',
                        str_operation=lambda x: x == 'T'),
                    Quantity(
                        'x_qe_xspectra_filecore',
                        r'filecore \(core-wavefunction file\): *(\S+)', dtype=str
                    ),
                    Quantity(
                        'x_qe_xspectra_main_plot_parameters',
                        rf'main plot parameters\:\s+([\s\S]+?){re_n} *{re_n}',
                        sub_parser=TextParser(quantities=[Quantity(
                            'key_val',
                            r'(\w+) *\[*.*\]*(\:) *(\S+)',
                            repeats=True, str_operation=to_val)]))
                ])
            ),
            Quantity(
                'scf',
                r'(Reading SCF save directory.+\s+\-+[\s\S]+?)\-{50}',
                sub_parser=TextParser(quantities=[
                    Quantity(
                        'x_qe_xspectra_save_directory',
                        r'Reading .*data from directory\:\s+(\S+)', dtype=str
                    ),
                    Quantity('file', r'file *(\S+?):', dtype=str, repeats=True),
                    Quantity('wafefunctions', r'wavefunctions\(.+\) * (.+)', ),
                    Quantity(
                        'n_parallel',
                        r'Parallelization info\s+\-+\s+sticks\:.+\s+'
                        r'((?:\w+ +\d+ +\d+ +\d+ +\d+ +\d+ +\d+\s+)+)',
                        str_operation=lambda x: [line.strip().split()[1:] for line in x.strip().splitlines()],
                        dtype=np.dtype(np.int32)
                    ),
                ])
            ),
            Quantity(
                'fermi_energy',
                r'Getting the Fermi energy\s+\-+([\s\S]+?)\-{50}',
                sub_parser=TextParser(quantities=[
                    Quantity('homo', rf'ehomo *\[eV\]: *{re_f}', dtype=np.float64),
                    Quantity('lumo', rf'elumo *\[eV\]: *{re_f}', dtype=np.float64),
                    Quantity('ef', rf'ef *\[eV\]: *{re_f}', dtype=np.float64),
                ])
            ),
            Quantity(
                'bravais_lattice_index',
                r'bravais\-lattice index *= *(\d+)', dtype=np.int32
            ),
            Quantity(
                'lattice_parameter', rf'lattice parameter \(alat\) *= *({re_f}) *a\.u\.',
                dtype=np.float64),
            Quantity(
                'unit_cell_volume',
                rf'unit\-cell volume *= *({re_f}) \(a\.u\.\)\^3',
                dtype=np.float64, unit=ureg.bohr ** 3
            ),
            Quantity(
                'n_atoms_cell',
                r'number of atoms/cell *= *(\d+)', dtype=np.int32
            ),
            Quantity(
                'n_atomic_types',
                r'number of atomic types *= *(\d+)', dtype=np.int32
            ),
            Quantity(
                'kinetic_energy_cutoff',
                rf'kinetic\-energy cut\-off *= *({re_f}) *Ry',
                dtype=np.float64, unit=ureg.rydberg
            ),
            Quantity(
                'charge_density_cutoff',
                rf'charge density cut\-off *= *({re_f}) *Ry',
                dtype=np.float64, unit=ureg.rydberg
            ),
            Quantity(
                'convergence_threshold',
                rf'convergence threshold *= *({re_f})', dtype=np.float64
            ),
            Quantity(
                'beta',
                rf'beta *= *({re_f})', dtype=np.float64
            ),
            Quantity(
                'exchange_correlation',
                r'Exchange\-correlation *= *(.+)', dtype=str
            ),
            Quantity(
                'n_kohn_sham_states',
                r'number of Kohn\-Sham states *= *(\d+)', dtype=np.int32
            ),
            Quantity(
                'crystal_axes',
                rf'crystal axes: \(cart\. coord\. in units of alat\)\s*'
                rf'((?:a\(\d+\) *= *\( *{re_f} *{re_f} *{re_f} *\)\s*)+)',
                str_operation=lambda x: np.array(
                    [line.strip().split()[-4:-1] for line in x.strip().splitlines()],
                    dtype=np.dtype(np.float64)
                )
            ),
            Quantity(
                'pseudopot',
                rf'(PseudoPot\. #[\s\S]+?){re_n} *{re_n}',
                repeats=True, sub_parser=TextParser(quantities=[
                    Quantity('element', r'for ([A-Z][a-z]*)'),
                    Quantity('file', r' read from file\:\s+(\S+)', dtype=str),
                    Quantity('md5_check_sum', r'MD5 check sum\: +(\S+)', dtype=str),
                    Quantity('type', r'Pseudo is (.+?),', flatten=False, dtype=str),
                    Quantity('zval', rf'Zval *= *({re_f})', dtype=np.float64),
                    Quantity('n_radial_grid_points', r'Using radial grid of *(\d+) points', dtype=np.int32),
                    Quantity('n_beta_functions', r'(\d+) beta functions', dtype=np.int32),
                    Quantity('l', r'l\(\d+\) *= *(\d+)', dtype=np.int32, repeats=True),
                    Quantity('n_q_coefficients', r'Q\(r\) pseudized with  (\d+) coefficients', dtype=np.int32),
                    Quantity(
                        'q_coefficients',
                        rf'rinner *= *((?:{re_f}\s+)+)',
                        str_operation=lambda x: x.strip().split(), dytpe=np.dtype(np.float64)
                    )
                ])
            ),
            Quantity(
                'cartesian_axes',
                rf'positions \(alat units\)\s+((?:\d+ +[A-Z][a-z]*.+{re_f} +{re_f} +{re_f} +\)\s*)+)',
                str_operation=to_positions
            ),
            Quantity('n_kpoints', r'number of k points *= *(\d+)', dtype=np.int32),
            Quantity('smearing', rf'(\S+) smearing, width \(Ry\) *= *({re_f})'),
            Quantity(
                'dense_grid',
                rf'(?:G\s+cutoff\s*=\s*({re_f})\s*\(|Dense\s*grid:)\s*(\d+)\s*'
                r'G\-vectors\)*\s*FFT\s+(?:dimensions|grid):\s*\(\s*([\d ,]+)\)',
                str_operation=lambda x: x.replace(',', ' ').split()),
            Quantity(
                'smooth_grid',
                rf'(?:G\s+cutoff\s*=\s*({re_f})\s*\(|Smooth\s*grid:)\s*(\d+)\s*'
                r'G\-vectors\s*(?:smooth grid|FFT dimensions)\s*:\s*\(\s*([\d ,]+)\)',
                str_operation=lambda x: x.replace(',', ' ').split()),
            Quantity(
                'potential_file',
                r'The potential is recalculated from file :\s+(\S+)', dtype=str
            ),
            Quantity(
                'negative rho',
                rf'negative rho \(up, down\):\s*({re_f})\s*({re_f})', dtype=np.dtype(np.float64)
            ),
            Quantity(
                'xanes',
                r'Starting XANES calculation([\s\S]+?xanes +: +)',
                sub_parser=TextParser(quantities=[
                    Quantity(
                        'algorithm',
                        r'\s*Method of calculation based on the\s*([a-zA-Z\s]*) algorithm',
                        repeats=False),
                    Quantity(
                        'step_1',
                        r'(Begin STEP 1 [\s\S]+?End STEP 1)',
                        sub_parser=TextParser(quantities=[
                            Quantity(
                                'k_calculation',
                                rf'((?:k\-point *# *1:|k\-point *:*1\s)[\s\S]+?){re_n} *{re_n}',
                                repeats=True, sub_parser=TextParser(quantities=[
                                    Quantity(
                                        'k_point',
                                        rf'{re_f},* +{re_f},* +{re_f}', dtype=np.dtype(np.float64)
                                    ),
                                    Quantity(
                                        'norm_initial_vector',
                                        rf'[Nn]orm.+?vector[=:] *{re_f}', dtype=np.float64
                                    ),
                                    Quantity(
                                        'converged',
                                        r'(=\> CONVERGED)', str_operation=lambda x: True
                                    ),
                                    Quantity(
                                        'converged',
                                        r'(not converged)', str_operation=lambda x: False
                                    ),
                                    Quantity(
                                        'n_iter_error',
                                        rf'iter +(\d+) with error= *{re_f}'
                                    ),
                                    Quantity(
                                        'n_iter_error',
                                        rf'final error after *(\d+) *iterations\: (.+)'
                                    )
                                ])
                            )
                        ])
                    ),
                    Quantity(
                        'step_2',
                        r'(Begin STEP 2 [\s\S]+?End STEP 2)',
                        sub_parser=TextParser(quantities=[
                            Quantity(
                                'energy_zero',
                                rf'(?:energy\-zero of the spectrum|xe0) \[eV\]: *({re_f})',
                                dtype=np.float64, unit='eV'
                            ),
                            Quantity(
                                'xemin',
                                rf'xemin \[eV\]: *({re_f})', dtype=np.float64, unit='eV'
                            ),
                            Quantity(
                                'xemax',
                                rf'xemax \[eV\]: *({re_f})', dtype=np.float64, unit='eV'
                            ),
                            Quantity(
                                'xnepoint',
                                r'xnepoint: *(\d+)', dtype=np.int32
                            ),
                            Quantity(
                                'broadening_parameter',
                                rf'constant broadening parameter \[eV\]: *{re_f}', dtype=np.float64
                            ),
                            Quantity(
                                'energy_core_level',
                                rf'Core level energy \[eV\]: *{re_f}', dtype=np.float64, unit='eV'
                            ),
                            Quantity(
                                'file',
                                r'Cross-section successfully written in (\S+)'
                            )
                        ])
                    )
                ])
            )
        ]


class QuantumEspressoXSpectraParser:
    def __init__(self):
        self.mainfile_parser = MainfileParser()
        self.xanesdata_parser = DataTextParser()

    def parse_system(self):
        sec_run = self.archive.run[-1]
        sec_system = sec_run.m_create(System)
        sec_atoms = sec_system.m_create(Atoms)
        alat = self.mainfile_parser.get('lattice_parameter', 1)
        crystal_axes = self.mainfile_parser.crystal_axes
        if crystal_axes is not None:
            sec_atoms.lattice_vectors = crystal_axes * alat * ureg.bohr
        cartesian_axes = self.mainfile_parser.cartesian_axes
        if cartesian_axes is not None:
            sec_atoms.labels = self.mainfile_parser.cartesian_axes[0]
            sec_atoms.positions = self.mainfile_parser.cartesian_axes[1] * alat * ureg.bohr
        sec_atoms.periodic = [True, True, True]

        system_keys = [
            'bravais_lattice_index', 'lattice_parameter', 'unit_cell_volume',
            'n_atoms_cell', 'n_atomic_sites']
        for key in system_keys:
            setattr(sec_system, f'x_qe_xspectra_{key}', self.mainfile_parser.get(key))

    def parse_method(self):
        scf = self.mainfile_parser.scf

        sec_run = self.archive.run[-1]
        sec_method = sec_run.m_create(Method)

        # Smearing
        smearing = self.mainfile_parser.smearing
        if smearing is not None:
            smearing_map = {'Methfessel-Paxton': 'methfessel-paxton'}
            sec_method.electronic = Electronic(smearing=Smearing(
                kind=smearing_map.get(smearing[0]),
                width=(smearing[1] * ureg.Ry).to_base_units().magnitude))

        # Pseudopotentials
        # TODO should be here or in the DFT entry??
        for pseudopot in self.mainfile_parser.get('pseudopot', []):
            sec_atom_parameters = sec_method.m_create(AtomParameters)
            sec_atom_parameters.label = pseudopot.element
            sec_atom_parameters.n_valence_electrons = pseudopot.zval
            atom_keys = [
                'file', 'md5_check_sum', 'type', 'n_radial_grid_points', 'n_l', 'l',
                'n_q_coefficients', 'q_coefficients']
            for key in atom_keys:
                setattr(sec_atom_parameters, f'x_qe_xspectra_{key}', pseudopot.get(key))

        # code-specific
        if scf is not None and scf.n_parallel is not None:
            sec_method.x_qe_xspectra_n_parallel_sticks = x_qe_xspectra_n_parallel(
                x_qe_xspectra_n_parallel_min=scf.n_parallel[0][:3], x_qe_xspectra_n_parallel_max=scf.n_parallel[1][:3],
                x_qe_xspectra_n_parallel_sum=scf.n_parallel[2][:3])
            sec_method.x_qe_xspectra_n_parallel_g_vectors = x_qe_xspectra_n_parallel(
                x_qe_xspectra_n_parallel_min=scf.n_parallel[0][3:], x_qe_xspectra_n_parallel_max=scf.n_parallel[1][3:],
                x_qe_xspectra_n_parallel_sum=scf.n_parallel[2][3:])

        method_keys = [
            'kinetic_energy_cutoff', 'charge_density_cutoff', 'convergence_threshold',
            'beta', 'exchange_correlation', 'n_kohn_sham_states']
        for key in method_keys:
            setattr(sec_method, f'x_qe_xspectra_{key}', self.mainfile_parser.get(key))

        # Photon
        sec_photon = sec_method.m_create(Photon)
        if self.mainfile_parser.input.get('x_qe_xspectra_calculation', '') == 'hpsi':
            self.logger.warning('Calculation ran in the debug option HPSI. Please, check your upload.')
            return
        sec_photon.multipole_type = self.mainfile_parser.input.get('x_qe_xspectra_calculation', '').split('_')[1]
        sec_photon.polarization = self.mainfile_parser.input.get('x_qe_xspectra_xepsilon', [])

        # Core-hole
        sec_method_core = sec_run.m_create(Method)
        if sec_run.m_xpath('method[0]'):
            sec_method_core.starting_method_ref = sec_run.method[0]
        sec_core_hole = sec_method_core.m_create(CoreHole)
        sec_core_hole.mode = 'absorption'  # XSPECTRA can only handle XAS/XANES -> absorption
        sec_core_hole.solver = self.mainfile_parser.xanes.get('algorithm', [])[0]
        # TODO talk with devs to get the edge info
        # sec_core_hole.edge
        if sec_run.x_qe_xspectra_input.x_qe_xspectra_main_plot_parameters.get('gamma_mode') == 'constant':
            sec_core_hole.broadening = sec_run.x_qe_xspectra_input.x_qe_xspectra_main_plot_parameters.get('using')

    def parse_scc(self):
        sec_run = self.archive.run[-1]
        xanes_file = self.mainfile_parser.xanes.get('step_2', {}).get('file')
        if xanes_file:
            self.xanesdata_parser.mainfile = os.path.join(self.maindir, xanes_file)
            data = self.xanesdata_parser.data

            sec_scc = sec_run.m_create(Calculation)
            sec_scc.system_ref = sec_run.system[-1]
            sec_scc.method_ref = sec_run.method[-1]

            sec_spectra = sec_scc.m_create(Spectra)
            sec_spectra.type = self.mainfile_parser.input.get('x_qe_xspectra_calculation', '').split('_')[0].upper()
            sec_spectra.n_energies = data.shape[0]
            sec_spectra.excitation_energies = data[:, 0] * ureg.eV
            unit_cell_volume = self.mainfile_parser.get('unit_cell_volume').magnitude  # in bohr^3
            sec_spectra.intensities = scipy.constants.fine_structure * data[:, 1] / (data[:, 0] * unit_cell_volume)

            for key, val in self.mainfile_parser.xanes.get('step_2', {}).items():
                if key != 'file':
                    setattr(sec_spectra, f'x_qe_xspectra_{key}', val)

    def parse(self, filepath, archive, logger):
        self.logger = logging.getLogger(__name__) if logger is None else logger
        self.archive = archive
        self.filepath = os.path.abspath(filepath)
        self.maindir = os.path.dirname(self.filepath)

        self.mainfile_parser.mainfile = self.filepath

        sec_run = self.archive.m_create(Run)
        sec_run.program = Program(
            name='Quantum ESPRESSO XSPECTRA', version=self.mainfile_parser.get('program_version', ''))

        start_time = self.mainfile_parser.start_time
        if start_time is not None:
            date = datetime.strptime(start_time.strip().replace(' ', ''), '%d%b%Y%H:%M:%S')
            sec_run.time_run = TimeRun(date_start=(date - datetime.utcfromtimestamp(0)).total_seconds())

        input = self.mainfile_parser.input
        if input is not None:
            sec_input = sec_run.m_create(x_qe_xspectra_input)
            for key, val in input.items():
                if key == 'x_qe_xspectra_main_plot_parameters':
                    val = {v[0].strip(): v[1] for v in val.get('key_val', [])}
                setattr(sec_input, key, val)

        # System
        self.parse_system()

        # Method
        self.parse_method()

        # Calculation
        self.parse_scc()

        # Workflow
        sec_workflow = archive.m_create(Workflow)
        sec_workflow.type = 'single_point'
        workflow = SinglePoint2()
        archive.workflow2 = workflow
