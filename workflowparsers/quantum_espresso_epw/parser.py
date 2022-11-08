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
from datetime import datetime

from nomad.units import ureg
from nomad.parsing.file_parser import TextParser, Quantity
from nomad.datamodel.metainfo.simulation.run import Run, Program, TimeRun
from nomad.datamodel.metainfo.simulation.method import (
    Method, Electronic, KMesh, AtomParameters
)
from nomad.datamodel.metainfo.simulation.system import (
    System, Atoms
)
from nomad.datamodel.metainfo.simulation.calculation import (
    Calculation, Energy
)
from .metainfo.quantum_espresso_epw import (
    x_qe_epw_irreducible_q_point, x_qe_epw_self_energy_migdal, x_qe_epw_self_energy,
    x_qe_epw_eliashberg_spectral_function_migdal_approximation, x_qe_epw_timimg
)


re_float = r'[-+]?\d+\.\d*(?:[Ee][-+]\d+)?'
re_n = r'[\n\r]'


class MainfileParser(TextParser):
    def __init__(self):
        super().__init__()

    def init_quantities(self):
        def to_positions(val_in):
            labels, masses, positions = [], [], []
            for line in val_in.strip().splitlines():
                line = line.strip().split()
                labels.append(line[1])
                masses.append(line[2])
                positions.append(line[-4:-1])
            masses = np.array(masses, np.dtype(np.float64))
            positions = np.array(positions, np.dtype(np.float64))
            return labels, masses, positions

        def to_k_points(val_in):
            kpoints, weights = [], []
            for line in val_in.strip().splitlines():
                line = line.split('(')[-1].strip().split('), wk =')
                kpoints.append(line[0].split())
                weights.append(line[1])
            return np.array(kpoints, np.dtype(np.float64)), np.array(weights, np.dtype(np.float64))

        self._quantities = [
            Quantity('program_version', r'Program EPW v\.([\d\.]+)', dtype=str),
            Quantity('start_time', r'starts on +(\w+) +at +([\d: ]+)', flatten=False, dtype=str),
            Quantity('restart', r'RESTART \- (RESTART)', dtype=str),
            Quantity(
                'bravais_lattice_index',
                r'bravais\-lattice index *= *(\d+)', dtype=np.int32
            ),
            Quantity(
                'lattice_parameter', rf'lattice parameter \(a_0\) *= *({re_float}) *a\.u\.',
                dtype=np.float64, unit=ureg.bohr
            ),
            Quantity(
                'unit_cell_volume',
                rf'unit\-cell volume *= *({re_float}) \(a\.u\.\)\^3',
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
                rf'kinetic\-energy cut\-off *= *({re_float}) *Ry',
                dtype=np.float64, unit=ureg.rydberg
            ),
            Quantity(
                'charge_density_cutoff',
                rf'charge density cut\-off *= *({re_float}) *Ry',
                dtype=np.float64, unit=ureg.rydberg
            ),
            Quantity(
                'convergence_threshold',
                rf'convergence threshold *= *({re_float})', dtype=np.float64
            ),
            Quantity(
                'exchange_correlation',
                r'Exchange\-correlation *= *(.+)', dtype=str
            ),
            Quantity(
                'lattice_vectors',
                rf'crystal axes: \(cart\. coord.\ in units of a_0\)\s+'
                rf'a\(1\) = \( *({re_float} +{re_float} +{re_float}) *\)\s+'
                rf'a\(2\) = \( *({re_float} +{re_float} +{re_float}) *\)\s+'
                rf'a\(3\) = \( *({re_float} +{re_float} +{re_float}) *\)\s+',
                dtype=np.dtype(np.float64), shape=[3, 3]
            ),
            Quantity(
                'cartesian_axes',
                rf'site n\. +atom +mass +positions \(a_0 units\)\s+'
                rf'((?:\d+ +\w+ +{re_float} +tau\( *\d+\) = \( *{re_float} +{re_float} +{re_float} +\)\s+)+)',
                str_operation=to_positions
            ),
            Quantity(
                'g_cutoff_fft_grid',
                rf'G cutoff *= *({re_float}) *\( *(\d+) G\-vectors\) *FFT grid: *\( (\d+), (\d+), (\d+)\)',
                dtype=np.dtype(np.float64)
            ),
            Quantity(
                'g_cutoff_smooth_grid',
                rf'G cutoff *= *({re_float}) *\( *(\d+) G\-vectors\) *smooth grid: *\( (\d+), (\d+), (\d+)\)',
                dtype=np.dtype(np.float64)
            ),
            Quantity('n_kpoints', r'number of k points *= *(\d+)', dtype=np.int32),
            Quantity(
                'gaussian_broadening',
                rf'gaussian broad\. \(Ry\)= +({re_float})',
                dtype=np.float64, unit=ureg.rydberg
            ),
            Quantity('n_gauss', r'gnauss *= *(\d+)', dtype=np.int32),
            Quantity(
                'k_points',
                rf'cart\. coord\. in units 2pi/a_0\s+'
                rf'((?:k\( *\d+\) *= *\( +{re_float} +{re_float} +{re_float} *\), wk = +{re_float}\s+)+)',
                str_operation=to_k_points
            ),
            Quantity(
                'pseudopot',
                rf'(PseudoPot\. #[\s\S]+?){re_n} *{re_n}',
                repeats=True, sub_parser=TextParser(quantities=[
                    Quantity('element', r'for ([A-Z][a-z]*)'),
                    Quantity('file', r' read from file\:\s+(\S+)', dtype=str),
                    Quantity('md5_check_sum', r'MD5 check sum\: +(\S+)', dtype=str),
                    Quantity('type', r'Pseudo is (.+?),', flatten=False, dtype=str),
                    Quantity('zval', rf'Zval *= *({re_float})', dtype=np.float64),
                    Quantity('n_radial_grid_points', r'Using radial grid of *(\d+) points', dtype=np.int32),
                    Quantity('n_beta_functions', r'(\d+) beta functions', dtype=np.int32),
                    Quantity('l', r'l\(\d+\) *= *(\d+)', dtype=np.int32, repeats=True),
                    Quantity('n_q_coefficients', r'Q\(r\) pseudized with  (\d+) coefficients', dtype=np.int32),
                    Quantity(
                        'q_coefficients',
                        rf'rinner *= *((?:{re_float}\s+)+)',
                        str_operation=lambda x: x.strip().split(), dytpe=np.dtype(np.float64)
                    )
                ])
            ),
            Quantity(
                'irreducible_q_point',
                r'(irreducible q point # +\d+\s+\=+[\s\S]+?\={50})',
                repeats=True, sub_parser=TextParser(quantities=[
                    Quantity('n_symmetries', r'Symmetries of small group of q\: (\d+)', dtype=np.int32),
                    Quantity('n_q_star', r'Number of q in the star *= *(\d+)', dtype=np.int32),
                    Quantity(
                        'q_star',
                        rf'List of q in the star\:\s+((?:\d+ +{re_float} +{re_float} +{re_float}\s+)+)',
                        dtype=np.dtype(np.float64)
                    )
                ])
            ),
            Quantity(
                'n_ws_vectors_electrons',
                r'Number of WS vectors for electrons *(\d+)', dtype=np.int32
            ),
            Quantity(
                'n_ws_vectors_phonons',
                r'Number of WS vectors for phonons *(\d+)', dtype=np.int32
            ),
            Quantity(
                'n_ws_vectors_electron_phonon',
                r'Number of WS vectors for electron\-phonon *(\d+)', dtype=np.int32
            ),
            Quantity(
                'n_max_cores',
                r'Maximum number of cores for efficient parallelization *(\d+)', dtype=np.int32
            ),
            Quantity(
                'use_ws',
                r'Results may improve by using use_ws == \.(TRUE)\.',
                str_operation=lambda x: True, dtype=bool
            ),
            Quantity(
                'q_mesh',
                r'Using uniform q\-mesh: *(\d+) +(\d+) +(\d+)', dtype=np.dtype(np.int32)
            ),
            Quantity(
                'n_q_mesh',
                r'Size of q point mesh for interpolation: *(\d+)', dtype=np.int32
            ),
            Quantity(
                'k_mesh',
                r'Using uniform k\-mesh: *(\d+) +(\d+) +(\d+)', dtype=np.dtype(np.int32)
            ),
            Quantity(
                'n_k_mesh',
                r'Size of k point mesh for interpolation: *(\d+)', dtype=np.int32
            ),
            Quantity(
                'n_max_kpoints_per_pool',
                r'Max number of k points per pool: *(\d+)', dtype=np.int32
            ),
            Quantity(
                'e_fermi_coarse_grid',
                rf'Fermi energy coarse grid = *({re_float}) eV',
                dtype=np.float64, unit=ureg.eV
            ),
            Quantity(
                'n_electrons',
                rf'The Fermi level will be determined with *({re_float}) electrons',
                dtype=np.float64
            ),
            Quantity(
                'e_fermi',
                rf'Fermi energy is calculated from the fine k\-mesh: Ef = *({re_float}) eV',
                dtype=np.float64, unit=ureg.eV
            ),
            Quantity(
                'self_energy_migdal_approximation',
                r'(Phonon \(Imaginary\) Self\-Energy in the Migdal Approximation\s+\=+[\s\S]+?\={50})',
                sub_parser=TextParser(quantities=[
                    Quantity(
                        'fermi_surface_thickness',
                        rf'Fermi Surface thickness = *({re_float}) eV',
                        dtype=np.float64, unit=ureg.eV
                    ),
                    Quantity(
                        'golden_rule_t',
                        rf'Golden Rule strictly enforced with T = *({re_float}) eV',
                        dtype=np.float64, unit=ureg.eV
                    ),
                    Quantity(
                        'gaussian_broadening',
                        rf'Gaussian Broadening: *({re_float}) eV',
                        dtype=np.float64, unit=ureg.eV
                    ),
                    Quantity('n_gauss', r'gauss *= *(\d+)', dtype=np.int32),
                    Quantity(
                        'dos_ef',
                        rf'DOS = *({re_float}) states/spin/eV/Unit Cell',
                        dtype=np.float64
                    ),
                    Quantity('e_fermi', rf'at Ef= *({re_float}) eV', dtype=np.float64, unit=ureg.eV),
                    Quantity(
                        'self_energy',
                        r'(ismear = +\d+ iq =.+\s+\-+[\s\S]+?\-{50})',
                        repeats=True, sub_parser=TextParser(quantities=[
                            Quantity('ismear', r'ismear = *(\d+)', dtype=np.int32),
                            Quantity('iq', r'iq = *(\d+)', dtype=np.int32),
                            Quantity('coord', rf'coord\.: +{re_float} +{re_float} +{re_float}', dtype=np.dtype(np.float64)),
                            Quantity('wt', rf'wt: +{re_float}', dtype=np.float64),
                            Quantity('temp', rf'Temp: *({re_float}) *K', dtype=np.float64, unit=ureg.K),
                            Quantity(
                                'lambda_gamma_omega',
                                rf'lambda___\( *\d+ *\)= *({re_float}) *gamma___= *({re_float}) meV *omega= *({re_float}) meV',
                                dtype=np.float64, repeats=True
                            ),
                            Quantity(
                                'lambda_gamma_omega_tr',
                                rf'lambda_tr\( *\d+ *\)= *({re_float}) *gamma_tr= *({re_float}) meV *omega= *({re_float}) meV',
                                dtype=np.float64, repeats=True
                            ),
                            Quantity('lambda_tot', rf'lambda___\( *tot *\)= *({re_float})'),
                            Quantity('lambda_tot_tr', rf'lambda_tr\( *tot *\)= *({re_float})'),
                        ])
                    )
                ])
            ),
            Quantity(
                'eliashberg_spectral_function_migdal_approximation',
                r'(Eliashberg Spectral Function in the Migdal Approximation\s+\=+[\s\S]+?\={50})',
                sub_parser=TextParser(quantities=[
                    Quantity('lambda', rf'lambda : *({re_float})', dtype=np.float64),
                    Quantity('lambda_tr', rf'lambda_tr : *({re_float})', dtype=np.float64),
                    Quantity('logavg', rf'logavg = *({re_float})', dtype=np.float64),
                    Quantity('l_a2f', rf'l_a2f = *({re_float})', dtype=np.float64),
                    Quantity(
                        'mu_tc', rf'mu = *({re_float}) Tc = *({re_float}) K',
                        dtype=np.dtype(np.float64), repeats=True
                    ),
                    Quantity(
                        'timing',
                        rf' +(.+?) +\: +({re_float})s CPU +({re_float})s WALL \( +(\d+) calls',
                        repeats=True, str_operation=lambda x: x.rsplit(' ', 3)
                    )
                ])
            )
        ]


class QuantumEspressoEPWParser:
    def __init__(self):
        self.mainfile_parser = MainfileParser()

    def parse(self, filepath, archive, logger):
        logger = logging.getLogger(__name__) if logger is None else logger
        self.archive = archive
        self.filepath = os.path.abspath(filepath)

        self.mainfile_parser.mainfile = self.filepath

        sec_run = self.archive.m_create(Run)
        sec_run.program = Program(
            name='Quantum Espresso EPW', version=self.mainfile_parser.get('program_version', ''))

        start_time = self.mainfile_parser.start_time
        if start_time is not None:
            date = datetime.strptime(start_time.replace(' ', ''), '%d%b%Y%H:%M:%S')
            sec_run.time_run = TimeRun(date_start=(date - datetime.utcfromtimestamp(0)).total_seconds())

        sec_method = sec_run.m_create(Method)
        sec_method.electronic = Electronic(n_electrons=self.mainfile_parser.n_electrons)
        g_cutoff_fft_grid = self.mainfile_parser.g_cutoff_fft_grid
        if g_cutoff_fft_grid is not None:
            sec_method.x_qe_epw_fft_g_cutoff = g_cutoff_fft_grid[0]
            sec_method.x_qe_epw_fft_g_vectors = g_cutoff_fft_grid[1]
            sec_method.x_qe_epw_fft_grid = g_cutoff_fft_grid[2:5]
        g_cutoff_smooth_grid = self.mainfile_parser.g_cutoff_smooth_grid
        if g_cutoff_smooth_grid is not None:
            sec_method.x_qe_epw_smooth_g_cutoff = g_cutoff_smooth_grid[0]
            sec_method.x_qe_epw_smooth_g_vectors = g_cutoff_smooth_grid[1]
            sec_method.x_qe_epw_smooth_grid = g_cutoff_smooth_grid[2:5]

        k_points = self.mainfile_parser.k_points
        if k_points is not None:
            sec_method.k_mesh = KMesh(points=k_points[0], weights=k_points[1])

        for pseudopot in self.mainfile_parser.get('pseudopot', []):
            sec_atom_parameters = sec_method.m_create(AtomParameters)
            sec_atom_parameters.label = pseudopot.element
            sec_atom_parameters.n_valence_electrons = pseudopot.zval
            atom_keys = [
                'file', 'md5_check_sum', 'type', 'n_radial_grid_points', 'n_l', 'l',
                'n_q_coefficients', 'q_coefficients']
            for key in atom_keys:
                setattr(sec_atom_parameters, f'x_qe_phonon_{key}', pseudopot.get(key))

        method_keys = [
            'n_ws_vectors_electrons', 'n_ws_vectors_phonons', 'n_ws_vectors_electron_phonon',
            'n_max_cores', 'use_ws', 'q_mesh', 'n_q_mesh', 'k_mesh', 'n_k_mesh',
            'n_max_kpoints_per_pool', 'gaussian_broadening', 'n_gauss'
        ]
        for key in method_keys:
            setattr(sec_method, f'x_qe_epw_{key}', self.mainfile_parser.get(key))

        # If this is a continuation (restart) run, not all fields have meaningful values so skip those
        # in such case.
        if self.mainfile_parser.get('restart') is None:
            method_keys = [
                'kinetic_energy_cutoff', 'charge_density_cutoff',
                'convergence_threshold', 'exchange_correlation', 'n_kpoints'
            ]
            for key in method_keys:
                setattr(sec_method, f'x_qe_epw_{key}', self.mainfile_parser.get(key))

            sec_system = sec_run.m_create(System)
            sec_atoms = sec_system.m_create(Atoms)
            alat = self.mainfile_parser.lattice_parameter
            lattice_vectors = self.mainfile_parser.lattice_vectors
            if lattice_vectors is not None:
                sec_atoms.lattice_vectors = np.dot(lattice_vectors, alat)

            cartesian_axes = self.mainfile_parser.cartesian_axes
            if cartesian_axes is not None:
                sec_atoms.labels = cartesian_axes[0]
                sec_atoms.positions = np.dot(cartesian_axes[2], alat)

            system_keys = [
                'bravais_lattice_index', 'lattice_parameter', 'unit_cell_volume', 'n_atoms_cell',
                'n_atomic_types'
            ]
            for key in system_keys:
                setattr(sec_system, f'x_qe_epw_{key}', self.mainfile_parser.get(key))

        for q_point in self.mainfile_parser.get('irreducible_q_point', []):
            sec_q_point = sec_system.m_create(x_qe_epw_irreducible_q_point)
            sec_q_point.x_qe_epw_n_symmetries = q_point. n_symmetries
            sec_q_point.x_qe_epw_n_q_star = q_point.n_q_star
            sec_q_point.x_qe_epw_q_star = np.reshape(q_point.q_star, (q_point.n_q_star, 4)).T[1:4].T

        sec_calc = sec_run.m_create(Calculation)
        sec_calc.energy = Energy(fermi=self.mainfile_parser.e_fermi)
        sec_calc.x_qe_epw_e_fermi_coarse_grid = self.mainfile_parser.e_fermi_coarse_grid
        self_energy_migdal = self.mainfile_parser.self_energy_migdal_approximation
        if self_energy_migdal is not None:
            sec_migdal = sec_calc.m_create(x_qe_epw_self_energy_migdal)
            sec_migdal.x_qe_epw_fermi_surface_thickness = self_energy_migdal.fermi_surface_thickness
            sec_migdal.x_qe_epw_golden_rule_t = self_energy_migdal.golden_rule_t
            sec_migdal.x_qe_epw_gaussian_broadening = self_energy_migdal.gaussian_broadening
            sec_migdal.x_qe_epw_n_gauss = self_energy_migdal.n_gauss
            sec_migdal.x_qe_epw_dos_ef = self_energy_migdal.dos_ef
            for self_energy in self_energy_migdal.get('self_energy', []):
                sec_self_energy = sec_migdal.m_create(x_qe_epw_self_energy)
                sec_self_energy.x_qe_epw_ismear = self_energy.ismear
                sec_self_energy.x_qe_epw_iq = self_energy.iq
                sec_self_energy.x_qe_epw_coord = self_energy.coord
                sec_self_energy.x_qe_epw_wt = self_energy.wt
                sec_self_energy.x_qe_epw_temp = self_energy.temp
                lambda_gamma_omega = np.transpose(self_energy.lambda_gamma_omega)
                sec_self_energy.x_qe_epw_lambda = lambda_gamma_omega[0]
                sec_self_energy.x_qe_epw_gamma = lambda_gamma_omega[1]
                sec_self_energy.x_qe_epw_omega = lambda_gamma_omega[2]
                lambda_gamma_omega = np.transpose(self_energy.lambda_gamma_omega_tr)
                sec_self_energy.x_qe_epw_lambda_tr = lambda_gamma_omega[0]
                sec_self_energy.x_qe_epw_gamma_tr = lambda_gamma_omega[1]
                sec_self_energy.x_qe_epw_omega_tr = lambda_gamma_omega[2]
                sec_self_energy.x_qe_epw_lambda_tot = self_energy.lambda_tot
                sec_self_energy.x_qe_epw_lambda_tot_tr = self_energy.lambda_tot_tr

        spectral_migdal = self.mainfile_parser.eliashberg_spectral_function_migdal_approximation
        if spectral_migdal is not None:
            sec_migdal = sec_calc.m_create(x_qe_epw_eliashberg_spectral_function_migdal_approximation)
            sec_migdal.x_qe_epw_lambda = spectral_migdal.get('lambda')
            sec_migdal.x_qe_epw_lambda_tr = spectral_migdal.lambda_tr
            sec_migdal.x_qe_epw_logavg = spectral_migdal.logavg
            sec_migdal.x_qe_epw_l_a2f = spectral_migdal.l_a2f
            sec_migdal.x_qe_epw_mu_tc = spectral_migdal.mu_tc
            for timing in spectral_migdal.get('timing', []):
                sec_timing = sec_migdal.m_create(x_qe_epw_timimg)
                sec_timing.x_qe_epw_task = timing[0]
                sec_timing.x_qe_epw_cpu_time = timing[1]
                sec_timing.x_qe_epw_wall_time = timing[2]
                sec_timing.x_qe_epw_n_calls = timing[3]
