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

from nomad.units import ureg
from nomad.parsing.file_parser import TextParser, Quantity
from nomad.datamodel.metainfo.simulation.run import Run, Program
from nomad.datamodel.metainfo.simulation.method import (
    Method, AtomParameters, Electronic, Smearing)
from nomad.datamodel.metainfo.simulation.system import System, Atoms
from nomad.datamodel.metainfo.simulation.calculation import (
    Calculation, VibrationalFrequencies
)

from .metainfo.quantum_espresso_phonon import (
    x_qe_phonon_n_parallel, x_qe_phonon_representation, x_qe_phonon_scf_iteration)


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
                labels.append(line[1])
                positions.append(line[-4:-1])
            return labels, np.array(positions, np.dtype(np.float64))

        calc_quantities = [
            Quantity(
                'g_cutoff_fft_grid',
                rf'G cutoff *= *({re_f}) *\( *(\d+) G\-vectors\) *FFT grid: *\( (\d+), (\d+), (\d+)\)',
                dtype=np.dtype(np.float64)
            ),
            Quantity(
                'g_cutoff_smooth_grid',
                rf'G cutoff *= *({re_f}) *\( *(\d+) G\-vectors\) *smooth grid: *\( (\d+), (\d+), (\d+)\)',
                dtype=np.dtype(np.float64)
            ),
            Quantity('n_kpoints', r'number of k points *= *(\d+)', dtype=np.int32),
            Quantity('smearing', rf'(\S+) smearing, width \(Ry\) *= *({re_f})'),
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
            Quantity('point_group', r', *(.+?) +point group', dtype=str, flatten=False),
            Quantity('alpha_ewald', rf' Alpha used in Ewald sum = *({re_f})', dtype=np.float64),
            Quantity('negative_rho', rf'negative rho \(up, down\): *({re_f}) *({re_f})', dtype=np.dtype(np.float64)),
            Quantity(
                'representation',
                r'(tion # *\d+ mode[\s\S]+?)(?:Represent|\Z)',
                repeats=True, sub_parser=TextParser(quantities=[
                    Quantity('number', r'tion # *(\d+)', dtype=np.int32),
                    Quantity(
                        'modes', r'modes* #([\d ]+)',
                        str_operation=lambda x: x.strip().split(), dtype=np.int32
                    ),
                    Quantity(
                        'scf', r'Self\-consistent Calculation([\s\S]+?)End of self-consistent calculation',
                        sub_parser=TextParser(quantities=[
                            Quantity(
                                'iter',
                                r'iter # +(\d+)',
                                repeats=True, dtype=np.int32
                            ),
                            Quantity(
                                'total_cpu_time',
                                rf'total cpu time *: *({re_f})',
                                repeats=True, dtype=np.float64
                            ),
                            Quantity(
                                'time',
                                rf'av\.it\.: *({re_f})',
                                repeats=True, dtype=np.float64
                            ),
                            Quantity(
                                'threshold',
                                rf'thresh *= *({re_f})',
                                repeats=True, dtype=np.float64
                            ),
                            Quantity(
                                'alpha_mix',
                                rf'alpha\_mix *= *({re_f})',
                                repeats=True, dtype=np.float64
                            ),
                            Quantity(
                                'ddv_scf',
                                rf'\|ddv_scf\|\^2 *= *({re_f})',
                                repeats=True, dtype=np.float64
                            ),
                            Quantity(
                                'fermi_energy_shift',
                                rf'Fermi energy shift \(Ry\) *= *({re_f}) +({re_f})',
                                repeats=True, dtype=np.dtype(np.float64)
                            )
                        ])
                    ),
                    Quantity('converged', r'(Convergence has been achieved)', str_operation=lambda x: True)
                ])
            ),
        ]

        self._quantities = [
            Quantity('program_version', r'Program PHONON v.([\d\.]+)', dtype=str),
            Quantity(
                'calculation', rf'(q = +{re_f} +{re_f} +{re_f} *{re_n}[\s\S]+?)(?:Calculation of|\Z)',
                repeats=True, sub_parser=TextParser(quantities=[
                    Quantity('q', r'q = +({re_f} +{re_f} +{re_f})', dtype=np.dtype(np.float64)),
                    Quantity(
                        'n_parallel',
                        r'Parallelization info\s+\-+\s+sticks\:.+\s+'
                        r'((?:\w+ +\d+ +\d+ +\d+ +\d+ +\d+ +\d+\s+)+)',
                        str_operation=lambda x: [line.strip().split()[1:] for line in x.strip().splitlines()],
                        dtype=np.dtype(np.int32)
                    ),
                    Quantity(
                        'bravais_lattice_index',
                        r'bravais\-lattice index *= *(\d+)', dtype=np.int32
                    ),
                    Quantity(
                        'lattice_parameter', rf'lattice parameter \(alat\) *= *({re_f}) *a\.u\.',
                        dtype=np.float64, unit=ureg.bohr
                    ),
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
                        'cartesian_axes',
                        rf'positions \(alat units\)\s+((?:\d+ +[A-Z][a-z]* .+{re_f} +{re_f} +{re_f} +\)\s*)+)',
                        str_operation=to_positions
                    ),
                    Quantity(
                        'dynamical_matrix',
                        r'(Computing dynamical matrix[\s\S]+?)(?:Diagonalizing|\Z)',
                        sub_parser=TextParser(quantities=calc_quantities)
                    ),
                    Quantity(
                        'frequencies',
                        rf'freq \( +\d+\) += +{re_f} \[THz\] += +({re_f}) \[cm\-1\]',
                        repeats=True, dtype=np.float64
                    )
                ])
            ),
        ]


class QuantumEspressoPhononParser:
    def __init__(self):
        self.mainfile_parser = MainfileParser()

    def parse(self, filepath, archive, logger):
        logger = logging.getLogger(__name__) if logger is None else logger
        self.archive = archive
        self.filepath = os.path.abspath(filepath)

        self.mainfile_parser.mainfile = self.filepath

        sec_run = self.archive.m_create(Run)
        sec_run.program = Program(
            name='Quantum Espresso Phonon', version=self.mainfile_parser.get('program_version', ''))

        for calculation in self.mainfile_parser.get('calculation', []):
            sec_calc = sec_run.m_create(Calculation)

            sec_method = sec_run.m_create(Method)
            n_parallel = calculation.get('n_parallel')
            if n_parallel is not None:
                sec_method.x_qe_phonon_n_parallel_sticks = x_qe_phonon_n_parallel(
                    x_qe_phonon_n_parallel_min=n_parallel[0][:3], x_qe_phonon_n_parallel_max=n_parallel[1][:3],
                    x_qe_phonon_n_parallel_sum=n_parallel[2][:3])
                sec_method.x_qe_phonon_n_parallel_g_vectors = x_qe_phonon_n_parallel(
                    x_qe_phonon_n_parallel_min=n_parallel[0][3:], x_qe_phonon_n_parallel_max=n_parallel[1][3:],
                    x_qe_phonon_n_parallel_sum=n_parallel[2][3:])

            sec_system = sec_run.m_create(System)
            sec_atoms = sec_system.m_create(Atoms)
            if calculation.get('crystal_axes') is not None:
                sec_atoms.lattice_vectors = calculation.crystal_axes * calculation.get('alat', 1) * ureg.bohr
            if calculation.get('cartesian_axes') is not None:
                sec_atoms.labels = calculation.cartesian_axes[0]
                sec_atoms.positions = calculation.cartesian_axes[1] * calculation.get('alat', 1) * ureg.bohr
            sec_atoms.periodic = [True, True, True]

            system_keys = [
                'bravais_lattice_index', 'lattice_parameter', 'unit_cell_volume',
                'n_atoms_cell', 'n_atomic_sites']
            for key in system_keys:
                setattr(sec_system, f'x_qe_phonon_{key}', calculation.get(key))

            method_keys = [
                'kinetic_energy_cutoff', 'charge_density_cutoff', 'convergence_threshold',
                'beta', 'exchange_correlation', 'n_kohn_sham_states']
            for key in method_keys:
                setattr(sec_method, f'x_qe_phonon_{key}', calculation.get(key))

            # vibrational frequencies
            if calculation.frequencies is not None:
                sec_calc.vibrational_frequencies.append(
                    VibrationalFrequencies(value=calculation.frequencies * (1 / ureg.cm)))

            # specs and results of dynamical matrix calculation
            if calculation.dynamical_matrix is not None:
                for pseudopot in calculation.dynamical_matrix.get('pseudopot', []):
                    sec_atom_parameters = sec_method.m_create(AtomParameters)
                    sec_atom_parameters.label = pseudopot.element
                    sec_atom_parameters.n_valence_electrons = pseudopot.zval
                    atom_keys = [
                        'file', 'md5_check_sum', 'type', 'n_radial_grid_points', 'n_l', 'l',
                        'n_q_coefficients', 'q_coefficients']
                    for key in atom_keys:
                        setattr(sec_atom_parameters, f'x_qe_phonon_{key}', pseudopot.get(key))

                if calculation.dynamical_matrix.g_cutoff_fft_grid is not None:
                    sec_method.x_qe_phonon_fft_g_cutoff = calculation.dynamical_matrix.g_cutoff_fft_grid[0]
                    sec_method.x_qe_phonon_fft_g_vectors = calculation.dynamical_matrix.g_cutoff_fft_grid[1]
                    sec_method.x_qe_phonon_fft_grid = calculation.dynamical_matrix.g_cutoff_fft_grid[2:5]
                if calculation.dynamical_matrix.g_cutoff_smooth_grid is not None:
                    sec_method.x_qe_phonon_smooth_g_cutoff = calculation.dynamical_matrix.g_cutoff_smooth_grid[0]
                    sec_method.x_qe_phonon_smooth_g_vectors = calculation.dynamical_matrix.g_cutoff_smooth_grid[1]
                    sec_method.x_qe_phonon_smooth_grid = calculation.dynamical_matrix.g_cutoff_smooth_grid[2:5]

                if calculation.dynamical_matrix.smearing is not None:
                    smearing_map = {'Methfessel-Paxton': 'methfessel-paxton'}
                    sec_method.electronic = Electronic(smearing=Smearing(
                        kind=smearing_map.get(calculation.dynamical_matrix.smearing[0]),
                        width=(calculation.dynamical_matrix.smearing[1] * ureg.Ry).to_base_units().magnitude))

                sec_method.x_qe_phonon_n_kpoints = calculation.dynamical_matrix.n_kpoints
                sec_method.x_qe_phonon_alpha_ewald = calculation.dynamical_matrix.alpha_ewald
                sec_method.x_qe_phonon_negative_rho = calculation.dynamical_matrix.negative_rho

                sec_system.x_qe_phonon_point_group = calculation.dynamical_matrix.point_group

                for representation in calculation.dynamical_matrix.get('representation', []):
                    sec_representation = sec_calc.m_create(x_qe_phonon_representation)
                    sec_representation.x_qe_phonon_number = representation.number
                    sec_representation.x_qe_phonon_modes = representation.modes
                    sec_representation.x_qe_phonon_converged = representation.converged
                    if representation.scf is not None:
                        for n, iter in enumerate(representation.scf.get('iter', [])):
                            sec_scf = sec_representation.m_create(x_qe_phonon_scf_iteration)
                            sec_scf.x_qe_phonon_iter = iter
                            sec_scf.x_qe_phonon_total_cpu_time = representation.scf.total_cpu_time[n]
                            sec_scf.x_qe_phonon_time = representation.scf.time[n]
                            sec_scf.x_qe_phonon_threshold = representation.scf.threshold[n]
                            sec_scf.x_qe_phonon_alpha_mix = representation.scf.alpha_mix[n]
                            sec_scf.x_qe_phonon_ddv_scf = representation.scf.ddv_scf[n]
                            if representation.scf.fermi_energy_shift is not None:
                                n_modes = len(representation.get('modes', []))
                                sec_scf.x_qe_phonon_fermi_energy_shift = representation.scf.fermi_energy_shift[n * n_modes: n * n_modes + 2]
