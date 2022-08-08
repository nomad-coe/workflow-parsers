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
import numpy as np            # pylint: disable=unused-import

from nomad.metainfo import (  # pylint: disable=unused-import
    MSection, MCategory, Category, Package, Quantity, Section, SubSection, SectionProxy,
    Reference, JSON
)
from nomad.datamodel.metainfo import simulation


m_package = Package()


class x_qe_phonon_n_parallel(MSection):

    m_def = Section(validate=False)

    x_qe_phonon_n_parallel_min = Quantity(
        type=np.dtype(np.float64),
        shape=[3],
        description='''
        ''')

    x_qe_phonon_n_parallel_max = Quantity(
        type=np.dtype(np.float64),
        shape=[3],
        description='''
        ''')

    x_qe_phonon_n_parallel_sum = Quantity(
        type=np.dtype(np.float64),
        shape=[3],
        description='''
        ''')


class Method(simulation.method.Method):

    m_def = Section(validate=False, extends_base_section=True)

    x_qe_phonon_kinetic_energy_cutoff = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='joule',
        description='''
        ''')

    x_qe_phonon_charge_density_cutoff = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='joule',
        description='''
        ''')

    x_qe_phonon_convergence_threshold = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        ''')

    x_qe_phonon_beta = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        ''')

    x_qe_phonon_exchange_correlation = Quantity(
        type=str,
        shape=[],
        description='''
        ''')

    x_qe_phonon_n_kohn_sham_states = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        ''')

    x_qe_phonon_parameters = Quantity(
        type=JSON,
        shape=[],
        description='''
        ''')

    x_qe_phonon_fft_g_cutoff = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        ''')

    x_qe_phonon_fft_g_vectors = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        ''')

    x_qe_phonon_fft_grid = Quantity(
        type=np.dtype(np.int32),
        shape=[3],
        description='''
        ''')

    x_qe_phonon_smooth_g_cutoff = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        ''')

    x_qe_phonon_smooth_g_vectors = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        ''')

    x_qe_phonon_smooth_grid = Quantity(
        type=np.dtype(np.int32),
        shape=[3],
        description='''
        ''')

    x_qe_phonon_n_kpoints = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        ''')

    x_qe_phonon_alpha_ewald = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        ''')

    x_qe_phonon_negative_rho = Quantity(
        type=np.dtype(np.float64),
        shape=[2],
        description='''
        ''')

    x_qe_phonon_n_parallel_sticks = SubSection(sub_section=x_qe_phonon_n_parallel.m_def)

    x_qe_phonon_n_parallel_g_vectors = SubSection(sub_section=x_qe_phonon_n_parallel.m_def)


class System(simulation.system.System):

    m_def = Section(validate=False, extends_base_section=True)

    x_qe_phonon_bravais_lattice_index = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        ''')

    x_qe_phonon_lattice_parameter = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='m',
        description='''
        ''')

    x_qe_phonon_unit_cell_volume = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='m ** 3',
        description='''
        ''')

    x_qe_phonon_n_atoms_cell = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        ''')

    x_qe_phonon_n_atomic_sites = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        ''')

    x_qe_phonon_point_group = Quantity(
        type=str,
        shape=[],
        description='''
        ''')


class x_qe_phonon_scf_iteration(MSection):

    m_def = Section(validate=False)

    x_qe_phonon_fermi_energy_shift = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='joule',
        description='''
        ''')

    x_qe_phonon_iter = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        ''')

    x_qe_phonon_time = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='s',
        description='''
        ''')

    x_qe_phonon_total_cpu_time = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='s',
        description='''
        ''')

    x_qe_phonon_threshold = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        ''')

    x_qe_phonon_alpha_mix = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        ''')

    x_qe_phonon_ddv_scf_2 = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        ''')


class x_qe_phonon_representation(MSection):

    m_def = Section(validate=False)

    x_qe_phonon_number = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        ''')

    x_qe_phonon_converged = Quantity(
        type=bool,
        shape=[],
        description='''
        ''')

    x_qe_phonon_modes = Quantity(
        type=np.dtype(np.int32),
        shape=['*'],
        description='''
        ''')

    x_qe_phonon_scf_iteration = SubSection(sub_section=x_qe_phonon_scf_iteration.m_def, repeats=True)


class Calculation(simulation.calculation.Calculation):

    m_def = Section(validate=False, extends_base_section=True)

    x_qe_phonon_representation = SubSection(sub_section=x_qe_phonon_representation.m_def)


class AtomParameters(simulation.method.AtomParameters):

    m_def = Section(validate=False, extends_base_section=True)

    x_qe_phonon_file = Quantity(
        type=str,
        shape=[],
        description='''
        ''')

    x_qe_phonon_md5_check_sum = Quantity(
        type=str,
        shape=[],
        description='''
        ''')

    x_qe_phonon_type = Quantity(
        type=str,
        shape=[],
        description='''
        ''')

    x_qe_phonon_n_radial_grid_points = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        ''')

    x_qe_phonon_n_beta_functions = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        ''')

    x_qe_phonon_l = Quantity(
        type=np.dtype(np.int32),
        shape=['x_qe_phonon_n_beta_functions'],
        description='''
        ''')

    x_qe_phonon_n_q_coefficients = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        ''')

    x_qe_phonon_l = Quantity(
        type=np.dtype(np.float64),
        shape=['x_qe_phonon_n_q_coefficients'],
        description='''
        ''')
