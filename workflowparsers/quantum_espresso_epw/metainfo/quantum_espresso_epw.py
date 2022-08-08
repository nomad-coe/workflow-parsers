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


class x_qe_epw_irreducible_q_point(MSection):

    m_def = Section(validate=False)

    x_qe_epw_n_symmetries = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        ''')

    x_qe_epw_n_q_star = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        ''')

    x_qe_epw_q_star = Quantity(
        type=np.dtype(np.int32),
        shape=['x_qe_epw_n_q_star', 3],
        description='''
        ''')


class System(simulation.system.System):

    m_def = Section(validate=False, extends_base_section=True)

    x_qe_epw_bravais_lattice_index = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        ''')

    x_qe_epw_lattice_parameter = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='m',
        description='''
        ''')

    x_qe_epw_unit_cell_volume = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='m ** 3',
        description='''
        ''')

    x_qe_epw_n_atoms_cell = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        ''')

    x_qe_epw_n_atomic_types = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        ''')

    x_qe_epw_irreducible_q_point = SubSection(sub_section=x_qe_epw_irreducible_q_point.m_def, repeats=True)


class Method(simulation.method.Method):

    m_def = Section(validate=False, extends_base_section=True)

    x_qe_epw_n_ws_vectors_electrons = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        ''')

    x_qe_epw_n_ws_vectors_phonons = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        ''')

    x_qe_epw_n_ws_vectors_electron_phonon = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        ''')

    x_qe_epw_n_max_cores = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        ''')

    x_qe_epw_use_ws = Quantity(
        type=bool,
        shape=[],
        description='''
        ''')

    x_qe_epw_q_mesh = Quantity(
        type=np.dtype(np.int32),
        shape=[3],
        description='''
        ''')

    x_qe_epw_n_q_mesh = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        ''')

    x_qe_epw_k_mesh = Quantity(
        type=np.dtype(np.int32),
        shape=[3],
        description='''
        ''')

    x_qe_epw_n_k_mesh = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        ''')

    x_qe_epw_n_max_kpoints_per_pool = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        ''')

    x_qe_epw_kinetic_energy_cutoff = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='joule',
        description='''
        ''')

    x_qe_epw_charge_density_cutoff = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='joule',
        description='''
        ''')

    x_qe_epw_convergence_threshold = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        ''')

    x_qe_epw_exchange_correlation = Quantity(
        type=str,
        shape=[],
        description='''
        ''')

    x_qe_epw_fft_g_cutoff = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        ''')

    x_qe_epw_fft_g_vectors = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        ''')

    x_qe_epw_fft_grid = Quantity(
        type=np.dtype(np.int32),
        shape=[3],
        description='''
        ''')

    x_qe_epw_smooth_g_cutoff = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        ''')

    x_qe_epw_smooth_g_vectors = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        ''')

    x_qe_epw_smooth_grid = Quantity(
        type=np.dtype(np.int32),
        shape=[3],
        description='''
        ''')

    x_qe_epw_n_kpoints = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        ''')

    x_qe_epw_gaussian_broadening = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='rydberg',
        description='''
        ''')

    x_qe_epw_n_gauss = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        ''')


class AtomParameters(simulation.method.AtomParameters):

    m_def = Section(validate=False, extends_base_section=True)

    x_qe_epw_file = Quantity(
        type=str,
        shape=[],
        description='''
        ''')

    x_qe_epw_md5_check_sum = Quantity(
        type=str,
        shape=[],
        description='''
        ''')

    x_qe_epw_type = Quantity(
        type=str,
        shape=[],
        description='''
        ''')

    x_qe_epw_n_radial_grid_points = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        ''')

    x_qe_epw_n_beta_functions = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        ''')

    x_qe_epw_l = Quantity(
        type=np.dtype(np.int32),
        shape=['x_qe_epw_n_beta_functions'],
        description='''
        ''')

    x_qe_epw_n_q_coefficients = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        ''')

    x_qe_epw_l = Quantity(
        type=np.dtype(np.float64),
        shape=['x_qe_epw_n_q_coefficients'],
        description='''
        ''')


class x_qe_epw_self_energy(MSection):

    m_def = Section(validate=False)

    x_qe_epw_ismear = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        ''')

    x_qe_epw_iq = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        ''')

    x_qe_epw_coord = Quantity(
        type=np.dtype(np.float64),
        shape=[3],
        description='''
        ''')

    x_qe_epw_wt = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        ''')

    x_qe_epw_temp = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='kelvin',
        description='''
        ''')

    x_qe_epw_n_lambda = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        ''')

    x_qe_epw_lambda = Quantity(
        type=np.dtype(np.float64),
        shape=['x_qe_epw_n_lambda'],
        description='''
        ''')

    x_qe_epw_gamma = Quantity(
        type=np.dtype(np.float64),
        shape=['x_qe_epw_n_lambda'],
        description='''
        ''')

    x_qe_epw_omega = Quantity(
        type=np.dtype(np.float64),
        shape=['x_qe_epw_n_lambda'],
        description='''
        ''')

    x_qe_epw_lambda_tr = Quantity(
        type=np.dtype(np.float64),
        shape=['x_qe_epw_n_lambda'],
        description='''
        ''')

    x_qe_epw_gamma_tr = Quantity(
        type=np.dtype(np.float64),
        shape=['x_qe_epw_n_lambda'],
        description='''
        ''')

    x_qe_epw_omega_tr = Quantity(
        type=np.dtype(np.float64),
        shape=['x_qe_epw_n_lambda'],
        description='''
        ''')

    x_qe_epw_lambda_tot = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        ''')

    x_qe_epw_lambda_tot_tr = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        ''')


class x_qe_epw_self_energy_migdal(MSection):

    m_def = Section(validate=False)

    x_qe_epw_fermi_surface_thickness = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='electron_volt',
        description='''
        ''')

    x_qe_epw_golden_rule_t = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='electron_volt',
        description='''
        ''')

    x_qe_epw_gaussian_broadening = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='electron_volt',
        description='''
        ''')

    x_qe_epw_n_gauss = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        ''')

    x_qe_epw_dos_ef = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        ''')

    x_qe_epw_self_energy = SubSection(sub_section=x_qe_epw_self_energy.m_def, repeats=True)


class x_qe_epw_timimg(MSection):

    m_def = Section(validate=False)

    x_qe_epw_task = Quantity(
        type=str,
        shape=[],
        description='''
        ''')

    x_qe_epw_cpu_time = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        ''')

    x_qe_epw_wall_time = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        ''')

    x_qe_epw_n_calls = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        ''')


class x_qe_epw_eliashberg_spectral_function_migdal_approximation(MSection):

    m_def = Section(validate=False)

    x_qe_epw_lambda = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        ''')

    x_qe_epw_lambda_tr = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        ''')

    x_qe_epw_logavg = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        ''')

    x_qe_epw_l_a2f = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        ''')

    x_qe_epw_mu_tc = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        ''')

    x_qe_epw_timimg = SubSection(sub_section=x_qe_epw_timimg.m_def, repeats=True)


class Calculation(simulation.calculation.Calculation):

    m_def = Section(validate=False, extends_base_section=True)

    x_qe_epw_e_fermi_coarse_grid = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='electron_volt',
        description='''
        ''')

    x_qe_epw_self_energy_migdal = SubSection(sub_section=x_qe_epw_self_energy_migdal.m_def)

    x_qe_epw_eliashberg_spectral_function_migdal_approximation = SubSection(sub_section=x_qe_epw_eliashberg_spectral_function_migdal_approximation.m_def)
