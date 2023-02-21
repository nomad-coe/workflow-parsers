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
    MSection, Package, Quantity, Section, SubSection, JSON
)
from nomad.datamodel.metainfo import simulation


m_package = Package()


class x_qe_xspectra_input(MSection):

    m_def = Section(validate=False)

    x_qe_xspectra_calculation = Quantity(
        type=str,
        description='''
        ''')

    x_qe_xspectra_xepsilon = Quantity(
        type=np.float64,
        shape=[3],
        description='''
        ''')

    x_qe_xspectra_xonly_plot = Quantity(
        type=bool,
        description='''
        ''')

    x_qe_xspectra_filecore = Quantity(
        type=str,
        description='''
        ''')

    x_qe_xspectra_main_plot_parameters = Quantity(
        type=JSON,
        description='''
        ''')


class x_qe_xspectra_pwscf_wavefunction(MSection):

    m_def = Section(validate=False)

    x_qe_xspectra_file = Quantity(
        type=str,
        description='''
        ''')

    x_qe_xspectra_wavefunctions = Quantity(
        type=str,
        description='''
        ''')


class x_qe_xspectra_n_parallel(MSection):

    m_def = Section(validate=False)

    x_qe_xspectra_n_parallel_min = Quantity(
        type=np.dtype(np.float64),
        shape=[3],
        description='''
        ''')

    x_qe_xspectra_n_parallel_max = Quantity(
        type=np.dtype(np.float64),
        shape=[3],
        description='''
        ''')

    x_qe_xspectra_n_parallel_sum = Quantity(
        type=np.dtype(np.float64),
        shape=[3],
        description='''
        ''')


class Run(simulation.run.Run):

    m_def = Section(validate=False, extends_base_section=True)

    x_qe_xspectra_input = SubSection(sub_section=x_qe_xspectra_input.m_def)


class Method(simulation.method.Method):

    m_def = Section(validate=False, extends_base_section=True)

    x_qe_xspectra_save_directory = Quantity(
        type=str,
        description='''
        ''')
    x_qe_xspectra_pwscf_wavefunction = SubSection(sub_section=x_qe_xspectra_pwscf_wavefunction, repeats=True)

    x_qe_xspectra_n_parallel_sticks = SubSection(sub_section=x_qe_xspectra_n_parallel.m_def)

    x_qe_xspectra_n_parallel_g_vectors = SubSection(sub_section=x_qe_xspectra_n_parallel.m_def)


class System(simulation.system.System):

    m_def = Section(validate=False, extends_base_section=True)

    x_qe_xspectra_bravais_lattice_index = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        ''')

    x_qe_xspectra_lattice_parameter = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='m',
        description='''
        ''')

    x_qe_xspectra_unit_cell_volume = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='m ** 3',
        description='''
        ''')

    x_qe_xspectra_n_atoms_cell = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        ''')

    x_qe_xspectra_n_atomic_sites = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        ''')

    x_qe_xspectra_point_group = Quantity(
        type=str,
        shape=[],
        description='''
        ''')


class AtomParameters(simulation.method.AtomParameters):

    m_def = Section(validate=False, extends_base_section=True)

    x_qe_xspectra_file = Quantity(
        type=str,
        shape=[],
        description='''
        ''')

    x_qe_xspectra_md5_check_sum = Quantity(
        type=str,
        shape=[],
        description='''
        ''')

    x_qe_xspectra_type = Quantity(
        type=str,
        shape=[],
        description='''
        ''')

    x_qe_xspectra_n_radial_grid_points = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        ''')

    x_qe_xspectra_n_beta_functions = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        ''')

    x_qe_xspectra_l = Quantity(
        type=np.dtype(np.int32),
        shape=['x_qe_xspectra_n_beta_functions'],
        description='''
        ''')

    x_qe_xspectra_n_q_coefficients = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        ''')

    x_qe_xspectra_l = Quantity(
        type=np.dtype(np.float64),
        shape=['x_qe_xspectra_n_q_coefficients'],
        description='''
        ''')


class Spectra(simulation.calculation.Spectra):

    m_def = Section(validate=False, extends_base_section=True)

    x_qe_xspectra_energy_zero = Quantity(
        type=np.float64,
        unit='eV',
        description='''
        ''')

    x_qe_xspectra_xemin = Quantity(
        type=np.float64,
        unit='eV',
        description='''
        ''')

    x_qe_xspectra_xemax = Quantity(
        type=np.float64,
        unit='eV',
        description='''
        ''')

    x_qe_xspectra_xnepoint = Quantity(
        type=np.float64,
        description='''
        ''')

    x_qe_xspectra_broadening_parameter = Quantity(
        type=np.float64,
        description='''
        ''')

    x_qe_xspectra_energy_core_level = Quantity(
        type=np.float64,
        unit='eV',
        description='''
        ''')
