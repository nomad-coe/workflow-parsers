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
import typing                 # pylint: disable=unused-import
from nomad.metainfo import (  # pylint: disable=unused-import
    MSection, MCategory, Category, Package, Quantity, Section, SubSection, SectionProxy,
    Reference
)

from nomad.datamodel.metainfo import simulation


m_package = Package()


class CalcTypes(MSection):

    m_def = Section(validate=False)

    x_mp_label = Quantity(
        type=str,
        shape=[],
        description='''
        ''')

    x_mp_value = Quantity(
        type=str,
        shape=[],
        description='''
        ''')


class Origins(MSection):

    m_def = Section(validate=False)

    x_mp_name = Quantity(
        type=str,
        shape=[],
        description='''
        ''')

    x_mp_task_id = Quantity(
        type=str,
        shape=[],
        description='''
        ''')

    x_mp_last_updated = Quantity(
        type=str,
        shape=[],
        description='''
        ''')


class Run(simulation.run.Run):

    m_def = Section(validate=False, extends_base_section=True)

    x_mp_emmet_version = Quantity(
        type=str,
        shape=[],
        description='''
        ''')

    x_mp_pymatgen_version = Quantity(
        type=str,
        shape=[],
        description='''
        ''')

    x_mp_build_date = Quantity(
        type=str,
        shape=[],
        description='''
        ''')

    x_mp_cif = Quantity(
        type=str,
        shape=[],
        description='''
        ''')

    x_mp_n_tags = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        ''')

    x_mp_tags = Quantity(
        type=str,
        shape=['x_mp_n_tags'],
        description='''
        ''')

    x_mp_material_id = Quantity(
        type=str,
        shape=[],
        description='''
        ''')

    x_mp_deprecated = Quantity(
        type=str,
        shape=[],
        description='''
        ''')

    x_mp_n_tasks = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        ''')

    x_mp_task_ids = Quantity(
        type=str,
        shape=['x_mp_n_tasks'],
        description='''
        ''')

    x_mp_icsd_id = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        ''')

    x_mp_icsd_ids = Quantity(
        type=np.dtype(np.int32),
        shape=['x_mp_n_tasks'],
        description='''
        ''')

    x_mp_last_updated = Quantity(
        type=str,
        shape=[],
        description='''
        ''')

    x_mp_created_at = Quantity(
        type=str,
        shape=[],
        description='''
        ''')

    x_mp_calc_types = SubSection(sub_section=CalcTypes.m_def, repeats=True)

    x_mp_origins = SubSection(sub_section=Origins.m_def, repeats=True)


class Composition(MSection):

    m_def = Section(validate=False)

    x_mp_label = Quantity(
        type=str,
        shape=[],
        description='''
        ''')

    x_mp_value = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        ''')


class Symmetry(MSection):

    m_def = Section(validate=False)

    x_mp_symprec = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        ''')

    x_mp_version = Quantity(
        type=str,
        shape=[],
        description='''
        ''')

    x_mp_source = Quantity(
        type=str,
        shape=[],
        description='''
        ''')

    x_mp_symbol = Quantity(
        type=str,
        shape=[],
        description='''
        ''')

    x_mp_number = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        ''')

    x_mp_point_group = Quantity(
        type=str,
        shape=[],
        description='''
        ''')

    x_mp_crystal_system = Quantity(
        type=str,
        shape=[],
        description='''
        ''')

    x_mp_hall = Quantity(
        type=str,
        shape=[],
        description='''
        ''')


class System(simulation.system.System):

    m_def = Section(validate=False, extends_base_section=True)

    x_mp_formula_anonymous = Quantity(
        type=str,
        shape=[],
        description='''
        ''')

    x_mp_oxide_type = Quantity(
        type=str,
        shape=[],
        description='''
        ''')

    x_mp_chemsys = Quantity(
        type=str,
        shape=[],
        description='''
        ''')

    x_mp_formula_pretty = Quantity(
        type=str,
        shape=[],
        description='''
        ''')

    x_mp_volume = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        ''')

    x_mp_density = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        ''')

    x_mp_density_atomic = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        ''')

    x_mp_nelements = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        ''')

    x_mp_elements = Quantity(
        type=str,
        shape=['x_mp_nelements'],
        description='''
        ''')

    x_mp_nsites = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        ''')

    x_mp_symmetry = SubSection(sub_section=Symmetry.m_def, repeats=True)

    x_mp_composition = SubSection(sub_section=Composition.m_def, repeats=True)

    x_mp_composition_reduced = SubSection(sub_section=Composition.m_def, repeats=True)


class Hubbard(MSection):

    m_def = Section(validate=False)

    x_mp_element = Quantity(
        type=str,
        shape=[],
        description='''
        ''')

    x_mp_hubbard = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        ''')


class Method(simulation.method.Method):

    m_def = Section(validate=False, extends_base_section=True)

    x_mp_is_compatible = Quantity(
        type=bool,
        shape=[],
        description='''
        ''')

    x_mp_is_hubbard = Quantity(
        type=bool,
        shape=[],
        description='''
        ''')

    x_mp_hubbards = SubSection(sub_section=Hubbard.m_def, repeats=True)


class Calculation(simulation.calculation.Calculation):

    m_def = Section(validate=False, extends_base_section=True)

    x_mp_uncorrected_energy_per_atom = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        ''')
