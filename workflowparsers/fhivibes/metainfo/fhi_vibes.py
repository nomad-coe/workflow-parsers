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
    Reference, JSON
)

from nomad.datamodel.metainfo import simulation


m_package = Package()


class x_fhi_vibes_method(MCategory):
    '''
    Parameters from vibes metadata belonging to section method.
    '''

    m_def = Category()


class x_fhi_vibes_section_calculator_parameters(MSection):
    '''
    Calculator parameters
    '''

    m_def = Section(validate=False)

    x_fhi_vibes_calculator_parameters_xc = Quantity(
        type=str,
        shape=[],
        description='''
        -
        ''')

    x_fhi_vibes_calculator_parameters_k_grid = Quantity(
        type=np.dtype(np.int32),
        shape=[3],
        description='''
        -
        ''')

    x_fhi_vibes_calculator_parameters_sc_accuracy_rho = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        -
        ''')

    x_fhi_vibes_calculator_parameters_relativistic = Quantity(
        type=str,
        shape=[],
        description='''
        -
        ''')

    x_fhi_vibes_calculator_parameters_compensate_multipole_errors = Quantity(
        type=bool,
        shape=[],
        description='''
        -
        ''')

    x_fhi_vibes_calculator_parameters_output_level = Quantity(
        type=str,
        shape=[],
        description='''
        -
        ''')

    x_fhi_vibes_calculator_parameters_compute_forces = Quantity(
        type=bool,
        shape=[],
        description='''
        -
        ''')

    x_fhi_vibes_calculator_parameters_compute_heat_flux = Quantity(
        type=bool,
        shape=[],
        description='''
        -
        ''')

    x_fhi_vibes_calculator_parameters_use_pimd_wrapper = Quantity(
        type=str,
        shape=[],
        description='''
        -
        ''')

    x_fhi_vibes_calculator_parameters_species_dir = Quantity(
        type=str,
        shape=[],
        description='''
        -
        ''')


class x_fhi_vibes_section_calculator(MSection):
    '''
    Calculator parameters
    '''

    m_def = Section(validate=False)

    x_fhi_vibes_calculator = Quantity(
        type=str,
        shape=[],
        description='''
        -
        ''')

    x_fhi_vibes_section_calculator_parameters = SubSection(
        sub_section=SectionProxy('x_fhi_vibes_section_calculator_parameters'),
        repeats=True)


class x_fhi_vibes_section_atoms(MSection):
    '''
    Calculator parameters
    '''

    m_def = Section(validate=False)

    x_fhi_vibes_atoms_kind = Quantity(
        type=str,
        shape=[],
        description='''
        -
        ''')

    x_fhi_vibes_atoms_natoms = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        -
        ''')

    x_fhi_vibes_atoms_pbc = Quantity(
        type=bool,
        shape=[3],
        description='''
        -
        ''')

    x_fhi_vibes_atoms_cell = Quantity(
        type=np.dtype(np.float64),
        unit='meter',
        shape=[3, 3],
        description='''
        -
        ''')

    x_fhi_vibes_atoms_positions = Quantity(
        type=np.dtype(np.float64),
        unit='meter',
        shape=['x_fhi_vibes_atoms_natoms', 3],
        description='''
        -
        ''')

    x_fhi_vibes_atoms_velocities = Quantity(
        type=np.dtype(np.float64),
        unit='meter / second',
        shape=['x_fhi_vibes_atoms_natoms', 3],
        description='''
        -
        ''')

    x_fhi_vibes_atoms_symbols = Quantity(
        type=str,
        shape=['x_fhi_vibes_atoms_natoms'],
        description='''
        -
        ''')

    x_fhi_vibes_atoms_masses = Quantity(
        type=np.dtype(np.float64),
        unit='kilogram',
        shape=['x_fhi_vibes_atoms_natoms'],
        description='''
        -
        ''')

    x_fhi_vibes_atoms_info = Quantity(
        type=JSON,
        shape=[],
        description='''
        -
        ''')


class x_fhi_vibes_section_MD(MSection):
    '''
    Molecular dynamics parameters
    '''

    m_def = Section(validate=False)

    x_fhi_vibes_MD_type = Quantity(
        type=str,
        shape=[],
        description='''
        -
        ''')

    x_fhi_vibes_MD_md_type = Quantity(
        type=str,
        shape=[],
        description='''
        -
        ''')

    x_fhi_vibes_MD_temperature = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        -
        ''')

    x_fhi_vibes_MD_friction = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        -
        ''')

    x_fhi_vibes_MD_fix_cm = Quantity(
        type=bool,
        shape=[],
        description='''
        -
        ''')

    x_fhi_vibes_MD_fs = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        -
        ''')

    x_fhi_vibes_MD_kB = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        -
        ''')

    x_fhi_vibes_MD_dt = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        -
        ''')

    x_fhi_vibes_MD_kg = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        -
        ''')


class x_fhi_vibes_section_relaxation_kwargs(MSection):
    '''
    Relaxation kwargs
    '''

    m_def = Section(validate=False)

    x_fhi_vibes_relaxation_kwargs_maxstep = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        -
        ''')

    x_fhi_vibes_relaxation_kwargs_restart = Quantity(
        type=str,
        shape=[],
        description='''
        -
        ''')


class x_fhi_vibes_section_relaxation(MSection):
    '''
    Relaxation parameters
    '''

    m_def = Section(validate=False)

    x_fhi_vibes_relaxation_type = Quantity(
        type=str,
        shape=[],
        description='''
        -
        ''')

    x_fhi_vibes_relaxation_optimizer = Quantity(
        type=str,
        shape=[],
        description='''
        -
        ''')

    x_fhi_vibes_relaxation_maxstep = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        -
        ''')

    x_fhi_vibes_relaxation_driver = Quantity(
        type=str,
        shape=[],
        description='''
        -
        ''')

    x_fhi_vibes_relaxation_fmax = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        -
        ''')

    x_fhi_vibes_relaxation_unit_cell = Quantity(
        type=bool,
        shape=[],
        description='''
        -
        ''')

    x_fhi_vibes_relaxation_fix_symmetry = Quantity(
        type=bool,
        shape=[],
        description='''
        -
        ''')

    x_fhi_vibes_relaxation_hydrostatic_strain = Quantity(
        type=bool,
        shape=[],
        description='''
        -
        ''')

    x_fhi_vibes_relaxation_constant_volume = Quantity(
        type=bool,
        shape=[],
        description='''
        -
        ''')

    x_fhi_vibes_relaxation_scalar_pressure = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        -
        ''')

    x_fhi_vibes_relaxation_decimals = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        -
        ''')

    x_fhi_vibes_relaxation_symprec = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        -
        ''')

    x_fhi_vibes_relaxation_workdir = Quantity(
        type=str,
        shape=[],
        description='''
        -
        ''')

    x_fhi_vibes_section_relaxation_kwargs = SubSection(
        sub_section=SectionProxy('x_fhi_vibes_section_relaxation_kwargs'),
        repeats=True)


class x_fhi_vibes_section_phonopy(MSection):
    '''
    Phonony parameters
    '''

    m_def = Section(validate=False)

    x_fhi_vibes_phonopy_version = Quantity(
        type=str,
        shape=[],
        description='''
        -
        ''')

    x_fhi_vibes_phonopy_supercell_matrix = Quantity(
        type=np.dtype(np.int32),
        shape=[3, 3],
        description='''
        -
        ''')

    x_fhi_vibes_phonopy_symprec = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        -
        ''')

    x_fhi_vibes_phonopy_displacement_dataset = Quantity(
        type=JSON,
        shape=[],
        description='''
        -
        ''')

    x_fhi_vibes_section_phonopy_primitive = SubSection(
        sub_section=SectionProxy('x_fhi_vibes_section_atoms'),
        repeats=True)


class x_fhi_vibes_section_vibes(MSection):
    '''
    Vibes specifications
    '''

    m_def = Section(validate=False)

    x_fhi_vibes_version = Quantity(
        type=str,
        shape=[],
        description='''
        -
        ''')


class x_fhi_vibes_section_settings(MSection):
    '''
    Metadata settings
    '''

    m_def = Section(validate=False)

    x_fhi_vibes_settings_common = Quantity(
        type=JSON,
        shape=[],
        description='''
        -
        ''')

    x_fhi_vibes_setttings_machine = Quantity(
        type=JSON,
        shape=[],
        description='''
        -
        ''')

    x_fhi_vibes_settings_calculator = Quantity(
        type=JSON,
        shape=[],
        description='''
        -
        ''')

    x_fhi_vibes_settings_files = Quantity(
        type=JSON,
        shape=[],
        description='''
        -
        ''')


class x_fhi_vibes_section_metadata(MSection):
    '''
    Metadata
    '''

    m_def = Section(validate=False)

    x_fhi_vibes_section_metadata_calculator = SubSection(
        sub_section=SectionProxy('x_fhi_vibes_section_calculator'),
        repeats=True)

    x_fhi_vibes_section_metadata_MD = SubSection(
        sub_section=SectionProxy('x_fhi_vibes_section_MD'),
        repeats=True)

    x_fhi_vibes_section_metadata_relaxation = SubSection(
        sub_section=SectionProxy('x_fhi_vibes_section_relaxation'),
        repeats=True)

    x_fhi_vibes_section_metadata_phonopy = SubSection(
        sub_section=SectionProxy('x_fhi_vibes_section_phonopy'),
        repeats=True)

    x_fhi_vibes_section_metadata_atoms = SubSection(
        sub_section=SectionProxy('x_fhi_vibes_section_atoms'),
        repeats=True)

    x_fhi_vibes_section_metadata_settings = SubSection(
        sub_section=SectionProxy('x_fhi_vibes_section_settings'),
        repeats=True)

    x_fhi_vibes_section_metadata_vibes = SubSection(
        sub_section=SectionProxy('x_fhi_vibes_section_vibes'),
        repeats=True)


class x_fhi_vibes_section_attributes(MSection):
    '''
    Dataset attributes
    '''

    m_def = Section(validate=False)

    x_fhi_vibes_attributes_name = Quantity(
        type=str,
        shape=[],
        description='''
        -
        ''')

    x_fhi_vibes_attributes_system_name = Quantity(
        type=str,
        shape=[],
        description='''
        -
        ''')

    x_fhi_vibes_attributes_natoms = Quantity(
        type=np.dtype(np.int32),
        shape=[3],
        description='''
        -
        ''')

    x_fhi_vibes_attributes_time_unit = Quantity(
        type=str,
        shape=[],
        description='''
        -
        ''')

    x_fhi_vibes_attributes_timestep = Quantity(
        type=np.dtype(np.float64),
        unit='second',
        shape=[],
        description='''
        -
        ''')

    x_fhi_vibes_attributes_nsteps = Quantity(
        type=np.dtype(np.int32),
        shape=[3],
        description='''
        -
        ''')

    x_fhi_vibes_attributes_symbols = Quantity(
        type=str,
        shape=['x_fhi_vibes_attributes_system_natoms'],
        description='''
        -
        ''')

    x_fhi_vibes_attributes_masses = Quantity(
        type=np.dtype(np.float64),
        unit='kilogram',
        shape=['x_fhi_vibes_attributes_system_natoms'],
        description='''
        -
        ''')

    x_fhi_vibes_attributes_hash = Quantity(
        type=str,
        shape=[],
        description='''
        -
        ''')

    x_fhi_vibes_attributes_hash_raw = Quantity(
        type=str,
        shape=[],
        description='''
        -
        ''')

    x_fhi_vibes_attributes_volume = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        -
        ''')

    x_fhi_vibes_attributes_sigma = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        -
        ''')

    x_fhi_vibes_attributes_st_size = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        -
        ''')

    x_fhi_vibes_attributes_number_of_atoms_primitive = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        -
        ''')

    x_fhi_vibes_attributes_number_of_atoms_supercell = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        -
        ''')

    x_fhi_vibes_attributes_force_constants_remapped_size = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        -
        ''')

    x_fhi_vibes_section_attributes_atoms = SubSection(
        sub_section=SectionProxy('x_fhi_vibes_section_atoms'),
        repeats=True)

    x_fhi_vibes_section_attributes_metadata = SubSection(
        sub_section=SectionProxy('x_fhi_vibes_section_metadata'),
        repeats=True)


class Method(simulation.method.Method):

    m_def = Section(validate=False, extends_base_section=True)

    x_fhi_vibes_section_attributes = SubSection(
        sub_section=SectionProxy('x_fhi_vibes_section_attributes'),
        repeats=True)


class Calculation(simulation.calculation.Calculation):

    m_def = Section(validate=False, extends_base_section=True)

    x_fhi_vibes_volume = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='meter**3',
        description='''
        -
        ''')

    x_fhi_vibes_displacements = Quantity(
        type=np.dtype(np.float64),
        shape=['number_of_atoms', 3, 3],
        unit='meter',
        description='''
        -
        ''')

    x_fhi_vibes_momenta = Quantity(
        type=np.dtype(np.float64),
        shape=['number_of_atoms', 3, 3],
        unit='kilogram * meter / second',
        description='''
        -
        ''')

    x_fhi_vibes_force_constants = Quantity(
        type=np.dtype(np.float64),
        shape=['x_fhi_vibes_attributes_number_of_atoms_primitive', 'x_fhi_vibes_attributes_number_of_atoms_primitive', 3, 3],
        unit='newton / meter',
        description='''
        -
        ''')

    x_fhi_vibes_force_constants_remapped = Quantity(
        type=np.dtype(np.float64),
        shape=['x_fhi_vibes_attributes_force_constants_remapped_size', 'x_fhi_vibes_attributes_force_constants_remapped_size'],
        unit='newton / meter',
        description='''
        -
        ''')

    x_fhi_vibes_atom_forces_harmonic = Quantity(
        type=np.dtype(np.float64),
        shape=['number_of_atoms', 3],
        unit='newton',
        description='''
        -
        ''')

    x_fhi_vibes_energy_potential_harmonic = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='joule',
        description='''
        -
        ''')

    x_fhi_vibes_heat_flux = Quantity(
        type=np.dtype(np.float64),
        shape=[3],
        unit='kilogram / second**3',
        description='''
        -
        ''')

    x_fhi_vibes_heat_flux_harmonic = Quantity(
        type=np.dtype(np.float64),
        shape=[3],
        unit='kilogram / second**3',
        description='''
        -
        ''')

    x_fhi_vibes_heat_flux_0_harmonic = Quantity(
        type=np.dtype(np.float64),
        shape=[3],
        unit='kilogram / second**3',
        description='''
        -
        ''')

    x_fhi_vibes_sigma_per_sample = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        -
        ''')

    x_fhi_vibes_pressure_kinetic = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='pascal',
        description='''
        -
        ''')

    x_fhi_vibes_pressure_potential = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='pascal',
        description='''
        -
        ''')

    # TODO in fhi aims this is raw id, however in vibes, where each calculation corresponds
    # to an scc, we get a list of these ids, it does not make sense to put these in sec_run
    x_fhi_vibes_aims_uuid = Quantity(
        type=str,
        shape=[],
        description='''
        -
        ''')

    x_fhi_vibes_MD_time = Quantity(
        type=np.dtype(np.float64),
        unit='second',
        shape=[],
        description='''
        -
        ''')
