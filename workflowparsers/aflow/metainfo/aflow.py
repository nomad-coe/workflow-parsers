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
    Datetime, MSection, MCategory, Category, Package, Quantity, Section, SubSection, SectionProxy,
    Reference
)

from nomad.datamodel.metainfo import simulation


m_package = Package()


class Run(simulation.run.Run):

    m_def = Section(validate=False, extends_base_section=True)

    x_aflow_aurl = Quantity(
        type=str,
        shape=[],
        description='''
        ''')

    x_aflow_auid = Quantity(
        type=str,
        shape=[],
        description='''
        ''')

    x_aflow_data_api = Quantity(
        type=str,
        shape=[],
        description='''
        ''')

    x_aflow_data_source = Quantity(
        type=str,
        shape=[],
        description='''
        ''')

    x_aflow_n_loop = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        ''')

    x_aflow_loop = Quantity(
        type=str,
        shape=['x_aflow_n_loop'],
        description='''
        ''')


class Method(simulation.method.Method):

    m_def = Section(validate=False, extends_base_section=True)

    x_aflow_code = Quantity(
        type=str,
        shape=[],
        description='''
        ''')

    x_aflow_species_pp = Quantity(
        type=str,
        shape=['x_aflow_nspecies'],
        description='''
        ''')

    x_aflow_n_dft_type = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        ''')

    x_aflow_dft_type = Quantity(
        type=str,
        shape=['x_aflow_n_dft_type'],
        description='''
        ''')

    x_aflow_dft_type = Quantity(
        type=str,
        shape=['x_aflow_n_dft_type'],
        description='''
        ''')

    x_aflow_species_pp_version = Quantity(
        type=str,
        shape=['x_aflow_nspecies'],
        description='''
        ''')

    x_aflow_species_pp_ZVAL = Quantity(
        type=np.dtype(np.int32),
        shape=['x_aflow_nspecies'],
        description='''
        ''')

    x_aflow_species_pp_AUID = Quantity(
        type=str,
        shape=['x_aflow_nspecies'],
        description='''
        ''')

    x_aflow_ldau_type = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        ''')

    x_aflow_ldau_l = Quantity(
        type=np.dtype(np.int32),
        shape=['x_aflow_nspecies'],
        description='''
        ''')

    x_aflow_ldau_u = Quantity(
        type=np.dtype(np.int32),
        shape=['x_aflow_nspecies'],
        description='''
        ''')

    x_aflow_ldau_j = Quantity(
        type=np.dtype(np.int32),
        shape=['x_aflow_nspecies'],
        description='''
        ''')

    x_aflow_valence_cell_iupac = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        ''')

    x_aflow_valence_cell_std = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        ''')

    x_aflow_energy_cutoff = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        ''')

    x_aflow_delta_electronic_energy_convergence = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        ''')

    x_aflow_delta_electronic_energy_threshold = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        ''')

    x_aflow_kpoints_relax = Quantity(
        type=np.dtype(np.int32),
        shape=[3],
        description='''
        ''')

    x_aflow_kpoints_static = Quantity(
        type=np.dtype(np.int32),
        shape=[3],
        description='''
        ''')

    x_aflow_n_kpoints_bands_path = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        ''')

    x_aflow_kpoints_bands_path = Quantity(
        type=str,
        shape=['x_aflow_n_kpoints_bands_path'],
        description='''
        ''')

    x_aflow_kpoints_bands_nkpts = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        ''')


class System(simulation.system.System):

    m_def = Section(validate=False, extends_base_section=True)

    x_aflow_compound = Quantity(
        type=str,
        shape=[],
        description='''
        ''')

    x_aflow_prototype = Quantity(
        type=str,
        shape=[],
        description='''
        ''')

    x_aflow_nspecies = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        ''')

    x_aflow_natoms = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        ''')

    x_aflow_natoms_orig = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        ''')

    x_aflow_composition = Quantity(
        type=np.dtype(np.int32),
        shape=['x_aflow_nspecies'],
        description='''
        ''')

    x_aflow_density = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        ''')

    x_aflow_density_orig = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        ''')

    x_aflow_scintillation_attenuation_length = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        ''')

    x_aflow_stoichiometry = Quantity(
        type=np.dtype(np.float64),
        shape=['x_aflow_nspecies'],
        description='''
        ''')

    x_aflow_species = Quantity(
        type=str,
        shape=['x_aflow_nspecies'],
        description='''
        ''')

    x_aflow_geometry = Quantity(
        type=np.dtype(np.float64),
        shape=[6],
        description='''
        ''')

    x_aflow_geometry_orig = Quantity(
        type=np.dtype(np.float64),
        shape=[6],
        description='''
        ''')

    x_aflow_volume_cell = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        ''')

    x_aflow_volume_atom = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        ''')

    x_aflow_volume_cell_orig = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        ''')

    x_aflow_volume_atom_orig = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        ''')

    x_aflow_n_sg = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        ''')

    x_aflow_sg = Quantity(
        type=str,
        shape=['x_aflow_n_sg'],
        description='''
        ''')

    x_aflow_sg2 = Quantity(
        type=str,
        shape=['x_aflow_n_sg'],
        description='''
        ''')

    x_aflow_spacegroup_orig = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        ''')

    x_aflow_spacegroup_relax = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        ''')

    x_aflow_Bravais_lattice_orig = Quantity(
        type=str,
        shape=[],
        description='''
        ''')

    x_aflow_lattice_variation_orig = Quantity(
        type=str,
        shape=[],
        description='''
        ''')

    x_aflow_lattice_system_orig = Quantity(
        type=str,
        shape=[],
        description='''
        ''')

    x_aflow_Pearson_symbol_orig = Quantity(
        type=str,
        shape=[],
        description='''
        ''')

    x_aflow_Bravais_lattice_relax = Quantity(
        type=str,
        shape=[],
        description='''
        ''')

    x_aflow_lattice_variation_relax = Quantity(
        type=str,
        shape=[],
        description='''
        ''')

    x_aflow_lattice_system_relax = Quantity(
        type=str,
        shape=[],
        description='''
        ''')

    x_aflow_Pearson_symbol_relax = Quantity(
        type=str,
        shape=[],
        description='''
        ''')

    x_aflow_crystal_family_orig = Quantity(
        type=str,
        shape=[],
        description='''
        ''')

    x_aflow_crystal_system_orig = Quantity(
        type=str,
        shape=[],
        description='''
        ''')

    x_aflow_crystal_class_orig = Quantity(
        type=str,
        shape=[],
        description='''
        ''')

    x_aflow_point_group_Hermann_Mauguin_orig = Quantity(
        type=str,
        shape=[],
        description='''
        ''')

    x_aflow_point_group_Schoenflies_orig = Quantity(
        type=str,
        shape=[],
        description='''
        ''')

    x_aflow_point_group_orbifold_orig = Quantity(
        type=str,
        shape=[],
        description='''
        ''')

    x_aflow_point_group_type_orig = Quantity(
        type=str,
        shape=[],
        description='''
        ''')

    x_aflow_point_group_order_orig = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        ''')

    x_aflow_point_group_structure_orig = Quantity(
        type=str,
        shape=[],
        description='''
        ''')

    x_aflow_Bravais_lattice_lattice_type_orig = Quantity(
        type=str,
        shape=[],
        description='''
        ''')

    x_aflow_Bravais_lattice_lattice_variation_type_orig = Quantity(
        type=str,
        shape=[],
        description='''
        ''')

    x_aflow_Bravais_lattice_lattice_system_orig = Quantity(
        type=str,
        shape=[],
        description='''
        ''')

    x_aflow_Bravais_superlattice_lattice_type_orig = Quantity(
        type=str,
        shape=[],
        description='''
        ''')

    x_aflow_Bravais_superlattice_lattice_variation_type_orig = Quantity(
        type=str,
        shape=[],
        description='''
        ''')

    x_aflow_Bravais_superlattice_lattice_system_orig = Quantity(
        type=str,
        shape=[],
        description='''
        ''')

    x_aflow_Pearson_symbol_superlattice_orig = Quantity(
        type=str,
        shape=[],
        description='''
        ''')

    x_aflow_reciprocal_geometry_orig = Quantity(
        type=np.dtype(np.float64),
        shape=[6],
        description='''
        ''')

    x_aflow_reciprocal_volume_cell_orig = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        ''')

    x_aflow_reciprocal_lattice_type_orig = Quantity(
        type=str,
        shape=[],
        description='''
        ''')

    x_aflow_reciprocal_lattice_variation_type_orig = Quantity(
        type=str,
        shape=[],
        description='''
        ''')

    x_aflow_n_symmetries = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        ''')

    x_aflow_Wyckoff_letters_orig = Quantity(
        type=np.dtype(np.int32),
        shape=['x_aflow_nspecies', 'x_aflow_n_symmetries'],
        description='''
        ''')

    x_aflow_Wyckoff_multiplicities_orig = Quantity(
        type=np.dtype(np.int32),
        shape=['x_aflow_nspecies', 'x_aflow_n_symmetries'],
        description='''
        ''')

    x_aflow_Wyckoff_site_symmetries_orig = Quantity(
        type=np.dtype(np.int32),
        shape=['x_aflow_nspecies', 'x_aflow_n_symmetries'],
        description='''
        ''')

    x_aflow_crystal_family = Quantity(
        type=str,
        shape=[],
        description='''
        ''')

    x_aflow_crystal_system = Quantity(
        type=str,
        shape=[],
        description='''
        ''')

    x_aflow_crystal_class = Quantity(
        type=str,
        shape=[],
        description='''
        ''')

    x_aflow_point_group_Hermann_Mauguin = Quantity(
        type=str,
        shape=[],
        description='''
        ''')

    x_aflow_point_group_Schoenflies = Quantity(
        type=str,
        shape=[],
        description='''
        ''')

    x_aflow_point_group_orbifold = Quantity(
        type=str,
        shape=[],
        description='''
        ''')

    x_aflow_point_group_type = Quantity(
        type=str,
        shape=[],
        description='''
        ''')

    x_aflow_point_group_order = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        ''')

    x_aflow_point_group_structure = Quantity(
        type=str,
        shape=[],
        description='''
        ''')

    x_aflow_Bravais_lattice_lattice_type = Quantity(
        type=str,
        shape=[],
        description='''
        ''')

    x_aflow_Bravais_lattice_lattice_variation_type = Quantity(
        type=str,
        shape=[],
        description='''
        ''')

    x_aflow_Bravais_lattice_lattice_system = Quantity(
        type=str,
        shape=[],
        description='''
        ''')

    x_aflow_Bravais_superlattice_lattice_type = Quantity(
        type=str,
        shape=[],
        description='''
        ''')

    x_aflow_Bravais_superlattice_lattice_variation_type = Quantity(
        type=str,
        shape=[],
        description='''
        ''')

    x_aflow_Bravais_superlattice_lattice_system = Quantity(
        type=str,
        shape=[],
        description='''
        ''')

    x_aflow_Pearson_symbol_superlattice = Quantity(
        type=str,
        shape=[],
        description='''
        ''')

    x_aflow_reciprocal_geometry = Quantity(
        type=np.dtype(np.float64),
        shape=[6],
        description='''
        ''')

    x_aflow_reciprocal_volume_cell = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        ''')

    x_aflow_reciprocal_lattice_type = Quantity(
        type=str,
        shape=[],
        description='''
        ''')

    x_aflow_reciprocal_lattice_variation_type = Quantity(
        type=str,
        shape=[],
        description='''
        ''')

    x_aflow_Wyckoff_letters = Quantity(
        type=np.dtype(np.int32),
        shape=['x_aflow_nspecies', 1],
        description='''
        ''')

    x_aflow_Wyckoff_multiplicities = Quantity(
        type=np.dtype(np.int32),
        shape=['x_aflow_nspecies', 1],
        description='''
        ''')

    x_aflow_Wyckoff_site_symmetries = Quantity(
        type=np.dtype(np.int32),
        shape=['x_aflow_nspecies', 1],
        description='''
        ''')

    x_aflow_prototype_label_orig = Quantity(
        type=str,
        shape=[],
        description='''
        ''')

    x_aflow_prototype_params_list_orig = Quantity(
        type=str,
        shape=[3],
        description='''
        ''')

    x_aflow_prototype_params_values_orig = Quantity(
        type=np.dtype(np.float64),
        shape=[3],
        description='''
        ''')

    x_aflow_prototype_label_relax = Quantity(
        type=str,
        shape=[],
        description='''
        ''')

    x_aflow_prototype_params_list_relax = Quantity(
        type=str,
        shape=[3],
        description='''
        ''')

    x_aflow_prototype_params_values_relax = Quantity(
        type=np.dtype(np.float64),
        shape=[3],
        description='''
        ''')


class Calculation(simulation.calculation.Calculation):

    m_def = Section(validate=False, extends_base_section=True)

    x_aflow_stress_tensor = Quantity(
        type=np.dtype(np.float64),
        shape=[9],
        description='''
        ''')

    x_aflow_pressure_residual = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        ''')

    x_aflow_Pulay_stress = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        ''')

    x_aflow_Egap = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        ''')

    x_aflow_Egap_fit = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        ''')

    x_aflow_Egap_type = Quantity(
        type=str,
        shape=[],
        description='''
        ''')

    x_aflow_enthalpy_formation_cell = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='joule',
        description='''
        ''')

    x_aflow_entropic_temperature = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        ''')

    x_aflow_PV = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        ''')

    x_aflow_spin_cell = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        ''')

    x_aflow_spinD = Quantity(
        type=np.dtype(np.float64),
        shape=['x_aflow_natoms'],
        description='''
        ''')

    x_aflow_spinF = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        ''')

    x_aflow_calculation_memory = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        ''')

    x_aflow_calculation_cores = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        ''')

    x_aflow_nbondxx = Quantity(
        type=np.dtype(np.float64),
        shape=[6],
        description='''
        ''')

    x_aflow_agl_thermal_conductivity_300K = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        ''')

    x_aflow_agl_debye = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        ''')

    x_aflow_agl_acoustic_debye = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        ''')

    x_aflow_agl_gruneisen = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        ''')

    x_aflow_agl_heat_capacity_Cv_300K = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        ''')

    x_aflow_agl_heat_capacity_Cp_300K = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        ''')

    x_aflow_agl_thermal_expansion_300K = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        ''')

    x_aflow_agl_bulk_modulus_static_300K = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        ''')

    x_aflow_agl_bulk_modulus_isothermal_300K = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        ''')

    x_aflow_agl_poisson_ratio_source = Quantity(
        type=str,
        shape=[],
        description='''
        ''')

    x_aflow_agl_vibrational_free_energy_300K_cell = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        ''')

    x_aflow_agl_vibrational_free_energy_300K_atom = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        ''')

    x_aflow_agl_vibrational_entropy_300K_cell = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        ''')

    x_aflow_agl_vibrational_entropy_300K_atom = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        ''')

    x_aflow_ael_poisson_ratio = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        ''')

    x_aflow_ael_bulk_modulus_voigt = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        ''')

    x_aflow_ael_bulk_modulus_reuss = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        ''')

    x_aflow_ael_shear_modulus_voigt = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        ''')

    x_aflow_ael_shear_modulus_reuss = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        ''')

    x_aflow_ael_bulk_modulus_vrh = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        ''')

    x_aflow_ael_shear_modulus_vrh = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        ''')

    x_aflow_ael_elastic_anisotropy = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        ''')

    x_aflow_ael_youngs_modulus_vrh = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        ''')

    x_aflow_ael_speed_sound_transverse = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        ''')

    x_aflow_ael_speed_sound_longitudinal = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        ''')

    x_aflow_ael_speed_sound_average = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        ''')

    x_aflow_ael_pughs_modulus_ratio = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        ''')

    x_aflow_ael_debye_temperature = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        ''')

    x_aflow_ael_applied_pressure = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        ''')

    x_aflow_ael_average_external_pressure = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        ''')

    x_aflow_ael_stiffness_tensor = Quantity(
        type=np.dtype(np.float64),
        shape=[6, 6],
        description='''
        ''')

    x_aflow_ael_compliance_tensor = Quantity(
        type=np.dtype(np.float64),
        shape=[6, 6],
        description='''
        ''')

    x_aflow_bader_net_charges = Quantity(
        type=np.dtype(np.float64),
        shape=['x_aflow_natoms'],
        description='''
        ''')

    x_aflow_bader_atomic_volumes = Quantity(
        type=np.dtype(np.float64),
        shape=['x_aflow_natoms'],
        description='''
        ''')

    x_aflow_n_files = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        ''')

    x_aflow_files = Quantity(
        type=str,
        shape=['x_aflow_n_files'],
        description='''
        ''')

    x_aflow_node_CPU_Model = Quantity(
        type=str,
        shape=[],
        description='''
        ''')

    x_aflow_node_CPU_Cores = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        ''')

    x_aflow_node_CPU_MHz = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        ''')

    x_aflow_node_RAM_GB = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        ''')

    x_aflow_catalog = Quantity(
        type=str,
        shape=[],
        description='''
        ''')

    x_aflow_aflowlib_version = Quantity(
        type=str,
        shape=[],
        description='''
        ''')

    x_aflow_aflowlib_date = Quantity(
        type=str,
        shape=['x_aflow_nspecies'],
        description='''
        ''')
