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


class x_mof_atoms(MSection):

    m_def = Section(validate=False)

    x_mof_atoms_experiment = SubSection(sub_section=simulation.system.Atoms)

    x_mof_atoms_optimised = SubSection(sub_section=simulation.system.Atoms)

    x_mof_sbu = SubSection(sub_section=simulation.system.AtomsGroup, repeats=True)

    x_mof_linker = SubSection(sub_section=simulation.system.AtomsGroup, repeats=True)

    x_mof_organic_sbu = SubSection(sub_section=simulation.system.AtomsGroup, repeats=True)


class System(simulation.system.System):

    m_def = Section(validate=False, extends_base_section=True)

    x_mof_refcode = Quantity(
        type=str,
        shape=[],
        description='CSD refcode of the MOF')

    x_mof_source = Quantity(
        type=str,
        shape=[],
        description='source of the MOF. Most are from CSD')

    x_mof_alias = Quantity(
        type=str,
        shape=[],
        description='nickname')

    x_mof_ccdc_number = Quantity(
        type=int,
        description='Unique ccdc number of the MOF')

    x_mof_csd_deposition_date = Quantity(
        type=str,
        description='Date on which the structure was deposited into CSD')

    x_mof_chemical_name = Quantity(
        type=str,
        shape=[],
        description='Chemical name of the MOF')

    x_mof_topology = Quantity(
        type=str,
        shape=[],
        description='Three letter topological symbol obtained from RSCR. Computed using MOFid python script')

    x_mof_pld = Quantity(
        type=np.dtype(np.float64),
        description='Local pore diameter')

    x_mof_lcd = Quantity(
        type=np.dtype(np.float64),
        description='Pore limiting diameter')

    x_mof_lfpd = Quantity(
        type=np.dtype(np.float64),
        description='Pore limiting diameter')

    x_mof_asa = Quantity(
        type=np.dtype(np.float64),
        units='meter**2/gram',
        description='Accessible surface area')

    x_mof_nasa = Quantity(
        type=np.dtype(np.float64),
        units='meter**2/gram',
        description='Non-accessible surface area')

    x_mof_density = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='Density of the MOF')

    x_mof_volume = Quantity(
        type=np.dtype(np.float64),
        units='Amstrong**3',
        shape=[],
        description='Volume of the MOF')

    x_mof_n_channel = Quantity(
        type=np.dtype(np.float64),
        units='Amstrong**3',
        shape=[],
        description='Number of channels present in the MOF')

    x_mof_space_group_symbol = Quantity(
        type=str,
        shape=[],
        description='Space group symbol computed from pymatgen')

    x_mof_space_group_number = Quantity(
        type=str,
        shape=[],
        description='Space_group_number computed from pymatgen')

    x_mof_point_group_symbol = Quantity(
        type=str,
        shape=[],
        description='Space_group_number computed from pymatgen')

    x_mof_crystal_system = Quantity(
        type=str,
        shape=[],
        description='Space_group_number computed from pymatgen')

    x_mof_hall_symbol = Quantity(
        type=str,
        shape=[],
        description='Space_group_number computed from pymatgen')

    x_mof_charge = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='Charge of system computed from pymatgen')

    x_mof_core_metal = Quantity(
        type=str,
        shape=[],
        description='Core_metals found in MOFs')

    x_mof_cn = Quantity(
        type=JSON,
        shape=[],
        description='Coordination metal of MOF')

    x_mof_is_rod = Quantity(
        type=bool,
        shape=[],
        description='Check to know whether the sbu of the MOF is rodlike. I.e. Whether it extends infinitely in one dimension')

    x_mof_is_paddlewheel = Quantity(
        type=bool,
        shape=[],
        description='Check to know whether the sbu of the MOF is paddlewheel.')

    x_mof_is_ferrocene = Quantity(
        type=bool,
        shape=[],
        description='Check to know whether the MOF is a ferrocene')

    x_mof_is_ui006 = Quantity(
        type=bool,
        shape=[],
        description='Check to know whether the sbu of the MOF is a UIO66 base.')

    x_mof_is_mof32 = Quantity(
        type=bool,
        shape=[],
        description='Check to know whether the sbu of the MOF is a MOF32 base.')

    x_mof_is_irmof = Quantity(
        type=bool,
        shape=[],
        description='Check to know whether the sbu of the MOF is a IRMOF (MOF-5) base.')

    x_mof_synthesis_method = Quantity(
        type=str,
        shape=[],
        description='Experimental method use in synthesising the MOF')

    x_mof_linker_reagents = Quantity(
        type=str,
        shape=[''],
        description='Organic reagents')

    x_mof_metal_reagents = Quantity(
        type=str,
        shape=['list of reagents'],
        description='Metal reagents')

    x_mof_temperature = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        units='celsius',
        description='Temperature used in the synthesis of the MOF')

    x_mof_time_h = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        units='hours',
        description='Time of sysnthesis in hours')

    x_mof_yield_percent = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='Yield Percent')

    x_mof_doi = Quantity(
        type=str,
        shape=[],
        description='DOI of MOF')

    x_mof_citation = Quantity(
        type=str,
        shape=[],
        description='citation of reference')

    x_mof_metal_os_1 = Quantity(
        type=str,
        shape=[],
        description='Verification of whether the first metal has an open metal site')

    x_mof_counterions1 = Quantity(
        type=str,
        shape=[],
        description='First counter ions on the metal')

    x_mof_metal_os_2 = Quantity(
        type=str,
        shape=[],
        description='Verification of whether the second metal has an open metal site')

    x_mof_counterions2 = Quantity(
        type=str,
        shape=[],
        description='Second counter ions on the metal')

    x_mof_metal_os_3 = Quantity(
        type=str,
        shape=[],
        description='Verification of whether the third metal has an open metal site')

    x_mof_counterions3 = Quantity(
        type=str,
        shape=[],
        description='Third counter ions on the metal')

    x_mof_mof_solvent1 = Quantity(
        type=str,
        shape=[],
        description='Pubmed solvent number of second solvent')

    x_sol_molratio1 = Quantity(
        type=str,
        shape=[],
        description='Mole ratio of first solvent')

    x_mof_solvent2 = Quantity(
        type=str,
        shape=[],
        description='Pubmed solvent number of second solvent')

    x_mof_sol_molratio2 = Quantity(
        type=str,
        shape=[],
        description='Mole ratio of second solvent')

    x_mof_solvent3 = Quantity(
        type=str,
        shape=[],
        description='Pubmed solvent number of third solvent')

    x_mof_sol_molratio3 = Quantity(
        type=str,
        shape=[],
        description='Mole ratio of third solvent')

    x_mof_solvent4 = Quantity(
        type=str,
        shape=[],
        description='Pubmed solvent number of fourth solvent')

    x_mof_sol_molratio4 = Quantity(
        type=str,
        shape=[],
        description='Mole ratio of fourth solvent')

    x_mof_solvent5 = Quantity(
        type=str,
        shape=[],
        description='Pubmed solvent number of fifth solvent')

    x_mof_sol_molratio5 = Quantity(
        type=str,
        shape=[],
        description='Mole ratio of fifth solvent')

    x_mof_additive1 = Quantity(
        type=str,
        shape=[],
        description='First additive')

    x_mof_additive2 = Quantity(
        type=str,
        shape=[],
        description='Second additive')

    x_mof_mof_additive3 = Quantity(
        type=str,
        shape=[],
        description='Third additive')

    x_mof_additive4 = Quantity(
        type=str,
        shape=[],
        description='Fourth additive4=')

    x_mof_additive5 = Quantity(
        type=str,
        shape=[],
        description='Fifth additive ')

    x_mof_atoms = SubSection(sub_section=x_mof_atoms)


class Method(simulation.method.Method):

    m_def = Section(validate=False, extends_base_section=True)

    x_mof_metadata = Quantity(
        type=JSON,
        shape=[],
        description='''
        ''')
