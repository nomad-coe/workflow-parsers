#
# Copyright The NOMAD Authors.
#
# This file is part of NOMAD. See https://nomad-lab.eu for further info.
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

import numpy as np

from nomad.metainfo import (
    Section, Quantity, MSection, SubSection, SectionProxy, Package
)
from nomad.datamodel.metainfo import simulation


m_package = Package()


class Calculation(simulation.calculation.Calculation):

    m_def = Section(validate=False, extends_base_section=True)

    x_lobster_abs_total_spilling = Quantity(
        type=float,
        shape=['number_of_spin_channels'],
        description='''
        Absolute total spilling (in all levels)
        when projecting from the original wave functions into the local basis.
        ''')

    x_lobster_abs_charge_spilling = Quantity(
        type=float,
        shape=['number_of_spin_channels'],
        description='''
        Absolute total spilling of density (in occupied levels)
        when projecting from the original wave functions into the local basis.
        ''')

    x_lobster_section_cohp = SubSection(
        sub_section=SectionProxy('x_lobster_section_cohp'))

    x_lobster_section_coop = SubSection(
        sub_section=SectionProxy('x_lobster_section_coop'))

    x_lobster_section_atom_projected_dos = SubSection(
        sub_section=SectionProxy('x_lobster_section_atom_projected_dos'),
        repeats=True)


class Method(simulation.method.Method):

    m_def = Section(validate=False, extends_base_section=True)

    x_lobster_code = Quantity(
        type=str,
        description='''
        Used PAW program
        ''')


class x_lobster_section_cohp(MSection):
    """
    This is a section containing the crystal orbital hamilton population (COHP)
    and integrated COHP (iCOHP) values.
    """
    m_def = Section(validate=False)

    x_lobster_number_of_cohp_pairs = Quantity(
        type=int,
        description='''
        Number of atom pairs for which are the COHPs and iCOHPs calculated.
        ''')

    x_lobster_cohp_atom1_labels = Quantity(
        type=str,
        shape=['x_lobster_number_of_cohp_pairs'],
        description='''
        Species and indices of the first atom for which is the specific COHP/iCOHP calculated
        ''')

    x_lobster_cohp_atom2_labels = Quantity(
        type=str,
        shape=['x_lobster_number_of_cohp_pairs'],
        description='''
        Species and indices of the second atom for which is the specific COHP/iCOHP calculated
        ''')

    x_lobster_cohp_distances = Quantity(
        type=np.dtype(np.float64),
        unit='meter',
        shape=['x_lobster_number_of_cohp_pairs'],
        description='''
        Distance between atoms of the pair for which is the specific COHP/iCOHP calculated.
        ''')

    x_lobster_cohp_translations = Quantity(
        type=np.dtype(np.int32),
        shape=['x_lobster_number_of_cohp_pairs', 3],
        description='''
        Vector connecting the unit-cell of the first atom with the one of the second atom

        This is only used with LOBSTER versions 3.0.0 and above, older versions use
        x_lobster_cohp_number_of_bonds instead.
        ''')

    x_lobster_integrated_cohp_at_fermi_level = Quantity(
        type=np.dtype(np.float32),
        unit='joule',
        shape=['number_of_spin_channels', 'x_lobster_number_of_cohp_pairs'],
        description='''
        Calculated iCOHP values ingregrated up to the Fermi level.
        ''')

    x_lobster_number_of_cohp_values = Quantity(
        type=int,
        description='''
        Number of energy values for the COHP and iCOHP.
        ''')

    x_lobster_cohp_energies = Quantity(
        type=np.dtype(np.float32),
        unit='joule',
        shape=['x_lobster_number_of_cohp_values'],
        description='''
        Array containing the set of discrete energy values for COHP and iCOHP.
        ''')

    x_lobster_cohp_values = Quantity(
        type=np.dtype(np.float32),
        shape=['x_lobster_number_of_cohp_pairs', 'number_of_spin_channels', 'x_lobster_number_of_cohp_values'],
        description='''
        Calculated COHP values.
        ''')

    x_lobster_integrated_cohp_values = Quantity(
        type=np.dtype(np.float32),
        unit='joule',
        shape=['x_lobster_number_of_cohp_pairs', 'number_of_spin_channels', 'x_lobster_number_of_cohp_values'],
        description='''
        Calculated iCOHP values.
        ''')

    x_lobster_average_cohp_values = Quantity(
        type=np.dtype(np.float32),
        shape=['number_of_spin_channels', 'x_lobster_number_of_cohp_values'],
        description='''
        Calculated COHP values averaged over all pairs.
        ''')

    x_lobster_average_integrated_cohp_values = Quantity(
        type=np.dtype(np.float32),
        unit='joule',
        shape=['number_of_spin_channels', 'x_lobster_number_of_cohp_values'],
        description='''
        Calculated iCOHP values averaged over all pairs.
        ''')

    x_lobster_cohp_number_of_bonds = Quantity(
        type=int,
        shape=['x_lobster_number_of_cohp_pairs'],
        description='''
        Number of bonds between first atom and the second atom (including
        the periodic images).

        This is only used in older LOBSTER versions, new versions print one line
        for every neighbor, so a pair which had x_lobster_icohp_number_of_bonds = 4
        in the old version would actually show as 4 lines in the ICOHPLIST or 4 columns
        in the COPHCAR in the new format.
        ''')


class x_lobster_section_coop(MSection):
    """
    This is a section containing the crystal orbital hamilton population (COOP)
    and integrated coop (iCOOP) values.
    """
    m_def = Section(validate=False)

    x_lobster_number_of_coop_pairs = Quantity(
        type=int,
        description='''
        Number of atom pairs for which are the COOPs and iCOOPs calculated.
        ''')

    x_lobster_coop_atom1_labels = Quantity(
        type=str,
        shape=['x_lobster_number_of_coop_pairs'],
        description='''
        Species and indices of the first atom for which is the specific COOP/iCOOP calculated
        ''')

    x_lobster_coop_atom2_labels = Quantity(
        type=str,
        shape=['x_lobster_number_of_coop_pairs'],
        description='''
        Species and indices of the second atom for which is the specific COOP/iCOOP calculated
        ''')

    x_lobster_coop_distances = Quantity(
        type=np.dtype(np.float64),
        unit='meter',
        shape=['x_lobster_number_of_coop_pairs'],
        description='''
        Distance between atoms of the pair for which is the specific COOP/iCOOP calculated.
        ''')

    x_lobster_coop_translations = Quantity(
        type=np.dtype(np.int32),
        shape=['x_lobster_number_of_coop_pairs', 3],
        description='''
        Vector connecting the unit-cell of the first atom with the one of the second atom

        This is only used with LOBSTER versions 3.0.0 and above, older versions use
        x_lobster_coop_number_of_bonds instead.
        ''')

    x_lobster_integrated_coop_at_fermi_level = Quantity(
        type=np.dtype(np.float32),
        unit='joule',
        shape=['number_of_spin_channels', 'x_lobster_number_of_coop_pairs'],
        description='''
        Calculated iCOOP values ingregrated up to the Fermi level.
        ''')

    x_lobster_number_of_coop_values = Quantity(
        type=int,
        description='''
        Number of energy values for the COOP and iCOOP.
        ''')

    x_lobster_coop_energies = Quantity(
        type=np.dtype(np.float32),
        unit='joule',
        shape=['x_lobster_number_of_coop_values'],
        description='''
        Array containing the set of discrete energy values for COOP and iCOOP.
        ''')

    x_lobster_coop_values = Quantity(
        type=np.dtype(np.float32),
        shape=['x_lobster_number_of_coop_pairs', 'number_of_spin_channels', 'x_lobster_number_of_coop_values'],
        description='''
        Calculated COOP values.
        ''')

    x_lobster_integrated_coop_values = Quantity(
        type=np.dtype(np.float32),
        unit='joule',
        shape=['x_lobster_number_of_coop_pairs', 'number_of_spin_channels', 'x_lobster_number_of_coop_values'],
        description='''
        Calculated iCOOP values.
        ''')

    x_lobster_average_coop_values = Quantity(
        type=np.dtype(np.float32),
        shape=['number_of_spin_channels', 'x_lobster_number_of_coop_values'],
        description='''
        Calculated COOP values averaged over all pairs.
        ''')

    x_lobster_average_integrated_coop_values = Quantity(
        type=np.dtype(np.float32),
        unit='joule',
        shape=['number_of_spin_channels', 'x_lobster_number_of_coop_values'],
        description='''
        Calculated iCOOP values averaged over all pairs.
        ''')

    x_lobster_coop_number_of_bonds = Quantity(
        type=int,
        shape=['x_lobster_number_of_coop_pairs'],
        description='''
        Number of bonds between first atom and the second atom (including
        the periodic images).

        This is only used in older LOBSTER versions, new versions print one line
        for every neighbor, so a pair which had x_lobster_icoop_number_of_bonds = 4
        in the old version would actually show as 4 lines in the ICOOPLIST or 4 columns
        in the COOPCAR in the new format.
        ''')


class x_lobster_section_atom_projected_dos(MSection):
    '''
    Section collecting the information on an atom projected density of states (DOS)
    evaluation.
    FIXME: this should ultimatelly go into some common section but that is not possible
    right now, see: https://matsci.org/t/section-atom-projected-dos/36008
    '''

    m_def = Section(validate=False)

    x_lobster_atom_projected_dos_energies = Quantity(
        type=np.dtype(np.float64),
        shape=['x_lobster_number_of_atom_projected_dos_values'],
        unit='joule',
        description='''
        Array containing the set of discrete energy values for the atom-projected density
        (electronic-energy) of states (DOS).
        ''')

    x_lobster_atom_projected_dos_lm = Quantity(
        type=np.dtype(np.int32),
        shape=['x_lobster_number_of_lm_atom_projected_dos', 2],
        description='''
        Tuples of $l$ and $m$ values for which x_lobster_atom_projected_dos_values_lm are given.
        For the quantum number $l$ the conventional meaning of azimuthal quantum number is
        always adopted. For the integer number $m$, besides the conventional use as
        magnetic quantum number ($l+1$ integer values from $-l$ to $l$), a set of
        different conventions is accepted (see the [m_kind wiki
        page](https://gitlab.rzg.mpg.de/nomad-lab/nomad-meta-info/wikis/metainfo/m-kind).
        The adopted convention is specified by atom_projected_dos_m_kind.
        ''')

    x_lobster_atom_projected_dos_m_kind = Quantity(
        type=str,
        description='''
        String describing what the integer numbers of $m$ in atom_projected_dos_lm mean.
        The allowed values are listed in the [m_kind wiki
        page](https://gitlab.rzg.mpg.de/nomad-lab/nomad-meta-info/wikis/metainfo/m-kind).
        ''')

    x_lobster_atom_projected_dos_values_lm = Quantity(
        type=np.dtype(np.float64),
        shape=['x_lobster_number_of_lm_atom_projected_dos', 'number_of_spin_channels',
               'x_lobster_number_of_atom_projected_dos_values'],
        description='''
        Values correspond to the number of states for a given energy (the set of discrete
        energy values is given in atom_projected_dos_energies) divided into contributions
        from each $l,m$ channel for the atom-projected density (electronic-energy) of
        states for atom specified in atom_projected_dos_atom_index.
        ''')

    x_lobster_atom_projected_dos_values_total = Quantity(
        type=np.dtype(np.float64),
        shape=['number_of_spin_channels', 'x_lobster_number_of_atom_projected_dos_values'],
        description='''
        Values correspond to the number of states for a given energy (the set of discrete
        energy values is given in atom_projected_dos_energies) divided into contributions
        summed up over all $l$ channels for the atom-projected density (electronic-energy)
        of states (DOS) for atom specified in atom_projected_dos_atom_index.
        ''')

    x_lobster_number_of_atom_projected_dos_values = Quantity(
        type=int,
        description='''
        Gives the number of energy values for the atom-projected density of states (DOS)
        based on x_lobster_atom_projected_dos_values_lm and
        x_lobster_atom_projected_dos_values_total.
        ''')

    x_lobster_number_of_lm_atom_projected_dos = Quantity(
        type=int,
        description='''
        Gives the number of $l$, $m$ combinations for the atom projected density of states
        (DOS) defined in x_lobster_section_atom_projected_dos.
        ''')

    x_lobster_atom_projected_dos_atom_index = Quantity(
        type=int,
        description='''
        Index of atom for which is the x_lobster_atom_projected_dos provided.
        ''')


m_package.__init_metainfo__()
