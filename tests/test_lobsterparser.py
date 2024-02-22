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

import pytest
import logging
import numpy as np

from nomad.datamodel import EntryArchive
from nomad.units import ureg as units

from workflowparsers.lobster import LobsterParser

e = (1 * units.e).to_base_units().magnitude
eV = (1 * units.e).to_base_units().magnitude


@pytest.fixture
def parser():
    return LobsterParser()


def A_to_m(value):
    return (value * units.angstrom).to_base_units().magnitude


def eV_to_J(value):
    return (value * units.eV).to_base_units().magnitude


# default pytest.approx settings are abs=1e-12, rel=1e-6 so it doesn't work for small numbers
# use the default just for comparison with zero
def approx(value):
    return pytest.approx(value, abs=0, rel=1e-6)


def test_Fe(parser):
    """
    Tests spin-polarized Fe calculation with LOBSTER 4.0.0
    """

    archive = EntryArchive()
    parser.parse('tests/data/lobster/Fe/lobsterout', archive, logging)

    run = archive.run[0]
    assert run.program.name == 'LOBSTER'
    assert run.clean_end is True
    assert run.program.version == '4.0.0'
    assert run.time_run.wall_start.magnitude == 1619687985

    assert len(run.calculation) == 1
    scc = run.calculation[0]
    assert len(scc.x_lobster_abs_total_spilling) == 2
    assert scc.x_lobster_abs_total_spilling[0] == approx(8.02)
    assert scc.x_lobster_abs_total_spilling[1] == approx(8.96)
    assert len(scc.x_lobster_abs_charge_spilling) == 2
    assert scc.x_lobster_abs_charge_spilling[0] == approx(2.97)
    assert scc.x_lobster_abs_charge_spilling[1] == approx(8.5)

    method = run.method
    assert len(method) == 1
    assert method[0].x_lobster_code == 'VASP'
    assert method[0].electrons_representation[0].basis_set[0].type == 'pbeVaspFit2015'

    # ICOHPLIST.lobster
    cohp = scc.x_lobster_section_cohp
    assert cohp.x_lobster_number_of_cohp_pairs == 20
    assert len(cohp.x_lobster_cohp_atom1_labels) == 20
    assert cohp.x_lobster_cohp_atom1_labels[19] == 'Fe2'
    assert len(cohp.x_lobster_cohp_atom2_labels) == 20
    assert cohp.x_lobster_cohp_atom1_labels[3] == 'Fe1'
    assert len(cohp.x_lobster_cohp_distances) == 20
    assert cohp.x_lobster_cohp_distances[0].magnitude == approx(A_to_m(2.831775))
    assert cohp.x_lobster_cohp_distances[13].magnitude == approx(A_to_m(2.45239))
    assert cohp.x_lobster_cohp_distances[19].magnitude == approx(A_to_m(2.831775))
    assert np.shape(cohp.x_lobster_cohp_translations) == (20, 3)
    assert (cohp.x_lobster_cohp_translations[0] == [0, 0, -1]).all()
    assert (cohp.x_lobster_cohp_translations[13] == [0, 0, 0]).all()
    assert (cohp.x_lobster_cohp_translations[19] == [0, 0, 1]).all()
    assert np.shape(cohp.x_lobster_integrated_cohp_at_fermi_level) == (2, 20)
    assert cohp.x_lobster_integrated_cohp_at_fermi_level[0, 0].magnitude == approx(
        eV_to_J(-0.08672)
    )
    assert cohp.x_lobster_integrated_cohp_at_fermi_level[0, 19].magnitude == approx(
        eV_to_J(-0.08672)
    )
    assert cohp.x_lobster_integrated_cohp_at_fermi_level[1, 19].magnitude == approx(
        eV_to_J(-0.16529)
    )
    assert cohp.x_lobster_integrated_cohp_at_fermi_level[1, 7].magnitude == approx(
        eV_to_J(-0.48790)
    )

    # COHPCAR.lobster
    assert len(cohp.x_lobster_cohp_energies) == 201
    assert cohp.x_lobster_cohp_energies[0].magnitude == approx(eV_to_J(-10.06030))
    assert cohp.x_lobster_cohp_energies[200].magnitude == approx(eV_to_J(3.00503))
    assert np.shape(cohp.x_lobster_average_cohp_values) == (2, 201)
    assert cohp.x_lobster_average_cohp_values[0][196] == approx(0.02406)
    assert cohp.x_lobster_average_cohp_values[1][200] == approx(0.01816)
    assert np.shape(cohp.x_lobster_average_integrated_cohp_values) == (2, 201)
    assert cohp.x_lobster_average_integrated_cohp_values[0][200].magnitude == approx(
        eV_to_J(-0.06616)
    )
    assert cohp.x_lobster_average_integrated_cohp_values[1][200].magnitude == approx(
        eV_to_J(-0.02265)
    )
    assert np.shape(cohp.x_lobster_cohp_values) == (20, 2, 201)
    assert cohp.x_lobster_cohp_values[10][1][200] == approx(0.02291)
    assert cohp.x_lobster_cohp_values[19][0][200] == approx(0.01439)
    assert np.shape(cohp.x_lobster_integrated_cohp_values) == (20, 2, 201)
    assert cohp.x_lobster_integrated_cohp_values[10][0][200].magnitude == approx(
        eV_to_J(-0.12881)
    )
    assert cohp.x_lobster_integrated_cohp_values[19][1][200].magnitude == approx(
        eV_to_J(-0.06876)
    )

    # ICOOPLIST.lobster
    coop = scc.x_lobster_section_coop
    assert coop.x_lobster_number_of_coop_pairs == 20
    assert len(coop.x_lobster_coop_atom1_labels) == 20
    assert coop.x_lobster_coop_atom1_labels[19] == 'Fe2'
    assert len(coop.x_lobster_coop_atom2_labels) == 20
    assert coop.x_lobster_coop_atom1_labels[3] == 'Fe1'
    assert len(coop.x_lobster_coop_distances) == 20
    assert coop.x_lobster_coop_distances[0].magnitude == approx(A_to_m(2.831775))
    assert coop.x_lobster_coop_distances[13].magnitude == approx(A_to_m(2.45239))
    assert coop.x_lobster_coop_distances[19].magnitude == approx(A_to_m(2.831775))
    assert np.shape(coop.x_lobster_coop_translations) == (20, 3)
    assert (coop.x_lobster_coop_translations[0] == [0, 0, -1]).all()
    assert (coop.x_lobster_coop_translations[13] == [0, 0, 0]).all()
    assert (coop.x_lobster_coop_translations[19] == [0, 0, 1]).all()
    assert np.shape(coop.x_lobster_integrated_coop_at_fermi_level) == (2, 20)
    assert coop.x_lobster_integrated_coop_at_fermi_level[0, 0].magnitude == approx(
        eV_to_J(-0.06882)
    )
    assert coop.x_lobster_integrated_coop_at_fermi_level[0, 19].magnitude == approx(
        eV_to_J(-0.06882)
    )
    assert coop.x_lobster_integrated_coop_at_fermi_level[1, 19].magnitude == approx(
        eV_to_J(-0.11268)
    )
    assert coop.x_lobster_integrated_coop_at_fermi_level[1, 7].magnitude == approx(
        eV_to_J(-0.05179)
    )

    # COOPCAR.lobster
    assert len(coop.x_lobster_coop_energies) == 201
    assert coop.x_lobster_coop_energies[0].magnitude == approx(eV_to_J(-10.06030))
    assert coop.x_lobster_coop_energies[200].magnitude == approx(eV_to_J(3.00503))
    assert np.shape(coop.x_lobster_average_coop_values) == (2, 201)
    assert coop.x_lobster_average_coop_values[0][196] == approx(-0.04773)
    assert coop.x_lobster_average_coop_values[1][200] == approx(-0.04542)
    assert np.shape(coop.x_lobster_average_integrated_coop_values) == (2, 201)
    assert coop.x_lobster_average_integrated_coop_values[0][200].magnitude == approx(
        eV_to_J(-0.12265)
    )
    assert coop.x_lobster_average_integrated_coop_values[1][200].magnitude == approx(
        eV_to_J(-0.14690)
    )
    assert np.shape(coop.x_lobster_coop_values) == (20, 2, 201)
    assert coop.x_lobster_coop_values[3][1][200] == approx(-0.01346)
    assert coop.x_lobster_coop_values[0][0][200] == approx(-0.04542)
    assert np.shape(coop.x_lobster_integrated_coop_values) == (20, 2, 201)
    assert coop.x_lobster_integrated_coop_values[10][0][199].magnitude == approx(
        eV_to_J(-0.07360)
    )
    assert coop.x_lobster_integrated_coop_values[19][1][200].magnitude == approx(
        eV_to_J(-0.13041)
    )

    # CHARGE.lobster
    charges = scc.charges
    assert len(charges) == 2
    mulliken = charges[0]
    assert mulliken.analysis_method == 'mulliken'
    assert np.shape(mulliken.value) == (2,)
    assert mulliken.value[0] == pytest.approx(0.0 * e, abs=1e-6)
    assert mulliken.value[1] == pytest.approx(0.0 * e, abs=1e-6)

    loewdin = charges[1]
    assert loewdin.analysis_method == 'loewdin'
    assert np.shape(loewdin.value) == (2,)
    assert loewdin.value[0] == pytest.approx(0.0 * e, abs=1e-6)
    assert loewdin.value[1] == pytest.approx(0.0 * e, abs=1e-6)

    # DOSCAR.lobster total and integrated DOS
    assert len(scc.dos_electronic) == 2
    dos_up = scc.dos_electronic[0]
    dos_down = scc.dos_electronic[1]
    assert dos_up.n_energies == 201
    assert len(dos_up.energies) == 201
    assert dos_up.energies[0].magnitude == approx(eV_to_J(-10.06030))
    assert dos_up.energies[16].magnitude == approx(eV_to_J(-9.01508))
    assert dos_up.energies[200].magnitude == approx(eV_to_J(3.00503))
    assert len(dos_up.total) == 1 and len(dos_down.total) == 1
    assert np.shape(dos_down.total[0].value) == (201,)
    assert dos_up.total[0].value[6] == pytest.approx(0.0, abs=1e-30)
    assert dos_up.total[0].value[200].magnitude == approx(0.26779 / eV)
    assert dos_down.total[0].value[195].magnitude == approx(0.37457 / eV)
    assert np.shape(dos_up.total[0].value_integrated) == (201,)
    assert dos_up.total[0].value_integrated[10] == approx(0.0 + 18)
    assert dos_up.total[0].value_integrated[188] == approx(11.07792 + 18)
    assert dos_down.total[0].value_integrated[200] == approx(10.75031 + 18)

    # DOSCAR.lobster atom and lm-projected dos
    assert len(dos_up.atom_projected) == 12 and len(dos_down.atom_projected) == 12
    assert (
        dos_up.atom_projected[0].atom_index == 0
        and dos_up.atom_projected[6].atom_index == 1
    )
    assert dos_up.atom_projected[0].m_kind == 'real_orbital'
    assert (dos_up.atom_projected[4].lm == [2, 1]).all()
    assert np.shape(dos_up.atom_projected[11].value) == (201,)
    assert dos_up.atom_projected[5].value[190].to('1/eV').magnitude == approx(0.00909)
    assert dos_down.atom_projected[5].value[190].to('1/eV').magnitude == approx(0.29205)


def test_NaCl(parser):
    """
    Test non-spin-polarized NaCl calculation with LOBSTER 3.2.0
    """

    archive = EntryArchive()
    parser.parse('tests/data/lobster/NaCl/lobsterout', archive, logging)

    run = archive.run[0]
    assert run.program.name == 'LOBSTER'
    assert run.clean_end is True
    assert run.program.version == '3.2.0'
    assert run.time_run.wall_start.magnitude == 1619713048

    assert len(run.calculation) == 1
    scc = run.calculation[0]
    assert len(scc.x_lobster_abs_total_spilling) == 1
    assert scc.x_lobster_abs_total_spilling[0] == approx(9.29)
    assert len(scc.x_lobster_abs_charge_spilling) == 1
    assert scc.x_lobster_abs_charge_spilling[0] == approx(0.58)

    method = run.method
    assert len(method) == 1
    assert method[0].x_lobster_code == 'VASP'
    assert method[0].electrons_representation[0].basis_set[0].type == 'pbeVaspFit2015'

    # ICOHPLIST.lobster
    cohp = scc.x_lobster_section_cohp
    assert cohp.x_lobster_number_of_cohp_pairs == 72
    assert len(cohp.x_lobster_cohp_atom1_labels) == 72
    assert cohp.x_lobster_cohp_atom1_labels[71] == 'Cl7'
    assert len(cohp.x_lobster_cohp_atom2_labels) == 72
    assert cohp.x_lobster_cohp_atom2_labels[43] == 'Cl6'
    assert len(cohp.x_lobster_cohp_distances) == 72
    assert cohp.x_lobster_cohp_distances[0].magnitude == approx(A_to_m(3.99586))
    assert cohp.x_lobster_cohp_distances[47].magnitude == approx(A_to_m(2.82550))
    assert cohp.x_lobster_cohp_distances[71].magnitude == approx(A_to_m(3.99586))
    assert np.shape(cohp.x_lobster_cohp_translations) == (72, 3)
    assert (cohp.x_lobster_cohp_translations[0] == [-1, 0, 0]).all()
    assert (cohp.x_lobster_cohp_translations[54] == [0, -1, 0]).all()
    assert (cohp.x_lobster_cohp_translations[71] == [0, 1, 0]).all()
    assert np.shape(cohp.x_lobster_integrated_cohp_at_fermi_level) == (1, 72)
    assert cohp.x_lobster_integrated_cohp_at_fermi_level[0, 0].magnitude == approx(
        eV_to_J(-0.02652)
    )
    assert cohp.x_lobster_integrated_cohp_at_fermi_level[0, 71].magnitude == approx(
        eV_to_J(-0.02925)
    )

    # COHPCAR.lobster
    assert len(cohp.x_lobster_cohp_energies) == 201
    assert cohp.x_lobster_cohp_energies[0].magnitude == approx(eV_to_J(-12.02261))
    assert cohp.x_lobster_cohp_energies[200].magnitude == approx(eV_to_J(2.55025))
    assert np.shape(cohp.x_lobster_average_cohp_values) == (1, 201)
    assert cohp.x_lobster_average_cohp_values[0][0] == pytest.approx(0.0)
    assert cohp.x_lobster_average_cohp_values[0][151] == approx(-0.03162)
    assert np.shape(cohp.x_lobster_average_integrated_cohp_values) == (1, 201)
    assert cohp.x_lobster_average_integrated_cohp_values[0][0].magnitude == approx(
        eV_to_J(-0.15834)
    )
    assert cohp.x_lobster_average_integrated_cohp_values[0][200].magnitude == approx(
        eV_to_J(-0.24310)
    )
    assert np.shape(cohp.x_lobster_cohp_values) == (72, 1, 201)
    assert cohp.x_lobster_cohp_values[1][0][200] == pytest.approx(0.0)
    assert cohp.x_lobster_cohp_values[71][0][140] == approx(-0.00403)
    assert np.shape(cohp.x_lobster_integrated_cohp_values) == (72, 1, 201)
    assert cohp.x_lobster_integrated_cohp_values[2][0][200].magnitude == approx(
        eV_to_J(-0.02652)
    )
    assert cohp.x_lobster_integrated_cohp_values[67][0][199].magnitude == approx(
        eV_to_J(-0.04137)
    )

    # ICOOPLIST.lobster
    coop = scc.x_lobster_section_coop
    assert coop.x_lobster_number_of_coop_pairs == 72
    assert len(coop.x_lobster_coop_atom1_labels) == 72
    assert coop.x_lobster_coop_atom1_labels[71] == 'Cl7'
    assert len(coop.x_lobster_coop_atom2_labels) == 72
    assert coop.x_lobster_coop_atom2_labels[0] == 'Na2'
    assert len(coop.x_lobster_coop_distances) == 72
    assert coop.x_lobster_coop_distances[0].magnitude == approx(A_to_m(3.99586))
    assert coop.x_lobster_coop_distances[12].magnitude == approx(A_to_m(2.82550))
    assert coop.x_lobster_coop_distances[71].magnitude == approx(A_to_m(3.99586))
    assert np.shape(coop.x_lobster_coop_translations) == (72, 3)
    assert (coop.x_lobster_coop_translations[0] == [-1, 0, 0]).all()
    assert (coop.x_lobster_coop_translations[13] == [0, 1, 0]).all()
    assert (coop.x_lobster_coop_translations[71] == [0, 1, 0]).all()
    assert np.shape(coop.x_lobster_integrated_coop_at_fermi_level) == (1, 72)
    assert coop.x_lobster_integrated_coop_at_fermi_level[0, 0].magnitude == approx(
        eV_to_J(-0.00519)
    )
    assert coop.x_lobster_integrated_coop_at_fermi_level[0, 71].magnitude == approx(
        eV_to_J(-0.00580)
    )

    # COOPCAR.lobster
    assert len(coop.x_lobster_coop_energies) == 201
    assert coop.x_lobster_coop_energies[0].magnitude == approx(eV_to_J(-12.02261))
    assert coop.x_lobster_coop_energies[200].magnitude == approx(eV_to_J(2.55025))
    assert np.shape(coop.x_lobster_average_coop_values) == (1, 201)
    assert coop.x_lobster_average_coop_values[0][0] == pytest.approx(0.0)
    assert coop.x_lobster_average_coop_values[0][145] == approx(0.03178)
    assert np.shape(coop.x_lobster_average_integrated_coop_values) == (1, 201)
    assert coop.x_lobster_average_integrated_coop_values[0][0].magnitude == approx(
        eV_to_J(0.00368)
    )
    assert coop.x_lobster_average_integrated_coop_values[0][200].magnitude == approx(
        eV_to_J(0.00682)
    )
    assert np.shape(coop.x_lobster_coop_values) == (72, 1, 201)
    assert coop.x_lobster_coop_values[1][0][200] == pytest.approx(0.0)
    assert coop.x_lobster_coop_values[71][0][143] == approx(0.01862)
    assert np.shape(coop.x_lobster_integrated_coop_values) == (72, 1, 201)
    assert coop.x_lobster_integrated_coop_values[2][0][200].magnitude == approx(
        eV_to_J(-0.00519)
    )
    assert coop.x_lobster_integrated_coop_values[71][0][199].magnitude == approx(
        eV_to_J(-0.00580)
    )

    # CHARGE.lobster
    charges = scc.charges
    assert len(charges) == 2
    mulliken = charges[0]
    assert mulliken.analysis_method == 'mulliken'
    # here the approx is not really working (changing the 0.78 to for example
    # 10 makes the test still pass)
    assert mulliken.value[0].magnitude == approx(0.78 * e)
    assert mulliken.value[7].magnitude == approx(-0.78 * e)

    loewdin = charges[1]
    assert loewdin.analysis_method == 'loewdin'
    assert loewdin.value[0].magnitude == approx(0.67 * e)
    assert loewdin.value[7].magnitude == approx(-0.67 * e)

    # DOSCAR.lobster total and integrated DOS
    assert len(scc.dos_electronic) == 1
    dos = scc.dos_electronic[0]
    assert dos.n_energies == 201
    assert len(dos.energies) == 201
    assert dos.energies[0].magnitude == approx(eV_to_J(-12.02261))
    assert dos.energies[25].magnitude == approx(eV_to_J(-10.20101))
    assert dos.energies[200].magnitude == approx(eV_to_J(2.55025))
    assert np.shape(dos.total[0].value) == (201,)
    assert dos.total[0].value[6].magnitude == pytest.approx(0.0, abs=1e-30)
    assert dos.total[0].value[162].magnitude == approx(20.24722 / eV)
    assert dos.total[0].value[200].magnitude == pytest.approx(0.0, abs=1e-30)
    assert np.shape(dos.total[0].value_integrated) == (201,)
    assert dos.total[0].value_integrated[10] == approx(7.99998 + 80)
    assert dos.total[0].value_integrated[160] == approx(27.09225 + 80)
    assert dos.total[0].value_integrated[200] == approx(31.99992 + 80)

    # DOSCAR.lobster atom and lm-projected dos
    assert len(dos.atom_projected) == 20
    dos.atom_projected[0].atom_index == 0
    dos.atom_projected[19].atom_index == 7
    assert dos.atom_projected[5].m_kind == 'real_orbital'
    assert (dos.atom_projected[17].lm == [1, 2]).all()
    assert np.shape(dos.atom_projected[13].value) == (201,)
    assert np.shape(dos.atom_projected[8].value) == (201,)
    assert dos.atom_projected[0].value[190].magnitude == pytest.approx(0.0, abs=1e-30)
    assert dos.atom_projected[19].value[141].magnitude == approx(0.32251 / eV)
    assert dos.atom_projected[16].value[152].magnitude == approx(0.00337 / eV)


def test_HfV(parser):
    """
    Test non-spin-polarized HfV2 calculation with LOBSTER 2.0.0,
    it has different ICOHPLIST.lobster and ICOOPLIST.lobster scheme.
    Also test backup structure parsing when no CONTCAR is present.
    """

    archive = EntryArchive()
    parser.parse('tests/data/lobster/HfV2/lobsterout', archive, logging)

    run = archive.run[0]
    assert run.program.name == 'LOBSTER'
    assert run.clean_end is True
    assert run.program.version == '2.0.0'

    assert len(run.calculation) == 1
    scc = run.calculation[0]
    assert len(scc.x_lobster_abs_total_spilling) == 1
    assert scc.x_lobster_abs_total_spilling[0] == approx(4.39)
    assert len(scc.x_lobster_abs_charge_spilling) == 1
    assert scc.x_lobster_abs_charge_spilling[0] == approx(2.21)

    # backup partial system parsing
    system = run.system
    assert len(system) == 1
    assert len(system[0].atoms.species) == 12
    assert (
        system[0].atoms.species == [72, 72, 72, 72, 23, 23, 23, 23, 23, 23, 23, 23]
    ).all()
    assert system[0].atoms.periodic == [True, True, True]

    # method
    method = run.method
    assert method[0].electrons_representation[0].basis_set[0].type == 'Koga'

    # ICOHPLIST.lobster
    cohp = scc.x_lobster_section_cohp
    assert cohp.x_lobster_number_of_cohp_pairs == 56
    assert len(cohp.x_lobster_cohp_atom1_labels) == 56
    assert cohp.x_lobster_cohp_atom1_labels[41] == 'V6'
    assert len(cohp.x_lobster_cohp_atom2_labels) == 56
    assert cohp.x_lobster_cohp_atom2_labels[16] == 'V9'
    assert len(cohp.x_lobster_cohp_distances) == 56
    assert cohp.x_lobster_cohp_distances[0].magnitude == approx(A_to_m(3.17294))
    assert cohp.x_lobster_cohp_distances[47].magnitude == approx(A_to_m(2.60684))
    assert cohp.x_lobster_cohp_distances[55].magnitude == approx(A_to_m(2.55809))
    assert cohp.x_lobster_cohp_translations is None
    assert len(cohp.x_lobster_cohp_number_of_bonds) == 56
    assert cohp.x_lobster_cohp_number_of_bonds[0] == 2
    assert cohp.x_lobster_cohp_number_of_bonds[53] == 1
    assert np.shape(cohp.x_lobster_integrated_cohp_at_fermi_level) == (1, 56)
    assert cohp.x_lobster_integrated_cohp_at_fermi_level[0, 0].magnitude == approx(
        eV_to_J(-1.72125)
    )
    assert cohp.x_lobster_integrated_cohp_at_fermi_level[0, 55].magnitude == approx(
        eV_to_J(-1.62412)
    )

    # ICOOPLIST.lobster
    coop = scc.x_lobster_section_coop
    assert coop.x_lobster_number_of_coop_pairs == 56
    assert len(coop.x_lobster_coop_atom1_labels) == 56
    assert coop.x_lobster_coop_atom1_labels[41] == 'V6'
    assert len(coop.x_lobster_coop_atom2_labels) == 56
    assert coop.x_lobster_coop_atom2_labels[11] == 'Hf4'
    assert len(coop.x_lobster_coop_distances) == 56
    assert coop.x_lobster_coop_distances[0].magnitude == approx(A_to_m(3.17294))
    assert coop.x_lobster_coop_distances[47].magnitude == approx(A_to_m(2.60684))
    assert coop.x_lobster_coop_distances[55].magnitude == approx(A_to_m(2.55809))
    assert coop.x_lobster_coop_translations is None
    assert len(coop.x_lobster_coop_number_of_bonds) == 56
    assert coop.x_lobster_coop_number_of_bonds[0] == 2
    assert coop.x_lobster_coop_number_of_bonds[53] == 1
    assert np.shape(coop.x_lobster_integrated_coop_at_fermi_level) == (1, 56)
    assert coop.x_lobster_integrated_coop_at_fermi_level[0, 0].magnitude == approx(
        eV_to_J(-0.46493)
    )
    assert coop.x_lobster_integrated_coop_at_fermi_level[0, 55].magnitude == approx(
        eV_to_J(-0.50035)
    )


def test_QE_Ni(parser):
    """
    Check that basic info is parsed properly when LOBSTER is run on top
    of Quantum Espresso calculations.
    """

    archive = EntryArchive()
    parser.parse('tests/data/lobster/Ni/lobsterout', archive, logging)

    run = archive.run[0]

    # QE system parsing
    system = run.system
    assert len(system) == 1
    assert system[0].atoms.labels == ['Ni']
    assert system[0].atoms.periodic == [True, True, True]
    assert len(system[0].atoms.positions) == 1
    assert (system[0].atoms.positions[0].magnitude == [0, 0, 0]).all()

    method = run.method
    assert len(method) == 1
    assert method[0].x_lobster_code == 'Quantum Espresso'
    assert method[0].electrons_representation[0].basis_set[0].type == 'Bunge'

    assert len(run.calculation) == 1
    scc = run.calculation[0]
    assert len(scc.x_lobster_abs_total_spilling) == 2
    assert scc.x_lobster_abs_total_spilling[0] == approx(36.14)
    assert scc.x_lobster_abs_total_spilling[1] == approx(36.11)
    assert len(scc.x_lobster_abs_charge_spilling) == 2
    assert scc.x_lobster_abs_charge_spilling[0] == approx(4.02)
    assert scc.x_lobster_abs_charge_spilling[1] == approx(3.37)

    assert run.clean_end is True


def test_failed_case(parser):
    """
    Check that we also handle gracefully a case where the lobster ends very early.
    Here it is because of a wrong CONTCAR.
    """

    archive = EntryArchive()
    parser.parse('tests/data/lobster/failed_case/lobsterout', archive, logging)

    run = archive.run[0]
    assert run.clean_end is False
