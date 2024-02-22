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

from nomad.datamodel import EntryArchive
from workflowparsers.aflow import AFLOWParser


def approx(value, abs=0, rel=1e-6):
    return pytest.approx(value, abs=abs, rel=rel)


@pytest.fixture(scope='module')
def parser():
    return AFLOWParser()


def test_aflowlib(parser):
    archive = EntryArchive()
    parser.parse('tests/data/aflow/Ag1Co1O2_ICSD_246157/aflowlib.json', archive, None)

    assert len(archive.run) == 3

    assert archive.run[0].program.version == 'aflow30847'
    assert archive.run[0].x_aflow_auid == 'aflow:fbc2cf03b9659c90'
    sec_system = archive.run[0].system[0]
    assert sec_system.atoms.lattice_vectors[1][0].magnitude == approx(-1.45323479e-10)
    assert sec_system.x_aflow_Pearson_symbol_superlattice == 'hP8'
    assert list(sec_system.x_aflow_Wyckoff_letters_orig) == [['c'], ['a'], ['f']]
    sec_method = archive.run[0].method[0]
    assert sec_method.x_aflow_delta_electronic_energy_convergence == approx(3.06569e-05)
    sec_calculation = archive.run[0].calculation[0]
    assert sec_calculation.energy.total.value.magnitude == approx(-5.58872856e-18)
    assert sec_calculation.forces.total.value[0][1].magnitude == approx(-2.14691669e-13)
    assert sec_calculation.thermodynamics[0].enthalpy.magnitude == approx(
        -5.58872856e-18
    )
    assert sec_calculation.x_aflow_pressure_residual == approx(2.95)

    run = archive.run[1]
    assert len(run.system) == len(run.calculation) == 24
    assert run.system[7].atoms.labels[6] == 'O'
    assert run.system[21].atoms.lattice_vectors[0][1].magnitude == approx(
        -2.5241880209758e-10
    )
    assert run.calculation[15].energy.total.value.magnitude == approx(-6.9854741e-19)
    assert run.calculation[9].stress.total.value[2][2].magnitude == approx(-1.594e08)
    # TODO currently workflow is not a repeating section
    # assert archive.workflow.results.n_deformations == 3
    # assert archive.workflow.results.strain_maximum == pytest.approx(0.01)
    # assert archive.workflow.results.n_strains == 8
    # assert archive.workflow.results.elastic_constants_matrix_second_order[0][1].magnitude == approx(7.45333e+10)
    # assert archive.workflow.results.bulk_modulus_voigt.magnitude == approx(1.50939e+11)
    # assert archive.workflow.results.pugh_ratio_hill == approx(0.298965)

    run = archive.run[2]
    assert len(run.system) == len(run.calculation) == 28
    assert run.system[3].atoms.positions[3][2].magnitude == approx(5.53515521e-10)
    assert run.calculation[19].thermodynamics[0].pressure.magnitude == (-1.6886e09)
    sec_thermo = archive.workflow2.results
    assert sec_thermo.temperature[12].magnitude == 120
    # assert sec_thermo.thermal_conductivity[18].magnitude == approx(4.924586)
    # assert sec_thermo.gruneisen_parameter[35] == approx(2.255801)
    assert sec_thermo.heat_capacity_c_p[87].magnitude == approx(3.73438425e-22)
    assert sec_thermo.vibrational_free_energy[93].magnitude == approx(-4.13010555e-19)


def test_aflowin(parser):
    archive = EntryArchive()
    parser.parse('tests/data/aflow/MgO/aflow.in', archive, None)

    assert len(archive.run) == 2

    assert archive.run[0].program.version == '3.2.1'
    sec_system = archive.run[0].system[0]
    assert sec_system.atoms.lattice_vectors[1][2].magnitude == approx(2.1277509e-10)
    assert sec_system.atoms.positions[1][1].magnitude == approx(2.1277509e-10)

    sec_scc = archive.run[1].calculation[0]
    assert sec_scc.dos_phonon[0].energies[80].magnitude == approx(5.71011064e-22)
    assert sec_scc.dos_phonon[0].total[0].value[1866].magnitude == approx(2.06688135e20)
    assert len(sec_scc.band_structure_phonon[0].segment) == 10
    assert sec_scc.band_structure_phonon[0].segment[3].kpoints[7][1] == approx(
        1.02984830
    )
    assert sec_scc.band_structure_phonon[0].segment[9].energies[0][10][
        3
    ].magnitude == approx(1.92480691e-21)

    assert archive.workflow2.results.qpoints[9249][0] == approx(-4.7619047619e-02)
    assert archive.workflow2.results.group_velocity[234][2][0].magnitude == approx(
        -133.348333
    )

    # TODO currently workflow is not a repeating section
    # assert archive.workflow.results.temperature[161].magnitude == approx(1610)
    # assert archive.workflow.results.internal_energy[190].magnitude == approx(1.58571787e-19)
    # assert archive.workflow.results.helmholtz_free_energy[108].magnitude == approx(-6.4052878e-20)
    # assert archive.workflow.results.entropy[10].magnitude == approx(4.96817858e-24)
    # assert archive.workflow.results.heat_capacity_c_v[35].magnitude == approx(6.72704591e-23)
