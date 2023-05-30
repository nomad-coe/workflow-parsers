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

import pytest

from nomad.datamodel import EntryArchive
from workflowparsers.atomate import AtomateParser


def approx(value, abs=0, rel=1e-6):
    return pytest.approx(value, abs=abs, rel=rel)


@pytest.fixture(scope='module')
def parser():
    return AtomateParser()


def test_all(parser):
    archive = EntryArchive()
    parser.parse('tests/data/atomate/mp-149/mp-149_materials.json', archive, None)

    run = archive.run[0]
    assert run.program.name == 'MaterialsProject'

    sec_system = run.system[0]
    assert sec_system.atoms.labels == ['Si', 'Si']
    assert sec_system.atoms.lattice_vectors[1][2].magnitude == approx(2.734364e-10)
    assert sec_system.atoms.positions[0][0].magnitude == approx(1.367182e-10)
    assert sec_system.x_mp_composition_reduced[0].x_mp_value == approx(1.0)
    assert sec_system.x_mp_symmetry[0].x_mp_symprec == approx(0.1)
    assert sec_system.x_mp_elements[0] == 'Si'
    assert sec_system.x_mp_volume == approx(40.88829284866483)
    assert sec_system.x_mp_formula_anonymous == 'A'

    # TODO currently, workflow2 is not repeating
    # TODO error loading metainfo in github action
    # assert archive.workflow2.method.energy_stress_calculator == 'VASP'
    # assert archive.workflow2.results.elastic_constants_matrix_second_order[2][1].magnitude == approx(5.3e+10)
    # assert archive.workflow2.results.compliance_matrix_second_order[1][0].magnitude == approx(-2.3e-09)
    # assert archive.workflow2.results.poisson_ratio_hill == approx(0.20424545172250694)
    # assert archive.workflow2.results.bulk_modulus_voigt.magnitude == approx(8.30112837e+10)

    # assert archive.workflow2.results.energies[5].magnitude == approx(-8.33261753e-19)
    # assert archive.workflow2.results.volumes[-4].magnitude == approx(2.43493103e-29)
    # assert len(archive.workflow2.results.eos_fit) == 8

    # for fit in archive.workflow2.results.eos_fit:
    #     if fit.function_name == 'mie_gruneisen':
    #         assert fit.fitted_energies[10].magnitude == approx(-8.62595192e-19)
    #     elif fit.function_name == 'vinet':
    #         assert fit.bulk_modulus_derivative == approx(4.986513157963165)
    #     elif fit.function_name == 'birch_euler':
    #         assert fit.equilibrium_energy.magnitude == approx(-8.69065241e-19)
    #     elif fit.function_name == 'murnaghan':
    #         assert fit.equilibrium_volume.magnitude == approx(2.04781109e-29)
    #     elif fit.function_name == 'pack_evans_james':
    #         assert fit.bulk_modulus.magnitude == approx(8.67365485e+10)
    # segment = run.calculation[-1].band_structure_phonon[0].segment
    # assert len(segment) == 10
    # assert segment[2].energies[0][7][3].magnitude == approx(7.33184304e-21)
    # assert segment[5].kpoints[9][1] == approx(0.32692307692)
    # assert segment[9].endpoints_labels == ['U', 'X']
    # dos = run.calculation[-1].dos_phonon[0]
    # assert dos.energies[20].magnitude == approx(3.49331979e-22)
    # assert dos.total[0].value[35].magnitude == approx(1.27718386e+19)
    phonon = archive.workflow2
    assert phonon.method.with_non_analytic_correction
    assert phonon.results.stability.formation_energy.magnitude == approx(0)
    assert phonon.results.stability.is_stable

    # assert archive.workflow2.method.energy_stress_calculator == 'VASP'
    # assert archive.workflow2.results.elastic_constants_matrix_second_order[2][1].magnitude == approx(5.3e+10)
    # assert archive.workflow2.results.compliance_matrix_second_order[1][0].magnitude == approx(-2.3e-09)
    # assert archive.workflow2.results.poisson_ratio_hill == approx(0.20424545172250694)
    # assert archive.workflow2.results.bulk_modulus_voigt.magnitude == approx(8.30112837e+10)
