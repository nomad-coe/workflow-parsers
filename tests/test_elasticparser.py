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
import numpy as np

from nomad.datamodel import EntryArchive
from workflowparsers.elastic import ElasticParser


def approx(value, abs=0, rel=1e-6):
    return pytest.approx(value, abs=abs, rel=rel)


@pytest.fixture(scope='module')
def parser():
    return ElasticParser()


def test_2nd(parser):
    archive = EntryArchive()
    parser.parse('tests/data/elastic/2nd/INFO_ElaStic', archive, None)

    sec_system = archive.run[0].system[0]
    assert np.shape(sec_system.atoms.positions) == (8, 3)
    assert sec_system.atoms.positions[3][1].magnitude == approx(2.5186875e-10)
    assert sec_system.atoms.lattice_vectors[2][2].magnitude == approx(6.7165e-10)
    assert sec_system.x_elastic_space_group_number == 227

    sec_method = archive.run[0].method[0]
    sec_fit_parameters = sec_method.x_elastic_section_fitting_parameters[0]
    assert sec_fit_parameters.x_elastic_fitting_parameters_eta[0] == 0.05

    assert archive.workflow2.method.energy_stress_calculator == 'exciting'
    results = archive.workflow2.results
    assert results.deformation_types[2][5] == '2eta'
    sec_strain = results.strain_diagrams
    assert len(sec_strain) == 7
    assert sec_strain[0].eta[1][3] == -0.02
    assert sec_strain[0].value[2][5] == approx(-3.30877062e-16)
    assert sec_strain[3].type == 'cross-validation'
    assert sec_strain[2].eta[1][2] == 0.03
    assert sec_strain[6].value[2][4] == approx(6.8708895e12)
    assert sec_strain[4].polynomial_fit_order == 6

    assert results.elastic_constants_notation_matrix_second_order[1][2] == 'C12'
    assert results.elastic_constants_matrix_second_order[0][2].magnitude == approx(
        1.008e11
    )
    assert results.compliance_matrix_second_order[3][3].magnitude == approx(1.75e-12)
    assert results.bulk_modulus_voigt.magnitude == approx(4.4937e11)
    assert results.shear_modulus_voigt.magnitude == approx(5.3074e11)
    assert results.bulk_modulus_reuss.magnitude == approx(4.4937e11)
    assert results.shear_modulus_reuss.magnitude == approx(5.2574e11)
    assert results.bulk_modulus_hill.magnitude == approx(4.4937e11)
    assert results.shear_modulus_hill.magnitude == approx(5.2824e11)
    assert results.young_modulus_voigt.magnitude == approx(1.14245e12)
    assert results.poisson_ratio_voigt == 0.08
    assert results.young_modulus_reuss.magnitude == approx(1.1347e12)
    assert results.poisson_ratio_reuss == 0.08
    assert results.young_modulus_hill.magnitude == approx(1.13858e12)
    assert results.poisson_ratio_hill == 0.08
    assert results.eigenvalues_elastic[1].magnitude == approx(1.3481e12)

    sec_scc = archive.run[0].calculation[0]
    assert len(sec_scc.calculations_path) == 33
    assert sec_scc.method_ref is not None
    assert sec_scc.system_ref is not None


def test_3rd(parser):
    archive = EntryArchive()
    parser.parse('tests/data/elastic/3rd/INFO_ElaStic', archive, None)

    # The strain diagram data cannot be parsed because of the inhomogeneous shape probably
    # due to error in output.
    # sec_strain = sec_elastic.strain_diagrams
    # assert len(sec_strain) == 7
    # assert len(sec_strain[1].eta) == 10
    # assert sec_strain[2].eta[1][3] == 0.07
    # assert sec_strain[3].value[8][7] == approx(2.06899957e-23)

    results = archive.workflow2.results
    assert results.elastic_constants_matrix_third_order[3][1][3].magnitude == approx(
        1.274e10
    )
    assert results.elastic_constants_matrix_third_order[5][2][5].magnitude == approx(
        1.2825e10
    )
    assert results.elastic_constants_matrix_third_order[0][0][1].magnitude == approx(
        -1.18334e12
    )


def test_stress(parser):
    # TODO no example found
    pass
