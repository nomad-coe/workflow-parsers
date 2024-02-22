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
from workflowparsers.quantum_espresso_xspectra import QuantumEspressoXSpectraParser


def approx(value, abs=0, rel=1e-6):
    return pytest.approx(value, abs=abs, rel=rel)


@pytest.fixture(scope='module')
def parser():
    return QuantumEspressoXSpectraParser()


def test_1(parser):
    archive = EntryArchive()
    parser.parse(
        'tests/data/quantum_espresso_xspectra/ms-10734/Spectra-1-1-1/0/dipole1/xanes.out',
        archive,
        None,
    )

    sec_run = archive.run[0]
    assert sec_run.m_xpath('x_qe_xspectra_input')

    # Program
    assert sec_run.program.name == 'Quantum ESPRESSO XSPECTRA'
    assert sec_run.program.version == '6.7MaX'

    # System
    sec_system = sec_run.system[0]
    assert sec_system.x_qe_xspectra_bravais_lattice_index == 0
    assert sec_system.x_qe_xspectra_unit_cell_volume.magnitude == approx(
        7.381107151717e-28
    )
    assert sec_system.x_qe_xspectra_n_atoms_cell == 72
    assert sec_system.atoms.labels[0] == 'Ti'
    assert sec_system.atoms.labels[4] == 'O'
    assert sec_system.atoms.positions[0][0].magnitude == approx(1.8823018373897882e-10)

    # Method
    sec_method = sec_run.method
    assert len(sec_method) == 2
    assert sec_method[0].m_xpath('electronic') and sec_method[0].m_xpath(
        'atom_parameters'
    )
    assert sec_method[0].m_xpath('photon')
    assert sec_method[0].photon[0].multipole_type == 'dipole'
    sec_core_hole = sec_method[1].core_hole
    assert sec_core_hole.solver == 'Lanczos'
    assert sec_core_hole.mode == 'absorption'
    assert sec_core_hole.broadening.magnitude == approx(0.89)

    # Calculation
    sec_scc = sec_run.calculation
    assert len(sec_scc) == 1
    assert sec_scc[0].system_ref == sec_system
    assert sec_scc[0].method_ref == sec_method[1]
    assert sec_scc[0].m_xpath('spectra')
    sec_spectra = sec_scc[0].spectra[0]
    assert sec_spectra.type == 'XANES'
    assert sec_spectra.n_energies == 400
    assert sec_spectra.excitation_energies[22].magnitude == approx(
        -1.6523701378886513e-18
    )
    assert sec_spectra.intensities[22] == approx(2.905e-5)
