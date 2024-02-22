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
from workflowparsers.fhivibes import FHIVibesParser


def approx(value, abs=0, rel=1e-6):
    return pytest.approx(value, abs=abs, rel=rel)


@pytest.fixture(scope='module')
def parser():
    return FHIVibesParser()


# TODO find out why tests fail on github
def _test_singlepoint(parser):
    archive = EntryArchive()
    parser.parse('tests/data/fhivibes/singlepoint.nc', archive, None)

    assert archive.workflow2.m_def.name == 'SinglePoint'

    sec_run = archive.run
    assert len(sec_run) == 10
    assert len(sec_run[2].calculation) == 1

    assert sec_run[8].system[0].atoms.positions[3][2].magnitude == approx(
        5.30098546e-10
    )
    assert sec_run[5].system[0].atoms.velocities[1][0].magnitude == approx(
        -2.18864066e03
    )

    sec_scc = sec_run[9].calculation[0]
    assert len(sec_scc.energy.contributions) == 2
    assert sec_scc.energy.contributions[1].value.magnitude == approx(-1.00925367e-14)

    sec_scc = sec_run[3].calculation[0]
    assert sec_scc.stress.contributions[0].value[1][2].magnitude == approx(
        -1.42111377e07
    )

    sec_scc = sec_run[1].calculation[0]
    assert sec_scc.stress.total.value[1][1].magnitude == approx(1.49076266e08)

    sec_scc = sec_run[6].calculation[0]
    assert sec_scc.forces.total.value[5][2].magnitude == approx(-3.47924808e-10)

    sec_scc = sec_run[5].calculation[0]
    assert sec_scc.thermodynamics[0].pressure.magnitude == approx(2.52108927e07)

    sec_scc = sec_run[2].calculation[0]
    assert sec_scc.x_fhi_vibes_pressure_kinetic.magnitude == approx(2.08283962e08)

    sec_scc = sec_run[8].calculation[0]
    assert sec_scc.x_fhi_vibes_energy_potential_harmonic.magnitude == approx(
        4.08242214e-20
    )


def _test_relaxation(parser):
    archive = EntryArchive()
    parser.parse('tests/data/fhivibes/relaxation.nc', archive, None)

    assert archive.workflow2.m_def.name == 'GeometryOptimization'

    assert len(archive.run) == 1

    sec_attrs = archive.run[0].method[0].x_fhi_vibes_section_attributes[0]
    assert sec_attrs.x_fhi_vibes_attributes_timestep.magnitude == approx(1e-15)
    assert len(sec_attrs.x_fhi_vibes_section_attributes_atoms) == 1
    sec_atoms = sec_attrs.x_fhi_vibes_section_attributes_atoms[0]
    assert len(sec_atoms.x_fhi_vibes_atoms_symbols) == 2
    assert sec_atoms.x_fhi_vibes_atoms_masses.magnitude == approx(4.66362397e-26)

    sec_metadata = sec_attrs.x_fhi_vibes_section_attributes_metadata[0]
    sec_relaxation = sec_metadata.x_fhi_vibes_section_metadata_relaxation[0]
    assert sec_relaxation.x_fhi_vibes_relaxation_maxstep == 0.2
    assert not sec_relaxation.x_fhi_vibes_relaxation_hydrostatic_strain
    assert sec_relaxation.x_fhi_vibes_relaxation_type == 'optimization'

    sec_sccs = archive.run[0].calculation
    assert len(sec_sccs) == 3
    assert sec_sccs[2].thermodynamics[0].volume.magnitude == approx(3.97721030e-29)
    assert sec_sccs[0].energy.contributions[1].value.magnitude == approx(
        -2.52313962e-15
    )


def _test_molecular_dynamics(parser):
    archive = EntryArchive()
    parser.parse('tests/data/fhivibes/molecular_dynamics.nc', archive, None)

    assert archive.workflow2.m_def.name == 'MolecularDynamics'

    sec_attrs = archive.run[0].method[0].x_fhi_vibes_section_attributes[0]
    sec_md = sec_attrs.x_fhi_vibes_section_attributes_metadata[
        0
    ].x_fhi_vibes_section_metadata_MD[0]
    assert sec_md.x_fhi_vibes_MD_md_type == 'Langevin'
    assert sec_md.x_fhi_vibes_MD_friction == 0.02

    sec_systems = archive.run[0].system
    assert len(sec_systems) == 11
    assert sec_systems[3].atoms.positions[6][1].magnitude == approx(1.39537854e-10)
    assert sec_systems[7].atoms.velocities[1][0].magnitude == approx(-249.97586102)
    assert sec_systems[2].atoms.lattice_vectors[0][2].magnitude == approx(
        2.20004000e-21
    )

    sec_sccs = archive.run[0].calculation
    assert sec_sccs[4].x_fhi_vibes_heat_flux_0_harmonic[1].magnitude == approx(
        1.40863863e13
    )
    assert sec_sccs[5].x_fhi_vibes_atom_forces_harmonic[3][0].magnitude == approx(
        8.40976902e-10
    )
    assert sec_sccs[6].x_fhi_vibes_momenta[7][2].magnitude == approx(-1.18929315e-24)


def _test_phonon(parser):
    archive = EntryArchive()
    parser.parse('tests/data/fhivibes/phonopy.nc', archive, None)

    assert archive.workflow2.m_def.name == 'Phonon'

    sec_attrs = archive.run[0].method[0].x_fhi_vibes_section_attributes[0]
    sec_phonon = sec_attrs.x_fhi_vibes_section_attributes_metadata[
        0
    ].x_fhi_vibes_section_metadata_phonopy[0]
    assert sec_phonon.x_fhi_vibes_phonopy_version == '2.6.1'
    sec_atoms = sec_phonon.x_fhi_vibes_section_phonopy_primitive[0]
    assert np.shape(sec_atoms.x_fhi_vibes_atoms_positions) == (2, 3)
    assert sec_atoms.x_fhi_vibes_atoms_cell[0][2].magnitude == approx(2.70925272e-10)

    sec_sccs = archive.run[0].calculation
    assert len(sec_sccs) == 1
    assert sec_sccs[0].forces.total.value[6][1].magnitude == approx(-3.96793297e-11)
    assert sec_sccs[0].x_fhi_vibes_displacements[2][1].magnitude == approx(0.0)

    sec_system = archive.run[0].system[0]
    assert sec_system.atoms.positions[3][2].magnitude == approx(5.41850544e-10)
    assert sec_system.atoms.lattice_vectors[1][1].magnitude == approx(5.41850544e-10)
