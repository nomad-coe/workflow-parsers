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
import os
import json
import numpy as np
from typing import List
import datetime

try:
    import asr
    from asr.core.cache import get_cache  # pylint: disable=E0611,E0401
    from asr.core.record import Record  # pylint: disable=E0611,E0401
except Exception:
    pass

from nomad.units import ureg
from nomad.datamodel import EntryArchive
from nomad.datamodel.metainfo.simulation.run import Run, Program, TimeRun
from nomad.datamodel.metainfo.simulation.system import System, Atoms
from nomad.datamodel.metainfo.simulation.calculation import (
    Calculation, BandStructure, BandEnergies, Energy, EnergyEntry, Forces, ForcesEntry,
    Stress, StressEntry)
from nomad.datamodel.metainfo.workflow import Workflow
from nomad.datamodel.metainfo.simulation import workflow as workflow2
from .metainfo.asr import (
    x_asr_resources, x_asr_metadata,
    x_asr_run_specification, x_asr_parameters, x_asr_code, x_asr_codes, x_asr_dependencies,
    x_asr_dependency)


class ASRRecord:
    def __init__(self, record=None):
        self._record = record
        self._converted = False

    @property
    def archive(self):
        if self._archive is None:
            self._archive = EntryArchive()
        if not self._converted:
            self.to_archive()
        return self._archive

    @property
    def record(self):
        return self._record

    @record.setter
    def record(self, value):
        self._record = value
        self._archive = None
        self._converted = False

    def _parse_system(self, atoms):
        system = self._archive.run[-1].m_create(System)
        system.atoms = Atoms(
            positions=atoms.get_positions() * ureg.angstrom,
            lattice_vectors=atoms.get_cell().array * ureg.angstrom,
            periodic=[True, True, True],
            labels=atoms.get_chemical_symbols())

    def _parse_c2db_relax(self):
        result = self.record.result
        for image in result.images:
            self._parse_system(image)
        workflow = self._archive.m_create(Workflow)
        workflow.type = 'geometry_optimization'
        workflow.run_ref = self._archive.run[-1]
        self.archive.workflow2 = workflow2.GeometryOptimization()

        calc = self._archive.run[-1].m_create(Calculation)
        calc.system_ref = self._archive.run[-1].system[-1]
        calc.energy = Energy(total=EnergyEntry(value=result.etot * ureg.eV))
        calc.forces = Forces(total=ForcesEntry(value=result.forces * ureg.eV / ureg.angstrom))
        stress = np.zeros((3, 3))
        stress[0][0] = result.stress[0]
        stress[1][1] = result.stress[1]
        stress[2][2] = result.stress[2]
        stress[0][1] = stress[1][0] = result.stress[3]
        stress[0][2] = stress[2][0] = result.stress[4]
        stress[1][2] = stress[2][1] = result.stress[5]
        calc.stress = Stress(total=StressEntry(value=stress))

    def _parse_c2db_phonopy(self):
        workflow = self._archive.m_create(Workflow)
        workflow.type = 'phonon'
        workflow.run_ref = self._archive.run[-1]
        self.archive.workflow2 = workflow2.Phonon()

        bands = self.record.result.data.get('omega_kl')

        path = self.record.result.data.get('path')
        hisym_kpts = [list(p) for p in path.special_points.values()]
        labels = list(path.special_points.keys())
        endpoints = []
        calc = self._archive.run[-1].m_create(Calculation)
        bandstructure = calc.m_create(BandStructure, Calculation.band_structure_phonon)
        for i, qpoint in enumerate(path.kpts):
            if list(qpoint) in hisym_kpts:
                endpoints.append(i)
            if len(endpoints) < 2:
                continue
            sec_segment = bandstructure.m_create(BandEnergies)
            energies = bands[endpoints[0]: endpoints[1] + 1]
            sec_segment.energies = np.reshape(energies, (1, *np.shape(energies)))
            sec_segment.kpoints = path.kpts[endpoints[0]: endpoints[1] + 1]
            sec_segment.endpoints_labels = [labels[hisym_kpts.index(list(path.kpts[i]))] for i in endpoints]
            endpoints = [i]

    def _parse_run(self):
        run = self._archive.m_create(Run)
        run.program = Program(name='ASR', version=asr.__version__)

        if self.record.resources is not None:
            run.time_run = TimeRun(
                date_start=self.record.resources.execution_start,
                date_end=self.record.resources.execution_end)
        resources = run.m_create(x_asr_resources)
        for key, val in self.record.resources.__dict__.items():
            try:
                setattr(resources, 'x_asr_%s' % key, val)
            except Exception:
                pass

        if self.record.metadata is not None:
            metadata = run.m_create(x_asr_metadata)
            metadata.x_asr_created = (
                self.record.metadata.created - datetime.datetime(1970, 1, 1)).total_seconds()
            metadata.x_asr_modified = (
                self.record.metadata.modified - datetime.datetime(1970, 1, 1)).total_seconds()
            metadata.x_asr_directory = self.record.metadata.directory

        # misc
        for key, val in self.record.__dict__.items():
            if hasattr(val, '__dict__'):
                continue
            try:
                setattr(run, 'x_asr_%s' % key, val)
            except Exception:
                pass

        if self.record.run_specification is not None:
            # parse original system info
            atoms = self.record.run_specification.parameters.atoms
            self._parse_system(atoms)
            run_spec = run.m_create(x_asr_run_specification)
            for key, val in self.record.run_specification.__dict__.items():
                if hasattr(val, '__dict__'):
                    continue
                try:
                    setattr(run_spec, 'x_asr_%s' % key, val)
                except Exception:
                    pass
            parameters = run_spec.m_create(x_asr_parameters)
            for key, val in self.record.run_specification.parameters.__dict__.items():
                if hasattr(val, '__dict__'):
                    continue
                try:
                    setattr(parameters, 'x_asr_%s' % key, val)
                except Exception:
                    pass
            codes = run_spec.m_create(x_asr_codes)
            for entry in self.record.run_specification.codes.codes:
                code = codes.m_create(x_asr_code)
                code.x_asr_package = entry.package
                code.x_asr_version = entry.version
                code.x_asr_git_hash = entry.git_hash

        if self.record.dependencies is not None:
            dependencies = run.m_create(x_asr_dependencies)
            for dep in self.record.dependencies.deps:
                dependency = dependencies.m_create(x_asr_dependency)
                dependency.x_asr_uid = dep.uid
                dependency.x_asr_revision = dep.revision

    def to_archive(self):
        self._parse_run()
        if self.record.name == 'asr.c2db.relax':
            self._parse_c2db_relax()
        elif self.record.name == 'asr.c2db.phonopy':
            self._parse_c2db_phonopy()
        self._converted = True
        return self._archive


class ASRParser:
    def __init__(self):
        pass

    def parse(self, mainfile: str, archive: EntryArchive, logger=None):
        with open(mainfile, 'rt') as f:
            archive_data = json.load(f)
            archive.m_update_from_dict(archive_data)


def asr_to_archives(directory: str, recipes: List[str] = None):
    '''
    Converts the asr results for the specified recipes under the given directory to the
    nomad archive format.
    '''
    # record can only be fetched on the directory
    try:
        cwd = os.getcwd()
        os.chdir(directory)
        cache = get_cache()

        records: List[Record] = []
        if recipes is None:
            records = cache.select()
        else:
            for recipe in recipes:
                records.extend(cache.select(name=recipe))
        asr_record = ASRRecord()
        for record in records:
            asr_record.record = record
            with open('archive_%s.json' % record.uid, 'w') as f:
                json.dump(asr_record.archive.m_to_dict(), f, indent=4)
    except Exception:
        pass
    finally:
        os.chdir(cwd)
