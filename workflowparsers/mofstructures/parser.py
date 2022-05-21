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
import logging
import json

from nomad.units import ureg
from nomad.datamodel import EntryArchive
from nomad.datamodel.metainfo.simulation.run import Run, Program
from nomad.datamodel.metainfo.simulation.system import System, Atoms, AtomsGroup
from nomad.datamodel.metainfo.simulation.method import Method
from nomad.datamodel.metainfo.workflow import Workflow
from workflowparsers.mofstructures.metainfo.mofstructures import x_mof_atoms


class MOFStructuresParser:

    def parse(self, filepath, archive, logger):
        self.filepath = os.path.abspath(filepath)
        self.archive = archive
        self.logger = logging.getLogger(__name__) if logger is None else logger
        self.maindir = os.path.dirname(self.filepath)

        try:
            with open(self.filepath) as f:
                data = json.load(f)
        except Exception:
            data = dict()

        def parse_atoms(source, target):
            target.labels = source.get('chemical_symbols')
            if source.get('positions') is not None:
                target.positions = source.get('positions') * ureg.angstrom
            if source.get('cell') is not None:
                target.lattice_vectors = source.get('cell') * ureg.angstrom

        def parse_system(entry):
            # Optimised is the MOF structure
            system = archive.run[-1].m_create(System)
            if entry.get('Optimised') is None:
                return

            system.atoms = Atoms()
            parse_atoms(entry['Optimised'], system.atoms)

            mof_atoms = system.m_create(x_mof_atoms)

            # add the reference structure from experiment
            if entry.get('Experiment') is not None:
                mof_atoms.x_mof_atoms_experiment = Atoms()
                parse_atoms(entry['Experiment'], mof_atoms.x_mof_atoms_experiment)

            # add structure building units
            # only unique sbus are included
            for name, sbu in entry.get('sbus', dict()).items():
                # mapping of sbus to the mof, atom_indices_mapping lists copies of sbu in mof
                for atom_indices in sbu.get('atom_indices_mapping', []):
                    atoms_group = AtomsGroup(atom_indices=atom_indices, label=name)
                    system.atoms_group.append(atoms_group)
                # add unique
                mof_atoms.x_mof_sbu.append(atoms_group)

            # add linkers
            for name, linker in entry.get('linkers', dict()).items():
                for atom_indices in linker.get('atom_indices_mapping', []):
                    atoms_group = AtomsGroup(atom_indices=atom_indices, label=name)
                    system.atoms_group.append(atoms_group)
                # add unique
                mof_atoms.x_mof_linker.append(atoms_group)

            return system

        # if this is already an archive, we simply parse the archive
        if 'run' in data:
            try:
                archive.m_update_from_dict(data)
                metadata_data = data.get(EntryArchive.metadata.name, None)
                # delete metadata
                if metadata_data is not None:
                    del(data[EntryArchive.metadata.name])

            except Exception:
                self.logger.error('Error parsing archive for MOF structure.')
            return

        header = data.pop('_HEADER')
        for name, entry in data.items():
            run = archive.m_create(Run)
            run.program = Program(name='MOF Structures', version=header.get('version', '0.0.1'))

            system = parse_system(entry)
            system.name = f'mof_{name}'

            # add metadata
            sec_method = run.m_create(Method)
            sec_method.x_mof_metadata = entry.get('Metadata')

            workflow = self.archive.m_create(Workflow)
            workflow.type = 'single_point'
            workflow.run_ref = run
            # add path to external calculation
            path = f'mof_{name}_calculation.out'
            try:
                archive_ref = archive.m_context.resolve_archive(f'../upload/archive/mainfile/{path}')
                self.logger.info(f'reference archive {archive_ref.workflow[0].calculations_ref[0].energy.total.value}')
                archive.workflow[-1].workflows_ref = [archive_ref.workflow[0]]
            except Exception:
                self.logger.warn('Could not resolve referenced calculations.')
