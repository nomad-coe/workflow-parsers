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

            # optimised structures
            mof_atoms.x_mof_atoms_optimised = Atoms()
            parse_atoms(entry['Optimised'], mof_atoms.x_mof_atoms_optimised)

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

            for name, o_sbu in entry.get('Organic_sbu', dict()).items():
                for atom_indices in o_sbu.get('atom_indices_mapping', []):
                    atoms_group = AtomsGroup(atom_indices=atom_indices, label=name)
                    system.atoms_group.append(atoms_group)
                # add unique
                mof_atoms.x_mof_organic_sbu.append(atoms_group)

            return system

        # if this is already an archive, we simply parse the archive
        if 'run' in data:
            try:
                archive.m_update_from_dict(data)
                metadata_data = data.get(EntryArchive.metadata.name, None)
                # delete metadata
                if metadata_data is not None:
                    del (data[EntryArchive.metadata.name])

            except Exception:
                self.logger.error('Error parsing archive for MOF structure.')
            return

        header = data.pop('_HEADER')
        for name, entry in data.items():
            run = archive.m_create(Run)
            run.program = Program(name='MOF Structures', version=header.get('version', '0.0.1'))

            system = parse_system(entry)
            system.name = f'mof_{name}'
            metadata = entry.get('Metadata')
            system.x_mof_refcode = metadata.get('CSD_Refcode')
            system.x_mof_alias = metadata.get('alias')
            system.x_mof_source = metadata.get('Source')
            system.x_mof_ccdc_number = metadata.get('ccdc_number')
            system.x_mof_csd_deposition_date = metadata.get('CSD_Deposition_date')
            system.x_mof_chemical_name = metadata.get('chemical_name')
            system.x_mof_topology = metadata.get('Topology')
            system.x_mof_pld = metadata.get('PLD')
            system.x_mof_lcd = metadata.get('LCD')
            system.x_mof_lfpd = metadata.get('LFPD')
            system.x_mof_asa = metadata.get('ASA_m2_g')
            system.x_mof_nasa = metadata.get('NASA_m2_g')
            system.x_mof_density = metadata.get('Density')
            system.x_mof_volume = metadata.get('Volume_A3')
            system.x_mof_n_channel = metadata.get('N_channel')
            system.x_mof_space_group_symbol = metadata.get('Space_group_symbol')
            system.x_mof_space_group_number = metadata.get('Space_group_number')
            system.x_mof_point_group_symbol = metadata.get('Point_group_symbol')
            system.x_mof_crystal_system = metadata.get('Crystal_system')
            system.x_mof_hall_symbol = metadata.get('Hall_symbol')
            system.x_mof_charge = metadata.get('charge')
            system.x_mof_is_rod = metadata.get('Is_rod') == 'yes'
            system.x_mof_is_paddlewheel = metadata.get('is_paddlewheel')
            system.x_mof_is_ferrocene = metadata.get('is_ferrocene')
            system.x_mof_is_paddle_water = metadata.get('is_Paddle_water')
            system.x_mof_is_ui006 = metadata.get('is_UIO66')
            system.x_mof_is_mof32 = metadata.get('is_MOF32')
            system.x_mof_is_irmof = metadata.get('is_IRMOF')
            system.x_mof_doi = metadata.get('DOI')
            system.x_mof_citation = metadata.get('citation')
            system.x_mof_core_metal = metadata.get('Core_metal')
            system.x_mof_cn = metadata.get('CN')
            exp_condition = metadata.get('Exp_condition')
            system.x_mof_synthesis_method = exp_condition.get('synthesis_method')
            system.x_mof_linker_reagents = exp_condition.get('linker_reagents')
            system.x_mof_metal_reagents = exp_condition.get('metal_reagents')
            system.x_mof_temperature = exp_condition.get('temperature')
            system.x_mof_time_h = exp_condition.get('time_h')
            system.x_mof_yield_percent = exp_condition.get('Yield_Percent')
            system.x_mof_metal_os_1 = exp_condition.get('metal_os_1')
            system.x_mof_counterions1 = exp_condition.get('counterions1')
            system.x_mof_metal_os_2 = exp_condition.get('metal_os_2')
            system.x_mof_counterions2 = exp_condition.get('counterions2')
            system.x_mof_metal_os_3 = exp_condition.get('metal_os_3')
            system.x_mof_counterions3 = exp_condition.get('counterions3')
            system.x_mof_solvent1 = exp_condition.get('solvent1')
            system.x_mof_sol_molratio1 = exp_condition.get('sol_molratio1')
            system.x_mof_solvent2 = exp_condition.get('solvent2')
            system.x_mof_sol_molratio2 = exp_condition.get('sol_molratio2')
            system.x_mof_solvent3 = exp_condition.get('solvent3')
            system.x_mof_sol_molratio3 = exp_condition.get('sol_molratio3')
            system.x_mof_solvent4 = exp_condition.get('solvent4')
            system.x_mof_sol_molratio4 = exp_condition.get('sol_molratio4')
            system.x_mof_solvent5 = exp_condition.get('solvent5')
            system.x_mof_sol_molratio5 = exp_condition.get('sol_molratio5')
            system.x_mof_additive1 = exp_condition.get('additive1')
            system.x_mof_additive2 = exp_condition.get('additive2')
            system.x_mof_additive3 = exp_condition.get('additive3')
            system.x_mof_additive4 = exp_condition.get('additive4')
            system.x_mof_additive5 = exp_condition.get('additive5')
            # add metadata
            sec_method = run.m_create(Method)
            sec_method.x_mof_metadata = metadata

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
