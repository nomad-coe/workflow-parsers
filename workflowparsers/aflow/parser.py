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
import numpy as np
import logging
import json
from io import StringIO
from ase.cell import Cell
from ase.io import vasp

from nomad.units import ureg
from nomad.parsing.file_parser import TextParser, Quantity
from nomad.datamodel.metainfo.simulation.run import Run, Program
from nomad.datamodel.metainfo.simulation.calculation import (
    Calculation, Energy, EnergyEntry, Forces, ForcesEntry, Stress, StressEntry,
    Thermodynamics, Dos, DosValues, BandStructure, BandEnergies)
from nomad.datamodel.metainfo.simulation.method import Method
from nomad.datamodel.metainfo.simulation.system import System, Atoms
from nomad.datamodel.metainfo.workflow import (
    Workflow, Elastic, DebyeModel, Phonon, Thermodynamics as WorkflowThermodynamics)
from nomad.datamodel.metainfo.simulation import workflow as workflow2


from .metainfo import m_env


class AflowOutParser(TextParser):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def init_quantities(self):

        def str_to_property(val_in):
            val = val_in.split('=')
            return val[0].strip().replace(' ', '_').lower(), val[-1].split('//')[0].strip()

        self._quantities = [
            Quantity(
                'property',
                r'\n *\[(.+)\](.+?=.+)', str_operation=str_to_property, repeats=True),
            Quantity(
                'section',
                r'(\[.+?\]START[\s\S]+?\]STOP)',
                repeats=True, sub_parser=TextParser(quantities=[
                    Quantity(
                        'name',
                        r'\[(.+)\]START', str_operation=lambda x: x.lower(), dtype=str),
                    Quantity(
                        'key_value',
                        r'\n *([^#]\S+)=(\S+)', repeats=True),
                    Quantity(
                        'array',
                        rf'\n *(\d[\s\S]+?\d\s*)\[.+?STOP', dtype=np.dtype(np.float64))]))]

    def parse(self, key=None):
        super().parse(key)
        for property in self._results.get('property', []):
            self._results[property[0]] = property[1]
        for section in self._results.get('section', []):
            if section.key_value is not None:
                result = dict()
                for key, value in section.get('key_value', []):
                    result[key] = value
                self._results[section.name] = result
            elif section.array is not None:
                self._results[section.name] = section.array


class AflowInParser(AflowOutParser):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def init_quantities(self):
        super().init_quantities()
        self._quantities += [
            Quantity(
                'aflow_version',
                r'Stefano Curtarolo \- \(AFLOW V([\d\.]+)\)'),
            Quantity(
                'poscar',
                r'\[VASP_POSCAR_MODE_EXPLICIT\]START\s*([\s\S]+?)\[VASP_POSCAR_MODE_EXPLICIT\]STOP',
                str_operation=lambda x: x, convert=False, repeats=True),
            Quantity(
                'aflow_composition',
                r'\[AFLOW\] COMPOSITION=(\S+)', sub_parser=TextParser(quantities=[
                    Quantity('species', r'([A-Z][a-z]*)', repeats=True, dtype=str),
                    Quantity('composition', r'(\d+)\|', repeats=True, dtype=np.dtype(np.int32))
                ])
            )
        ] + [Quantity(
            module.lower(),
            r'\n *\[AFLOW\_%s\]CALC([\s\S]+?)\[AFLOW\] \*' % module, sub_parser=TextParser(quantities=[
                Quantity(
                    'parameters',
                    r'\[AFLOW\_%s\](.+?)=(\S+)' % module, repeats=True)])) for module in [
                        'AEL', 'AGL', 'APL', 'QHA', 'AAPL']]

    def parse(self, key=None):
        super().parse(key)

        if self.get('poscar') is not None and self._results.get('geometry') is None:
            try:
                atoms = vasp.read_vasp(StringIO(self._results['poscar'][-1]))
                self._results['cell'] = atoms.get_cell()
                self._results['geometry'] = atoms.get_cell().cellpar()
                composition = self._results['aflow_composition']
                self._results['species'] = composition.species
                self._results['composition'] = [int(c) for c in composition._results['composition']]
                self._results['positions_cartesian'] = atoms.get_positions()
            except Exception:
                pass

        if self._results.get('loop') is None:
            self._results['loop'] = [module for module in [
                'ael', 'agl', 'apl', 'qha', 'aapl'] if module in self._results]


class AFLOWParser:
    def __init__(self):
        self.ael_parser = AflowOutParser()
        self.agl_parser = AflowOutParser()
        self.apl_parser = AflowOutParser()
        self.aflowin_parser = AflowInParser()
        self._metainfo_env = m_env

        self._metainfo_map = {
            'stiffness_tensor': 'elastic_constants_matrix_second_order',
            'compliance_tensor': 'compliance_matrix_second_order',
            'poisson_ratio': 'poisson_ratio_hill',
            'bulk_modulus_vrh': 'bulk_modulus_hill',
            'shear_modulus_vrh': 'shear_modulus_hill',
            'youngs_modulus_vrh': 'Young_modulus_hill',
            'pughs_modulus_ratio': 'pugh_ratio_hill', 'applied_pressure': 'x_aflow_ael_applied_pressure',
            'average_external_pressure': 'x_aflow_ael_average_external_pressure'
        }

    def init_parser(self):
        if '.json' in self.filepath:
            self.aflow_data = json.load(open(self.filepath))
        else:
            self.aflowin_parser.mainfile = self.filepath
            self.aflow_data = self.aflowin_parser

    def get_aflow_file(self, filename):
        files = [f for f in os.listdir(self.maindir) if filename in f]
        if not files:
            files = ['']
        return os.path.join(self.maindir, files[0])

    def parse_structures(self, module):
        try:
            structures = json.load(open(os.path.join(
                self.maindir, '%s_energy_structures.json' % module))).get('%s_energy_structures' % module, [])
        except Exception:
            structures = []

        for structure in structures:
            sec_calc = self.archive.run[-1].m_create(Calculation)
            sec_thermo = sec_calc.m_create(Thermodynamics)
            if structure.get('energy') is not None:
                sec_calc.energy = Energy(total=EnergyEntry(value=structure.get('energy') * ureg.eV))
            if structure.get('pressure') is not None:
                sec_thermo.pressure = structure.get('pressure') * ureg.kbar
            if structure.get('stress_tensor') is not None:
                sec_calc.stress = Stress(total=StressEntry(value=structure.get('stress_tensor') * ureg.kbar))
            if structure.get('structure') is not None:
                sec_system = self.archive.run[-1].m_create(System)
                sec_system.atoms = Atoms()
                struc = structure.get('structure')
                sec_system.atoms.labels = [atom.get('name') for atom in struc.get('atoms', [])]
                sec_system.atoms.concentrations = [atom.get('occupancy') for atom in struc.get('atoms', [])]
                if struc.get('lattice') is not None:
                    sec_system.atoms.lattice_vectors = struc.get(
                        'lattice') * ureg.angstrom * struc.get('scale', 1)
                positions = [atom.get('position') for atom in struc.get('atoms', [])]
                if struc.get('coordinates_type', 'direct').lower().startswith('d'):
                    if sec_system.atoms.lattice_vectors is not None:
                        positions = np.dot(positions, sec_system.atoms.lattice_vectors)
                sec_system.atoms.positions = positions

    def parse_agl(self):
        sec_run = self.archive.m_create(Run)
        sec_run.program = Program(
            name='AFlow', version=self.aflow_data.get('aflow_version', 'unknown'))

        self.parse_structures('AGL')

        self.agl_parser.mainfile = self.get_aflow_file('aflow.agl.out')
        thermal_properties = self.agl_parser.get('agl_thermal_properties_temperature')
        if thermal_properties is None:
            return

        sec_workflow = self.archive.m_create(Workflow)
        sec_workflow.run_ref = sec_run
        sec_workflow.type = 'debye_model'
        sec_debye = sec_workflow.m_create(DebyeModel)
        sec_thermo = sec_workflow.m_create(WorkflowThermodynamics)

        thermal_properties = np.reshape(thermal_properties, (len(thermal_properties) // 9, 9))
        thermal_properties = np.transpose(thermal_properties)
        energies = self.agl_parser.get('agl_energies_temperature')
        energies = np.reshape(energies, (len(energies) // 9, 9))
        energies = np.transpose(energies)

        sec_thermo.temperature = thermal_properties[0] * ureg.K
        sec_thermo.gibbs_free_energy = energies[1] * ureg.eV
        sec_thermo.vibrational_free_energy = energies[2] * ureg.meV
        sec_thermo.vibrational_internal_energy = energies[3] * ureg.meV
        sec_thermo.vibrational_entropy = energies[4] * ureg.meV / ureg.K
        sec_thermo.heat_capacity_c_v = thermal_properties[4] * ureg.boltzmann_constant
        sec_thermo.heat_capacity_c_p = thermal_properties[5] * ureg.boltzmann_constant
        sec_debye.thermal_conductivity = thermal_properties[1] * ureg.watt / ureg.m * ureg.K
        sec_debye.debye_temperature = thermal_properties[2] * ureg.K
        sec_debye.gruneisen_parameter = thermal_properties[3]
        sec_debye.thermal_expansion = thermal_properties[6] / ureg.K
        sec_debye.bulk_modulus_static = thermal_properties[7] * ureg.GPa
        sec_debye.bulk_modulus_isothermal = thermal_properties[8] * ureg.GPa

    def parse_ael(self):
        sec_run = self.archive.m_create(Run)
        sec_run.program = Program(
            name='AFlow', version=self.aflow_data.get('aflow_version', 'unknown'))

        self.parse_structures('AEL')

        self.ael_parser.mainfile = self.get_aflow_file('aflow.ael.out')
        sec_workflow = self.archive.m_create(Workflow)
        workflow = workflow2.Elastic(method=workflow2.ElasticMethod(), results=workflow2.ElasticResults())
        sec_workflow.run_ref = sec_run
        sec_workflow.type = 'elastic'
        sec_elastic = sec_workflow.m_create(Elastic)
        sec_elastic.energy_stress_calculator = 'vasp'
        sec_elastic.calculation_method = 'stress'
        sec_elastic.elastic_constants_order = 2
        workflow.method.energy_stress_calculator = 'vasp'
        workflow.method.calculation_method = 'stress'
        workflow.method.elastic_constants_order = 2

        paths = [d for d in self.aflow_data.get('files', []) if d.startswith('ARUN.AEL')]
        deforms = np.array([d.split('_')[-2:] for d in paths], dtype=np.dtype(np.float64))
        strains = [d[1] for d in deforms if d[0] == 1]
        sec_elastic.n_deformations = int(max(np.transpose(deforms)[0]))
        sec_elastic.n_strains = len(strains)
        sec_elastic.strain_maximum = max(strains) - 1.0
        workflow.results.n_deformations = int(max(np.transpose(deforms)[0]))
        workflow.results.n_strains = len(strains)
        workflow.method.strain_maximum = max(strains) - 1.0

        for key, val in self.ael_parser.get('ael_results', {}).items():
            key = key.replace('ael_', '')
            key = self._metainfo_map.get(key, key)
            if 'modulus' in key or 'pressure' in key:
                val = val * ureg.GPa
            elif 'speed' in key:
                val = val * (ureg.m / ureg.s)
            elif 'temperature' in key:
                val = val * ureg.K
            setattr(sec_elastic, key, val)
            setattr(workflow.results, key, val)

        if self.ael_parser.ael_stiffness_tensor is not None:
            sec_elastic.elastic_constants_matrix_second_order = np.reshape(
                self.ael_parser.ael_stiffness_tensor, (6, 6)) * ureg.GPa

        if self.ael_parser.ael_compliance_tensor is not None:
            sec_elastic.compliance_matrix_second_order = np.reshape(
                self.ael_parser.ael_compliance_tensor, (6, 6))

        self.archive.workflow2 = workflow

    def parse_apl(self):
        sec_run = self.archive.m_create(Run)
        sec_run.program = Program(
            name='AFlow', version=self.aflow_data.get('aflow_version', 'unknown'))
        sec_scc = sec_run.m_create(Calculation)

        try:
            dos = np.transpose(
                np.loadtxt(self.get_aflow_file('flow.apl.phonon_dos.out.xz')))
        except Exception:
            dos = None

        if dos is not None:
            sec_dos = sec_scc.m_create(Dos, Calculation.dos_phonon)
            sec_dos.energies = dos[2] * ureg.millielectron_volt
            sec_dos.total.append(DosValues(value=dos[3] * (1 / ureg.millielectron_volt)))

        try:
            kpoints = np.transpose(
                np.loadtxt(self.get_aflow_file('aflow.apl.hskpoints.out.xz')))
            n_kpoints = int(max(kpoints[3])) + 1
            kpoints = kpoints[:3]
            kpoints = np.reshape(kpoints, (3, len(kpoints[0]) // n_kpoints, n_kpoints))
            kpoints = np.transpose(kpoints, axes=(1, 2, 0))

            bandstructure = np.transpose(
                np.loadtxt(self.get_aflow_file('aflow.apl.phonon_dispersion.out.xz')))
            bandstructure = bandstructure[2:]
            bandstructure = np.reshape(bandstructure, (
                len(bandstructure), len(bandstructure[0]) // n_kpoints, n_kpoints))
            bandstructure = np.transpose(bandstructure, axes=(1, 2, 0))
        except Exception:
            kpoints = None

        if kpoints is not None:
            sec_bandstructure = sec_scc.m_create(BandStructure, Calculation.band_structure_phonon)
            for n_segment in range(len(kpoints)):
                sec_segment = sec_bandstructure.m_create(BandEnergies)
                sec_segment.kpoints = kpoints[n_segment]
                sec_segment.energies = np.reshape(
                    bandstructure[n_segment], (1, *np.shape(bandstructure[n_segment]))) * ureg.millielectron_volt

        self.apl_parser.mainfile = self.get_aflow_file('aflow.apl.thermodynamic_properties.out')

        sec_workflow = self.archive.m_create(Workflow)
        sec_workflow.type = 'phonon'
        sec_workflow.run_ref = sec_run
        workflow = workflow2.Phonon(method=workflow2.PhononMethod(), results=workflow2.PhononResults())

        sec_phonon = sec_workflow.m_create(Phonon)
        sec_phonon.force_calculator = 'vasp'
        workflow.method.force_calculator = 'vasp'
        mesh = self.aflowin_parser.get('aflow_apl_dosmesh')
        if mesh is not None:
            try:
                cell = Cell.fromcellpar(self.aflowin_parser.geometry)
                sec_phonon.mesh_density = np.product([int(m) for m in mesh.split('x')]) / cell.volume
                workflow.method.mesh_density = np.product([int(m) for m in mesh.split('x')]) / cell.volume
            except Exception:
                pass

        self.apl_parser.mainfile = self.get_aflow_file('aflow.apl.group_velocities.out')
        group_velocity = self.apl_parser.get('apl_group_velocity')
        if group_velocity is not None:
            try:
                qpoints = self.apl_parser.apl_qpoints
                qpoints = np.reshape(qpoints, (len(qpoints) // 4, 4))
                sec_phonon.qpoints = np.transpose(np.transpose(qpoints)[1:])
                group_velocity = np.reshape(
                    group_velocity, (len(qpoints), len(group_velocity) // len(qpoints)))
                group_velocity = np.transpose(np.transpose(group_velocity)[1:])
                sec_phonon.group_velocity = np.reshape(group_velocity, (
                    len(group_velocity), len(group_velocity[0]) // 3, 3)) * ureg.kilometer / ureg.second
                workflow.results.qpoints = np.transpose(np.transpose(qpoints)[1:])
                workflow.results.group_velocity = np.reshape(group_velocity, (
                    len(group_velocity), len(group_velocity[0]) // 3, 3)) * ureg.kilometer / ureg.second
            except Exception:
                pass

        self.apl_parser.mainfile = self.get_aflow_file('aflow.apl.thermodynamic_properties.out.xz')
        apl_thermo = self.apl_parser.get('apl_thermo')
        if apl_thermo is not None:
            apl_thermo = np.transpose(np.reshape(apl_thermo, (len(apl_thermo) // 6, 6)))
            sec_thermo = sec_workflow.m_create(WorkflowThermodynamics)
            sec_thermo.temperature = apl_thermo[0] * ureg.kelvin
            sec_thermo.internal_energy = apl_thermo[2] * ureg.millielectron_volt
            sec_thermo.helmholtz_free_energy = apl_thermo[3] * ureg.millielectron_volt
            sec_thermo.entropy = apl_thermo[4] * ureg.boltzmann_constant
            sec_thermo.heat_capacity_c_v = apl_thermo[5] * ureg.boltzmann_constant
            workflow.results.temperature = apl_thermo[0] * ureg.kelvin
            workflow.results.internal_energy = apl_thermo[2] * ureg.millielectron_volt
            workflow.results.helmholtz_free_energy = apl_thermo[3] * ureg.millielectron_volt
            workflow.results.entropy = apl_thermo[4] * ureg.boltzmann_constant
            workflow.results.heat_capacity_c_v = apl_thermo[5] * ureg.boltzmann_constant

        self.archive.workflow2 = workflow

        # TODO parse systems for each displacements

        # TODO parse displacements, force constants, dynamical matrix

    def parse(self, filepath, archive, logger):
        self.filepath = os.path.abspath(filepath)
        self.archive = archive
        self.maindir = os.path.dirname(self.filepath)
        self.logger = logger if logger is not None else logging

        self.init_parser()

        sec_run = self.archive.m_create(Run)
        sec_run.program = Program(
            name='AFlow', version=self.aflow_data.get('aflow_version', 'unknown'))

        # parse run metadata
        run_quantities = ['aurl', 'auid', 'data_api', 'data_source', 'loop']
        for key in run_quantities:
            val = self.aflow_data.get(key)
            if val is not None:
                setattr(sec_run, 'x_aflow_%s' % key, val)

        # TODO The OUTCAR file will be read by the vasp parser and so the complete
        # metadata for both system and method should be filled in by vasp parser.
        # parse structure from aflow_data
        sec_system = sec_run.m_create(System)
        sec_system.atoms = Atoms()
        lattice_parameters = self.aflow_data.get('geometry')
        if lattice_parameters is not None:
            cell = self.aflow_data.get('cell', Cell.fromcellpar(lattice_parameters))
            sec_system.atoms.lattice_vectors = cell.array * ureg.angstrom
            sec_system.atoms.periodic = [True, True, True]
        species = self.aflow_data.get('species', [])
        atom_labels = []
        for n, specie in enumerate(species):
            atom_labels += [specie] * self.aflow_data['composition'][n]
        sec_system.atoms.labels = atom_labels
        sec_system.atoms.positions = self.aflow_data.get('positions_cartesian', []) * ureg.angstrom

        # parse system metadata from aflow_data
        system_quantities = [
            'compound', 'prototype', 'nspecies', 'natoms', 'natoms_orig', 'composition',
            'density', 'density_orig', 'scintillation_attenuation_length', 'stoichiometry',
            'species', 'geometry', 'geometry_orig', 'volume_cell', 'volume_atom',
            'volume_cell_orig', 'volume_atom_orig', 'n_sg', 'sg', 'sg2', 'spacegroup_orig',
            'spacegroup_relax', 'Bravais_lattice_orig', 'lattice_variation_orig',
            'lattice_system_orig', 'Pearson_symbol_orig', 'Bravais_lattice_relax',
            'lattice_variation_relax', 'lattice_system_relax', 'Pearson_symbol_relax',
            'crystal_family_orig', 'crystal_system_orig', 'crystal_class_orig',
            'point_group_Hermann_Mauguin_orig', 'point_group_Schoenflies_orig',
            'point_group_orbifold_orig', 'point_group_type_orig', 'point_group_order_orig',
            'point_group_structure_orig', 'Bravais_lattice_lattice_type_orig',
            'Bravais_lattice_lattice_variation_type_orig', 'Bravais_lattice_lattice_system_orig',
            'Bravais_superlattice_lattice_type_orig', 'Bravais_superlattice_lattice_variation_type_orig',
            'Bravais_superlattice_lattice_system_orig', 'Pearson_symbol_superlattice_orig',
            'reciprocal_geometry_orig', 'reciprocal_volume_cell_orig', 'reciprocal_lattice_type_orig',
            'reciprocal_lattice_variation_type_orig', 'Wyckoff_letters_orig',
            'Wyckoff_multiplicities_orig', 'Wyckoff_site_symmetries_orig',
            'crystal_family', 'crystal_system', 'crystal_class', 'point_group_Hermann_Mauguin',
            'point_group_Schoenflies', 'point_group_orbifold', 'point_group_type', 'point_group_order',
            'point_group_structure', 'Bravais_lattice_lattice_type', 'Bravais_lattice_lattice_variation_type',
            'Bravais_lattice_lattice_system', 'Bravais_superlattice_lattice_type',
            'Bravais_superlattice_lattice_variation_type', 'Bravais_superlattice_lattice_system',
            'Pearson_symbol_superlattice', 'reciprocal_geometry', 'reciprocal_volume_cell',
            'reciprocal_lattice_type', 'reciprocal_lattice_variation_type', 'Wyckoff_letters',
            'Wyckoff_multiplicities', 'Wyckoff_site_symmetries', 'prototype_label_orig',
            'prototype_params_list_orig', 'prototype_params_values_orig', 'prototype_label_relax',
            'prototype_params_list_relax', 'prototype_params_values_relax']
        for key in system_quantities:
            val = self.aflow_data.get(key)
            if val is not None:
                if 'Wyckoff' in key:
                    val = np.array(val, dtype=object)
                setattr(sec_system, 'x_aflow_%s' % key, val)

        # parse method metadata from self.aflow_data
        method_quantities = [
            'code', 'species_pp', 'n_dft_type', 'dft_type', 'dft_type', 'species_pp_version',
            'species_pp_ZVAL', 'species_pp_AUID', 'ldau_type', 'ldau_l', 'ldau_u', 'ldau_j',
            'valence_cell_iupac', 'valence_cell_std', 'energy_cutoff',
            'delta_electronic_energy_convergence', 'delta_electronic_energy_threshold',
            'kpoints_relax', 'kpoints_static', 'n_kpoints_bands_path', 'kpoints_bands_path',
            'kpoints_bands_nkpts']
        sec_method = sec_run.m_create(Method)
        for key in method_quantities:
            val = self.aflow_data.get(key)
            if val is not None:
                setattr(sec_method, 'x_aflow_%s' % key, val)

        # parse basic calculation quantities from self.aflow_data
        sec_scc = sec_run.m_create(Calculation)
        sec_scc.energy = Energy()
        sec_scc.forces = Forces()
        sec_thermo = sec_scc.m_create(Thermodynamics)
        if self.aflow_data.get('energy_cell') is not None:
            sec_scc.energy.total = EnergyEntry(value=self.aflow_data['energy_cell'] * ureg.eV)
        if self.aflow_data.get('forces') is not None:
            sec_scc.forces.total = ForcesEntry(value=self.aflow_data['forces'] * ureg.eV / ureg.angstrom)
        if self.aflow_data.get('enthalpy_cell') is not None:
            sec_thermo.enthalpy = self.aflow_data['enthalpy_cell'] * ureg.eV
        if self.aflow_data.get('entropy_cell') is not None:
            sec_thermo.entropy = self.aflow_data['entropy_cell'] * ureg.eV / ureg.K
        if self.aflow_data.get('calculation_time') is not None:
            sec_scc.time_calculation = self.aflow_data['calculation_time'] * ureg.s
        calculation_quantities = [
            'stress_tensor', 'pressure_residual', 'Pulay_stress', 'Egap', 'Egap_fit', 'Egap_type',
            'enthalpy_formation_cell', 'entropic_temperature', 'PV', 'spin_cell', 'spinD',
            'spinF', 'calculation_memory', 'calculation_cores', 'nbondxx',
            'agl_thermal_conductivity_300K', 'agl_debye', 'agl_acoustic_debye', 'agl_gruneisen',
            'agl_heat_capacity_Cv_300K', 'agl_heat_capacity_Cp_300K', 'agl_thermal_expansion_300K',
            'agl_bulk_modulus_static_300K', 'agl_bulk_modulus_isothermal_300K', 'agl_poisson_ratio_source',
            'agl_vibrational_free_energy_300K_cell', 'agl_vibrational_free_energy_300K_atom',
            'agl_vibrational_entropy_300K_cell', 'agl_vibrational_entropy_300K_atom',
            'ael_poisson_ratio', 'ael_bulk_modulus_voigt', 'ael_bulk_modulus_reuss',
            'ael_shear_modulus_voigt', 'ael_shear_modulus_reuss', 'ael_bulk_modulus_vrh',
            'ael_shear_modulus_vrh', 'ael_elastic_anisotropy', 'ael_youngs_modulus_vrh',
            'ael_speed_sound_transverse', 'ael_speed_sound_longitudinal', 'ael_speed_sound_average',
            'ael_pughs_modulus_ratio', 'ael_debye_temperature', 'ael_applied_pressure',
            'ael_average_external_pressure', 'ael_stiffness_tensor', 'ael_compliance_tensor',
            'bader_net_charges', 'bader_atomic_volumes', 'n_files', 'files', 'node_CPU_Model',
            'node_CPU_Cores', 'node_CPU_MHz', 'node_RAM_GB', 'catalog', 'aflowlib_version',
            'aflowlib_date']
        for key in calculation_quantities:
            val = self.aflow_data.get(key)
            if val is not None:
                setattr(sec_scc, 'x_aflow_%s' % key, val)

        for module in self.aflow_data.get('loop', []):
            if module == 'ael':
                self.parse_ael()
            elif module == 'agl':
                self.parse_agl()
            elif module == 'apl':
                self.parse_apl()
