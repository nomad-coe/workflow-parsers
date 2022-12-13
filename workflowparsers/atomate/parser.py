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
import numpy as np

from nomad.units import ureg
from nomad.datamodel.metainfo.simulation.run import Run, Program
from nomad.datamodel.metainfo.workflow import (
    Workflow, Elastic, EquationOfState, EOSFit, Thermodynamics, Stability, Decomposition,
    Phonon)
from nomad.datamodel.metainfo.simulation import workflow as workflow2
from nomad.datamodel.metainfo.simulation.system import System, Atoms
from nomad.datamodel.metainfo.simulation.method import (
    Method, DFT, Electronic, XCFunctional, Functional, BasisSet, BasisSetCellDependent)
from nomad.datamodel.metainfo.simulation.calculation import (
    Calculation, Dos, DosValues, BandStructure, BandEnergies)
from .metainfo.atomate import Composition, Symmetry


class AtomateParser:
    def __init__(self):
        # NOTE This is not technically the atomate parser but for the json file materials
        # project produces for its workflow runs. They use atomate for workflow and the
        # emmet package for post-processing.
        # TODO extend this to read atomate output. There are probably some deviations in
        # names.
        pass

    def init_parser(self):
        try:
            self.data = json.load(open(self.filepath))
        except Exception:
            self.logger.error('Failed to load json file.')

    def parse_elastic(self, source):
        sec_workflow = self.archive.m_create(Workflow)
        sec_workflow.type = 'elastic'
        sec_elastic = sec_workflow.m_create(Elastic)
        workflow = workflow2.Elastic(method=workflow2.ElasticMethod(), results=workflow2.ElasticResults())
        sec_elastic.energy_stress_calculator = 'VASP'
        sec_elastic.calculation_method = 'stress'
        source = source.get('elasticity', source)
        sec_elastic.elastic_constants_order = source.get('order', 2)
        workflow.method.energy_stress_calculator = 'VASP'
        workflow.method.calculation_method = 'stress'
        workflow.method.elastic_constants_order = source.get('order', 2)

        deformations = source.get('deformations')
        if deformations is not None:
            sec_elastic.n_deformations = len(deformations)
            workflow.results.n_deformations = len(deformations)

        elastic_tensor = source.get('elastic_tensor')
        if elastic_tensor is not None:
            sec_elastic.elastic_constants_matrix_second_order = elastic_tensor * ureg.GPa
            workflow.results.elastic_constants_matrix_second_order = elastic_tensor * ureg.GPa

        compliance_tensor = source.get('compliance_tensor')
        if compliance_tensor is not None:
            sec_elastic.compliance_matrix_second_order = compliance_tensor * (1 / ureg.GPa)
            workflow.results.compliance_matrix_second_order = compliance_tensor * (1 / ureg.GPa)

        if source.get('g_reuss') is not None:
            sec_elastic.shear_modulus_reuss = source['g_reuss'] * ureg.GPa
            workflow.results.shear_modulus_reuss = source['g_reuss'] * ureg.GPa
        if source.get('g_voigt') is not None:
            sec_elastic.shear_modulus_voigt = source['g_voigt'] * ureg.GPa
            workflow.results.shear_modulus_voigt = source['g_voigt'] * ureg.GPa
        if source.get('g_vrh') is not None:
            sec_elastic.shear_modulus_hill = source['g_vrh'] * ureg.GPa
            workflow.results.shear_modulus_hill = source['g_vrh'] * ureg.GPa
        if source.get('homogeneous_poisson') is not None:
            sec_elastic.poisson_ratio_hill = source['homogeneous_poisson']
            workflow.results.poisson_ratio_hill = source['homogeneous_poisson']
        if source.get('k_reuss') is not None:
            sec_elastic.bulk_modulus_reuss = source['k_reuss'] * ureg.GPa
            workflow.results.bulk_modulus_reuss = source['k_reuss'] * ureg.GPa
        if source.get('k_voigt') is not None:
            sec_elastic.bulk_modulus_voigt = source['k_voigt'] * ureg.GPa
            workflow.results.bulk_modulus_voigt = source['k_voigt'] * ureg.GPa
        if source.get('k_vrh') is not None:
            sec_elastic.bulk_modulus_hill = source['k_vrh'] * ureg.GPa
            workflow.results.bulk_modulus_hill = source['k_vrh'] * ureg.GPa
        self.archive.workflow2 = workflow

    def parse_eos(self, source):
        sec_workflow = self.archive.m_create(Workflow)
        sec_workflow.type = 'equation_of_state'
        workflow = workflow2.EquationOfState(
            method=workflow2.EquationOfStateMethod(), results=workflow2.EquationOfStateResults())
        sec_eos = sec_workflow.m_create(EquationOfState)
        if source.get('volumes') is not None:
            sec_eos.volumes = source['volumes'] * ureg.angstrom ** 3
            workflow.results.volumes = source['volumes'] * ureg.angstrom ** 3
        if source.get('energies') is not None:
            sec_eos.energies = source['energies'] * ureg.eV
            workflow.results.energies = source['energies'] * ureg.eV
        for fit_function, result in source.get('eos', {}).items():
            sec_eos_fit = sec_eos.m_create(EOSFit)
            sec_eos_fit2 = workflow.results.m_create(workflow2.EOSFit)
            sec_eos_fit.function_name = fit_function
            sec_eos_fit2.function_name = fit_function
            if result.get('B') is not None:
                sec_eos_fit.bulk_modulus = result['B'] * ureg.eV / ureg.angstrom ** 3
                sec_eos_fit2.bulk_modulus = result['B'] * ureg.eV / ureg.angstrom ** 3
            if result.get('C') is not None:
                sec_eos_fit.bulk_modulus_derivative = result['C']
                sec_eos_fit2.bulk_modulus_derivative = result['C']
            if result.get('E0') is not None:
                sec_eos_fit.equilibrium_energy = result['E0'] * ureg.eV
                sec_eos_fit2.equilibrium_energy = result['E0'] * ureg.eV
            if result.get('V0') is not None:
                sec_eos_fit.equilibrium_volume = result['V0'] * ureg.angstrom ** 3
                sec_eos_fit2.equilibrium_volume = result['V0'] * ureg.angstrom ** 3
            if result.get('eos_energies') is not None:
                sec_eos_fit.fitted_energies = result['eos_energies'] * ureg.eV
                sec_eos_fit2.fitted_energies = result['eos_energies'] * ureg.eV
        self.archive.workflow2 = workflow

    def parse_thermo(self, data):
        sec_workflow = self.archive.m_create(Workflow)
        workflow = self.archive.workflow2
        if not workflow:
            workflow = workflow2.Thermodynamics()
        if not workflow.results:
            workflow.results = workflow2.ThermodynamicsResults()
        sec_workflow.type = 'thermodynamics'
        sec_thermo = sec_workflow.m_create(Thermodynamics)
        sec_stability = sec_thermo.m_create(Stability)
        sec_stability2 = workflow2.Stability()
        sec_stability.formation_energy = data.get(
            'formation_energy_per_atom', 0) * data.get('nsites', 1) * ureg.eV
        sec_stability2.formation_energy = data.get(
            'formation_energy_per_atom', 0) * data.get('nsites', 1) * ureg.eV
        sec_stability.delta_formation_energy = data.get('energy_above_hull', 0) * ureg.eV
        sec_stability2.delta_formation_energy = data.get('energy_above_hull', 0) * ureg.eV
        sec_stability.is_stable = data.get('is_stable')
        sec_stability2.is_stable = data.get('is_stable')
        if data.get('decomposes_to') is not None:
            for system in data.get('decomposes_to'):
                sec_decomposition = sec_stability.m_create(Decomposition)
                sec_decomposition2 = sec_stability2.m_create(workflow2.Decomposition)
                sec_decomposition.formula = system.get('formula')
                sec_decomposition2.formula = system.get('formula')
                sec_decomposition.fraction = system.get('amount')
                sec_decomposition2.fraction = system.get('amount')
        workflow.results.stability = sec_stability2
        self.archive.workflow2 = workflow

    def parse_phonon(self, data):
        sec_workflow = self.archive.m_create(Workflow)
        sec_workflow.type = 'phonon'
        sec_phonon = sec_workflow.m_create(Phonon)
        # TODO is vasp always mp calculator?
        sec_phonon.force_calculator = 'vasp'
        workflow = workflow2.Phonon(method=workflow2.PhononMethod(), results=workflow2.PhononResults())
        workflow.method.force_calculator = 'vasp'

        calculations = self.archive.run[-1].calculation
        calc = calculations[-1] if calculations else self.archive.run[-1].m_create(Calculation)

        if data.get('ph_dos') is not None:
            sec_dos = calc.m_create(Dos, Calculation.dos_phonon)
            sec_dos.energies = data['ph_dos']['frequencies'] * ureg.THz * ureg.h
            dos = data['ph_dos']['densities'] * (1 / (ureg.THz * ureg.h))
            sec_dos.total.append(DosValues(value=dos))

        if data.get('ph_bs') is not None:
            sec_phonon.with_non_analytic_correction = data['ph_bs'].get('has_nac')
            workflow.method.with_non_analytic_correction = data['ph_bs'].get('has_nac')
            sec_bs = calc.m_create(BandStructure, Calculation.band_structure_phonon)
            bands = np.transpose(data['ph_bs']['bands'])
            qpoints = data['ph_bs']['qpoints']
            labels = data['ph_bs']['labels_dict']
            hisym_qpts = list(labels.values())
            labels = list(labels.keys())
            endpoints = []
            for i, qpoint in enumerate(qpoints):
                if qpoint in hisym_qpts:
                    endpoints.append(i)
                if len(endpoints) < 2:
                    continue
                sec_segment = sec_bs.m_create(BandEnergies)
                energies = bands[endpoints[0]: endpoints[1] + 1]
                sec_segment.energies = np.reshape(energies, (1, *np.shape(energies))) * ureg.THz * ureg.h
                sec_segment.kpoints = qpoints[endpoints[0]: endpoints[1] + 1]
                sec_segment.endpoints_labels = [labels[hisym_qpts.index(qpoints[i])] for i in endpoints]
                endpoints = []

        calc.system_ref = self.archive.run[-1].system[0]
        self.archive.workflow2 = workflow

        # TODO add eigendisplacements

    def parse_tasks(self, data):
        if len(data['calcs_reversed']) == 0:
            return

        xc_func_mapping = {
            'PAW_PBE': ['GGA_X_PBE', 'GGA_C_PBE']
        }

        sec_method = self.archive.run[-1].m_create(Method)
        sec_xc_functional = XCFunctional()
        for potcar_type in data['calcs_reversed'][0].get('input', {}).get('potcar_type', []):
            for xc_functional in xc_func_mapping.get(potcar_type, []):
                if '_X_' in xc_functional or xc_functional.endswith('_X'):
                    sec_xc_functional.exchange.append(Functional(name=xc_functional))
                elif '_C_' in xc_functional or xc_functional.endswith('_C'):
                    sec_xc_functional.correlation.append(Functional(name=xc_functional))
                elif 'HYB' in xc_functional:
                    sec_xc_functional.hybrid.append(Functional(name=xc_functional))
                else:
                    sec_xc_functional.contributions.append(Functional(name=xc_functional))

        sec_method.dft = DFT(xc_functional=sec_xc_functional)
        sec_method.electronic = Electronic(method="DFT")

        encut = data['calcs_reversed'][0].get('input', {}).get('incar', {}).get('ENCUT')
        prec = data['calcs_reversed'][0].get('input', {}).get('incar', {}).get('PREC')
        if encut is not None and prec is not None:
            sec_basis = sec_method.m_create(BasisSet)
            sec_basis.type = 'plane waves'
            sec_basis_set_cell_dependent = sec_basis.m_create(BasisSetCellDependent)
            sec_basis_set_cell_dependent.kind = 'plane waves'
            prec = 1.3 if 'acc' in prec else 1.0
            sec_basis_set_cell_dependent.planewave_cutoff = encut * prec * ureg.eV

        self.archive.run[-1].calculation[0].method_ref = sec_method

    def parse(self, filepath, archive, logger):
        self.filepath = os.path.abspath(filepath)
        self.archive = archive
        self.logger = logger if logger is not None else logging.getLogger(__name__)
        self.maindir = os.path.dirname(self.filepath)

        self.init_parser()

        sec_run = archive.m_create(Run)
        sec_run.program = Program(name='MaterialsProject', version='1.0.0')

        #  TODO system should be referenced
        structure = self.data.get('structure')
        if structure is not None:
            labels = [site['label'] for site in structure.get('sites')]
            positions = [site['xyz'] for site in structure.get('sites')]
            cell = structure.get('lattice', {}).get('matrix')
            sec_system = sec_run.m_create(System)
            sec_atoms = sec_system.m_create(Atoms)
            if cell is not None:
                sec_atoms.lattice_vectors = cell * ureg.angstrom
                sec_atoms.periodic = [True, True, True]
            if positions:
                sec_atoms.positions = positions * ureg.angstrom
            if labels:
                sec_atoms.labels = labels

        for key, val in self.data.get('composition', {}).items():
            sec_system.x_mp_composition.append(
                Composition(x_mp_label=key, x_mp_value=val))

        for key, val in self.data.get('composition_reduced', {}).items():
            sec_system.x_mp_composition_reduced.append(
                Composition(x_mp_label=key, x_mp_value=val))

        symmetry = self.data.get('symmetry')
        if symmetry is not None:
            sec_symmetry = sec_system.m_create(Symmetry)
            for key, val in symmetry.items():
                try:
                    setattr(sec_symmetry, 'x_mp_%s' % key, val)
                except Exception:
                    pass

        # misc
        sec_system.x_mp_elements = [e.get('element') if isinstance(
            e, dict) else e for e in self.data.get('elements', [])]
        for key, val in self.data.items():
            try:
                setattr(sec_system, 'x_mp_%s' % key, val)
            except Exception:
                pass

        # temporary fix to go through workflow normalization
        sec_calc = sec_run.m_create(Calculation)
        sec_calc.system_ref = sec_system

        # TODO should we use the MP api for workflow results?
        workflow_files = [f for f in os.listdir(
            self.maindir) if f.endswith('.json') and f != os.path.basename(self.filepath)]
        for filename in workflow_files:
            try:
                data = json.load(open(os.path.join(self.maindir, filename)))
            except Exception:
                continue
            # make sure data matches that of system
            # TODO maybe better to simply compare filename prefix so no need to load data
            if data.get('material_id', data.get('task_id')) != self.data.get('material_id'):
                continue

            if 'elasticity' in data:
                self.parse_elastic(data)
            if 'eos' in data:
                self.parse_eos(data)
            if 'ph_bs' in data or 'ph_dos' in data:
                self.parse_phonon(data)
            if 'property_name' in data and data.get('property_name') == 'thermo':
                self.parse_thermo(data)
            if 'calcs_reversed' in data:
                self.parse_tasks(data)
