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
from ase import Atoms as aseAtoms

from nomad.units import ureg
from nomad.parsing.file_parser import Quantity, TextParser

from nomad.datamodel.metainfo.simulation.run import Run, Program
from nomad.datamodel.metainfo.simulation.method import Method
from nomad.datamodel.metainfo.simulation.system import System, Atoms
from nomad.datamodel.metainfo.simulation.calculation import Calculation
from nomad.datamodel.metainfo.workflow import Workflow, Elastic, StrainDiagrams
from nomad.datamodel.metainfo.simulation import workflow as workflow2

from .metainfo.elastic import x_elastic_section_fitting_parameters


class InfoParser(TextParser):
    def __init__(self):
        super().__init__(None)

    def init_quantities(self):
        self._quantities = [
            Quantity(
                'order', r'\s*Order of elastic constants\s*=\s*([0-9]+)', repeats=False,
                dtype=int),
            Quantity(
                'calculation_method', r'\s*Method of calculation\s*=\s*([-a-zA-Z]+)\s*',
                repeats=False),
            Quantity(
                'code_name', r'\s*DFT code name\s*=\s*([-a-zA-Z]+)', repeats=False),
            Quantity(
                'space_group_number', r'\s*Space-group number\s*=\s*([0-9]+)', repeats=False),
            Quantity(
                'equilibrium_volume', r'\s*Volume of equilibrium unit cell\s*=\s*([0-9.]+)\s*',
                unit='angstrom ** 3'),
            Quantity(
                'max_strain', r'\s*Maximum Lagrangian strain\s*=\s*([0-9.]+)', repeats=False),
            Quantity(
                'n_strains', r'\s*Number of distorted structures\s*=\s*([0-9]+)', repeats=False)]


class StructureParser(TextParser):
    def __init__(self):
        super().__init__(None)

    def init_quantities(self):
        def get_sym_pos(val):
            val = val.strip().replace('\n', '').split()
            sym = []
            pos = []
            for i in range(0, len(val), 4):
                sym.append(val[i + 3].strip())
                pos.append([float(val[j]) for j in range(i, i + 3)])
            sym_pos = dict(symbols=sym, positions=pos)
            return sym_pos

        self._quantities = [
            Quantity(
                'cellpar', r'a\s*b\s*c\n([\d\.\s]+)\n\s*alpha\s*beta\s*gamma\n([\d\.\s]+)\n+',
                repeats=False),
            Quantity(
                'sym_pos', r'Atom positions:\n\n([\s\d\.A-Za-z]+)\n\n',
                str_operation=get_sym_pos, repeats=False, convert=False)]


class DistortedParametersParser(TextParser):
    def __init__(self):
        super().__init__(None)

    def init_quantities(self):
        self._quantities = [Quantity(
            'deformation', r'Lagrangian strain\s*=\s*\(([eta\s\d\.,]+)\)',
            str_operation=lambda x: x.replace(',', '').split(), repeats=True, dtype=str)]


class FitParser(TextParser):
    def __init__(self):
        super().__init__(None)

    def init_quantities(self):

        def split_eta_val(val):
            order, val = val.strip().split(' order fit.')
            val = [float(v) for v in val.strip().split()]
            return order, val[0::2], val[1::2]

        self._quantities = [Quantity(
            'fit', r'(\w+ order fit\.\n[\d.\s\neE\-\+]+)\n', repeats=True, convert=False,
            str_operation=split_eta_val)]


class ElasticConstant2Parser(TextParser):
    def __init__(self):
        super().__init__(None)

    def init_quantities(self):
        self._quantities = [
            Quantity(
                'voigt', r'Symmetry[\s\S]+\n\s*\n([C\d\s\n\(\)\-\+\/\*]+)\n',
                shape=(6, 6), dtype=str, repeats=False),
            Quantity(
                'elastic_constant', r'Elastic constant[\s\S]+in GPa\s*:\s*\n\n([\-\d\.\s\n]+)\n',
                shape=(6, 6), dtype=float, unit='GPa', repeats=False),
            Quantity(
                'compliance', r'Elastic compliance[\s\S]+in 1/GPa\s*:\s*\n\n([\-\d\.\s\n]+)\n',
                shape=(6, 6), dtype=float, unit='1/GPa', repeats=False)]

        def str_to_modulus(val_in):
            val_in = val_in.strip().split()
            key = val_in[0]
            unit = val_in[-1] if len(val_in) == 3 else None
            val = float(val_in[1])
            val = val * ureg.GPa if unit is not None else val
            return key, val

        self._quantities.append(Quantity(
            'modulus', r',\s*(\w+)\s*=\s*([\-\+\w\. ]+?)\n', str_operation=str_to_modulus,
            repeats=True))

        self._quantities.append(Quantity(
            'eigenvalues',
            r'Eigenvalues of elastic constant \(stiffness\) matrix:\s*\n+([\-\d\.\n\s]+)\n',
            unit='GPa', repeats=False))


class ElasticConstant3Parser(TextParser):
    def __init__(self):
        super().__init__(None)

    def init_quantities(self):
        def arrange_matrix(val):
            val = val.strip().split('\n')
            matrix = [v.strip().split() for v in val if v.strip()]
            matrix = np.array(matrix).reshape((12, 18))
            arranged = []
            for i in range(2):
                for j in range(3):
                    arranged.append(
                        matrix[i * 6: (i + 1) * 6, j * 6: (j + 1) * 6].tolist())
            return arranged

        self._quantities = [
            Quantity(
                'elastic_constant', r'\%\s*\n([\s0-6A-L]*)[\n\s\%1-6\-ij]*([\s0-6A-L]*)\n',
                str_operation=arrange_matrix, dtype=str, repeats=False, convert=False),
            Quantity(
                'cijk', r'(C\d\d\d)\s*=\s*([\-\d\.]+)\s*GPa', repeats=True, convert=False)]


class ElasticParser:
    def __init__(self):
        self._mainfile = None
        self.logger = None
        self._deform_dirs = None
        self._deform_dir_prefix = 'Dst'
        self._dirs = []
        self.info = InfoParser()
        self.structure = StructureParser()
        self.distorted_parameters = DistortedParametersParser()
        self.fit = FitParser()
        self.elastic_constant_2 = ElasticConstant2Parser()
        self.elastic_constant_3 = ElasticConstant3Parser()

    @property
    def deformation_dirs(self):
        if self._deform_dirs is None:
            self._deform_dirs = [
                os.path.join(self.maindir, d) for d in self._dirs if d.startswith(self._deform_dir_prefix)]

        return self._deform_dirs

    def get_elastic_files(self, filename, extension, dirname=None):
        dirs = self._dirs if dirname is None else os.listdir(dirname)
        dirname = self.maindir if dirname is None else dirname
        filenames = [d for d in dirs if filename in d and d.endswith(extension)]
        if len(filenames) > 1:
            filenames = [d for d in filenames if d.startswith(filename)]
        if len(filenames) == 0:
            return
        return os.path.join(dirname, filenames[0])

    def get_references_to_calculations(self):
        def output_file(dirname):
            code = self.info.get('code_name', '').lower()
            if code == 'exciting':
                return os.path.join(dirname, 'INFO.OUT')
            elif code == 'wien':
                return os.path.join(dirname, '%s_Converged.scf' % os.path.basename(dirname))
            elif code == 'quantum':
                return os.path.join(dirname, '%s.out' % os.path.basename(dirname))
            else:
                return None

        references = []
        for deform_dir in self.deformation_dirs:
            sub_dirs = os.listdir(deform_dir)
            for sub_dir in sub_dirs:
                calc_dir = os.path.join(deform_dir, sub_dir)
                out_file = output_file(calc_dir)
                if out_file is not None and os.path.isfile(out_file):
                    references.append(out_file)

        return references

    def get_structure_info(self):
        path = os.path.join(self.maindir, 'sgroup.out')
        if not os.path.isfile(path):
            return

        self.structure.mainfile = path

        cellpar = self.structure.get('cellpar', None)
        sym_pos = self.structure.get('sym_pos', {})

        sym = sym_pos.get('symbols', None)
        pos = sym_pos.get('positions', None)

        if cellpar is None or sym is None or pos is None:
            return

        structure = aseAtoms(cell=cellpar, scaled_positions=pos, symbols=sym, pbc=True)

        positions = structure.get_positions()
        positions = positions * ureg.angstrom
        cell = structure.get_cell()
        cell = cell * ureg.angstrom

        return sym, positions, cell

    def get_strain_energy(self):
        strains, energies = [], []

        for deform_dir in self.deformation_dirs:
            filenames = [d for d in os.listdir(deform_dir) if d.endswith('Energy.dat')]
            if not filenames:
                continue

            path = os.path.join(deform_dir, filenames[-1])
            data = np.loadtxt(path).T
            strains.append(list(data[0]))
            # the peculiarity of the x_elastic_strain_diagram_values metainfo that it does
            # not have the energy unit
            energies.append((data[1] * ureg.hartree).to('J').magnitude)
        if len(np.shape(energies)) != 2:
            strains, energies = [], []
        return strains, energies

    def get_strain_stress(self):
        strains = {'Lagrangian-stress': [], 'Physical-stress': []}
        stresses = {'Lagrangian-stress': [], 'Physical-stress': []}

        for deform_dir in self.deformation_dirs:
            filenames = [d for d in os.listdir(deform_dir) if d.endswith('stress.dat')]

            for filename in filenames:
                path = os.path.join(deform_dir, filename)
                if not os.path.isfile(path):
                    continue

                with open(path) as f:
                    lines = f.readlines()

                strain, stress = [], []
                for line in lines:
                    val = line.strip().split()
                    if not val[0].strip().replace('.', '').isdecimal():
                        continue

                    strain.append(float(val[0]))
                    stress.append([float(v) for v in val[1:7]])

                stype = filename.rstrip('.dat').split('_')[-1]
                strains[stype].append(strain)
                stresses[stype].append(stress)

        return strains, stresses

    def get_deformation_types(self):
        path = os.path.join(self.maindir, 'Distorted_Parameters')
        self.distorted_parameters.mainfile = path
        return self.distorted_parameters.get('deformation')

    def _get_fit(self, path_dir, file_ext):
        path_dir = os.path.join(self.maindir, path_dir)

        if not os.path.isdir(path_dir):
            return

        paths = [p for p in os.listdir(path_dir) if p.endswith(file_ext)]
        paths.sort()

        if not paths:
            return

        eta, val = {}, {}
        for path in paths:
            self.fit.mainfile = os.path.join(path_dir, path)
            fit_results = self.fit.get('fit', [])
            for result in fit_results:
                eta.setdefault(result[0], [])
                val.setdefault(result[0], [])
                eta[result[0]].append(result[1])
                val[result[0]].append(result[2])

        return eta, val

    def get_energy_fit(self):
        energy_fit = dict()

        for file_ext in ['d2E.dat', 'd3E.dat', 'ddE.dat']:
            result = self._get_fit('Energy-vs-Strain', file_ext)
            if result is None:
                continue

            result = list(result)
            result[1] = {
                key: (val * ureg.GPa).to('Pa').magnitude for key, val in result[1].items()}
            energy_fit['d2e'] = result

        result = self._get_fit('Energy-vs-Strain', 'CVe.dat')
        if result is not None:
            result = list(result)
            result[1] = {
                key: (val * ureg.hartree).to('J').magnitude for key, val in result[1].items()}
            energy_fit['cross-validation'] = result

        return energy_fit

    def get_stress_fit(self):
        stress_fit = dict()
        stress_fit['dtn'] = [[]] * 6
        stress_fit['cross-validation'] = [[]] * 6

        for strain_index in range(1, 7):
            result = self._get_fit('Stress-vs-Strain', '%d_dS.dat' % strain_index)
            if result is not None:
                result[1] = {key: val * ureg.GPa for key, val in result[1].items()}
                stress_fit['dtn'][strain_index - 1] = result

            result = self._get_fit('Stress-vs-Strain', '%d_CVe.dat' % strain_index)
            if result is not None:
                result[1] = {key: val * ureg.hartree for key, val in result[1].items()}
                stress_fit['cross-validation'][strain_index - 1] = result

        return stress_fit

    def get_input(self):
        paths = os.listdir(self.maindir)
        path = None
        order = self.info.get('order', 2)
        for p in paths:
            if 'ElaStic_' in p and p.endswith('.in') and str(order) in p:
                path = p
                break

        if path is None:
            return

        calc_method = self.info.get('calculation_method')

        eta_ec = []
        fit_ec = []

        def _is_number(var):
            try:
                float(var)
                return True
            except Exception:
                return False

        path = os.path.join(self.maindir, path)

        with open(path) as f:
            while True:
                line = f.readline()
                if not line:
                    break

                if calc_method.lower() == 'energy':
                    _, eta, fit = line.strip().split()
                    eta_ec.append(float(eta))
                    fit_ec.append(int(fit))

                elif calc_method.lower() == 'stress':
                    val = line.strip().split()
                    if not _is_number(val[0]):
                        eta_ec.append([float(val[i + 1]) for i in range(6)])
                    else:
                        fit_ec.append([int(val[i]) for i in range(6)])

                else:
                    pass

        return eta_ec, fit_ec

    def get_elastic_constants_order2(self):
        path = self.get_elastic_files('ElaStic_2nd', 'out')
        self.elastic_constant_2.mainfile = path

        matrices = dict()
        for key in ['voigt', 'elastic_constant', 'compliance']:
            val = self.elastic_constant_2.get(key, None)
            if val is not None:
                matrices[key] = val

        moduli = dict()
        for modulus in self.elastic_constant_2.get('modulus', []):
            moduli[modulus[0]] = modulus[1]

        eigenvalues = self.elastic_constant_2.get('eigenvalues')

        return matrices, moduli, eigenvalues

    def get_elastic_constants_order3(self):
        path = self.get_elastic_files('ElaStic_3rd.out', 'out')
        self.elastic_constant_3.mainfile = path

        elastic_constant_str = self.elastic_constant_3.get('elastic_constant')
        if elastic_constant_str is None:
            return

        cijk = dict()
        for element in self.elastic_constant_3.get('cijk', []):
            cijk[element[0]] = float(element[1])

        # formulas for the coefficients
        coeff_A = cijk.get('C111', 0) + cijk.get('C112', 0) - cijk.get('C222', 0)
        coeff_B = -(cijk.get('C115', 0) + 3 * cijk.get('C125', 0)) / 2
        coeff_C = (cijk.get('C114', 0) + 3 * cijk.get('C124', 0)) / 2
        coeff_D = -(2 * cijk.get('C111', 0) + cijk.get('C112', 0) - 3 * cijk.get('C222', 0)) / 4
        coeff_E = -cijk.get('C114', 0) - 2 * cijk.get('C124', 0)
        coeff_F = -cijk.get('C115', 0) - 2 * cijk.get('C125', 0)
        coeff_G = -(cijk.get('C115', 0) - cijk.get('C125', 0)) / 2
        coeff_H = (cijk.get('C114', 0) - cijk.get('C124', 0)) / 2
        coeff_I = (2 * cijk.get('C111', 0) - cijk.get('C112', 0) - cijk.get('C222', 0)) / 4
        coeff_J = (cijk.get('C113', 0) - cijk.get('C123', 0)) / 2
        coeff_K = -(cijk.get('C144', 0) - cijk.get('C155', 0)) / 2

        space_group_number = self.info.get('space_group_number')

        if space_group_number <= 148:  # rhombohedral II
            coefficients = dict(
                A=coeff_A, B=coeff_B, C=coeff_C, D=coeff_D, E=coeff_E, F=coeff_F, G=coeff_G,
                H=coeff_H, I=coeff_I, J=coeff_J, K=coeff_K)
        elif space_group_number <= 167:  # rhombohedral I
            coefficients = dict(
                A=coeff_A, B=coeff_C, C=coeff_D, D=coeff_E, E=coeff_H, F=coeff_I, G=coeff_J,
                H=coeff_K)
        elif space_group_number <= 176:  # hexagonal II
            coefficients = dict(
                A=coeff_A, B=coeff_D, C=coeff_I, D=coeff_J, E=coeff_K)
        elif space_group_number <= 194:  # hexagonal I
            coefficients = dict(
                A=coeff_A, B=coeff_D, C=coeff_I, D=coeff_J, E=coeff_K)
        else:
            coefficients = dict()

        # assign values to the elastic constant matrix from the independent ec and coefficients
        elastic_constant = np.zeros((6, 6, 6))
        for i in range(6):
            for j in range(6):
                for k in range(6):
                    val = elastic_constant_str[i][j][k]
                    if val == '0':
                        elastic_constant[i][j][k] = 0
                    elif val.isdigit():
                        elastic_constant[i][j][k] = cijk['C%s' % val]
                    else:
                        elastic_constant[i][j][k] = coefficients.get(val, 0)

        return elastic_constant

    def parse_strain(self):
        sec_elastic = self.archive.workflow[0].elastic
        method = self.info['calculation_method'].lower()

        n_strains = self.info['n_strains']
        poly_fit_2 = int((n_strains - 1) / 2)
        poly_fit = {
            '2nd': poly_fit_2, '3rd': poly_fit_2 - 1, '4th': poly_fit_2 - 1,
            '5th': poly_fit_2 - 2, '6th': poly_fit_2 - 2, '7th': poly_fit_2 - 3}

        if method == 'energy':
            strain, energy = self.get_strain_energy()
            if not strain:
                self.logger.warn('Error getting strain and energy data')
                return

            sec_strain_diagram = sec_elastic.m_create(StrainDiagrams)
            sec_strain_diagram.type = 'energy'
            sec_strain_diagram.n_eta = len(strain[0])
            sec_strain_diagram.eta = strain
            sec_strain_diagram.value = energy
            sec_strain_diagram2 = self.archive.workflow2.results.m_create(workflow2.StrainDiagrams)
            sec_strain_diagram2.type = 'energy'
            sec_strain_diagram2.n_eta = len(strain[0])
            sec_strain_diagram2.eta = strain
            sec_strain_diagram2.value = energy

            energy_fit = self.get_energy_fit()
            if not energy_fit:
                self.logger.warn('Error getting energy fit data')
                return

            for diagram_type in ['cross-validation', 'd2e']:
                for fit_order in energy_fit[diagram_type][0].keys():
                    sec_strain_diagram = sec_elastic.m_create(StrainDiagrams)
                    sec_strain_diagram.type = diagram_type
                    sec_strain_diagram.polynomial_fit_order = int(fit_order[:-2])
                    sec_strain_diagram.n_eta = poly_fit.get(fit_order, None)
                    sec_strain_diagram.eta = energy_fit[diagram_type][0][fit_order]
                    sec_strain_diagram.value = energy_fit[diagram_type][1][fit_order]
                    sec_strain_diagram2 = self.archive.workflow2.results.m_create(workflow2.StrainDiagrams)
                    sec_strain_diagram2.type = diagram_type
                    sec_strain_diagram2.polynomial_fit_order = int(fit_order[:-2])
                    sec_strain_diagram2.n_eta = poly_fit.get(fit_order, None)
                    sec_strain_diagram2.eta = energy_fit[diagram_type][0][fit_order]
                    sec_strain_diagram2.value = energy_fit[diagram_type][1][fit_order]

        elif method == 'stress':
            strain, stress = self.get_strain_stress()
            for diagram_type in ['Lagrangian-stress', 'Physical-stress']:
                strain_i = strain[diagram_type]
                if not strain_i:
                    continue
                stress_i = np.transpose(np.array(stress[diagram_type]), axes=(2, 0, 1))

                for si in range(6):
                    sec_strain_diagram = sec_elastic.m_create(StrainDiagrams)
                    sec_strain_diagram.type = diagram_type
                    sec_strain_diagram.stress_voigt_component = si + 1
                    sec_strain_diagram.n_eta = len(strain_i[0])
                    sec_strain_diagram.eta = strain_i
                    sec_strain_diagram.value = stress_i[si]
                    sec_strain_diagram2 = self.archive.workflow2.results.m_create(workflow2.StrainDiagrams)
                    sec_strain_diagram2.type = diagram_type
                    sec_strain_diagram2.stress_voigt_component = si + 1
                    sec_strain_diagram2.n_eta = len(strain_i[0])
                    sec_strain_diagram2.eta = strain_i
                    sec_strain_diagram2.value = stress_i[si]

            stress_fit = self.get_stress_fit()
            for diagram_type in ['cross-validation', 'dtn']:
                for si in range(6):
                    if len(stress_fit[diagram_type][si]) == 0:
                        continue
                    for fit_order in stress_fit[diagram_type][si][0].keys():
                        sec_strain_diagram = sec_elastic.m_create(StrainDiagrams)
                        sec_strain_diagram.type = diagram_type
                        sec_strain_diagram.stress_voigt_component = si + 1
                        sec_strain_diagram.polynomial_fit_order = int(fit_order[:-2])
                        sec_strain_diagram.n_eta = poly_fit.get(fit_order, None)
                        sec_strain_diagram.eta = stress_fit[diagram_type][si][0][fit_order]
                        sec_strain_diagram.value = np.array(stress_fit[diagram_type][si][1][fit_order])
                        sec_strain_diagram2 = self.archive.workflow2.results.m_create(workflow2.StrainDiagrams)
                        sec_strain_diagram2.type = diagram_type
                        sec_strain_diagram2.stress_voigt_component = si + 1
                        sec_strain_diagram2.polynomial_fit_order = int(fit_order[:-2])
                        sec_strain_diagram2.n_eta = poly_fit.get(fit_order, None)
                        sec_strain_diagram2.eta = stress_fit[diagram_type][si][0][fit_order]
                        sec_strain_diagram2.value = np.array(stress_fit[diagram_type][si][1][fit_order])

    def parse_elastic_constant(self):
        sec_elastic = self.archive.workflow[0].elastic
        sec_results = self.archive.workflow2.results

        order = self.info['order']

        if order == 2:
            matrices, moduli, eigenvalues = self.get_elastic_constants_order2()
            sec_elastic.elastic_constants_notation_matrix_second_order = matrices.get('voigt')
            sec_elastic.elastic_constants_matrix_second_order = matrices.get('elastic_constant')
            sec_elastic.compliance_matrix_second_order = matrices.get('compliance')

            sec_elastic.bulk_modulus_voigt = moduli.get('B_V', moduli.get('K_V'))
            sec_elastic.shear_modulus_voigt = moduli.get('G_V')

            sec_elastic.bulk_modulus_reuss = moduli.get('B_R', moduli.get('K_R'))
            sec_elastic.shear_modulus_reuss = moduli.get('G_R')

            sec_elastic.bulk_modulus_hill = moduli.get('B_H', moduli.get('K_H'))
            sec_elastic.shear_modulus_hill = moduli.get('G_H')

            sec_elastic.young_modulus_voigt = moduli.get('E_V')
            sec_elastic.poisson_ratio_voigt = moduli.get('nu_V')
            sec_elastic.young_modulus_reuss = moduli.get('E_R')
            sec_elastic.poisson_ratio_reuss = moduli.get('nu_R')
            sec_elastic.young_modulus_hill = moduli.get('E_H')
            sec_elastic.poisson_ratio_hill = moduli.get('nu_H')

            sec_elastic.eigenvalues_elastic = eigenvalues

            sec_results.elastic_constants_notation_matrix_second_order = matrices.get('voigt')
            sec_results.elastic_constants_matrix_second_order = matrices.get('elastic_constant')
            sec_results.compliance_matrix_second_order = matrices.get('compliance')

            sec_results.bulk_modulus_voigt = moduli.get('B_V', moduli.get('K_V'))
            sec_results.shear_modulus_voigt = moduli.get('G_V')

            sec_results.bulk_modulus_reuss = moduli.get('B_R', moduli.get('K_R'))
            sec_results.shear_modulus_reuss = moduli.get('G_R')

            sec_results.bulk_modulus_hill = moduli.get('B_H', moduli.get('K_H'))
            sec_results.shear_modulus_hill = moduli.get('G_H')

            sec_results.young_modulus_voigt = moduli.get('E_V')
            sec_results.poisson_ratio_voigt = moduli.get('nu_V')
            sec_results.young_modulus_reuss = moduli.get('E_R')
            sec_results.poisson_ratio_reuss = moduli.get('nu_R')
            sec_results.young_modulus_hill = moduli.get('E_H')
            sec_results.poisson_ratio_hill = moduli.get('nu_H')

            sec_results.eigenvalues_elastic = eigenvalues

        elif order == 3:
            elastic_constant = self.get_elastic_constants_order3()
            if elastic_constant is not None:
                sec_elastic.elastic_constants_matrix_third_order = elastic_constant * ureg.GPa
                sec_results.elastic_constants_matrix_third_order = elastic_constant * ureg.GPa

    def init_parser(self):
        self._deform_dirs = None
        self.maindir = os.path.dirname(self.filepath)
        self.info.mainfile = self.filepath
        self.info.logger = self.logger
        self.structure.logger = self.logger
        self.distorted_parameters.logger = self.logger
        self.fit.logger = self.logger
        self.elastic_constant_2.logger = self.logger
        self.elastic_constant_3.logger = self.logger
        self._dirs = os.listdir(self.maindir)

    def reuse_parser(self, parser):
        self.info.quantities = parser.info.quantities
        self.structure.quantities = parser.structure.quantities
        self.distorted_parameters.quantities = parser.distorted_parameters.quantities
        self.fit.quantities = parser.fit.quantities
        self.elastic_constant_2.quantities = parser.elastic_constant_2.quantities
        self.elastic_constant_3.quantities = parser.elastic_constant_3.quantities

    def parse(self, filepath, archive, logger):
        self.filepath = os.path.abspath(filepath)
        self.archive = archive
        self.logger = logger if logger is not None else logging

        self.init_parser()

        sec_run = self.archive.m_create(Run)

        sec_run.program = Program(name='elastic', version='1.0')

        sec_system = sec_run.m_create(System)

        symbols_positions_cell = self.get_structure_info()
        volume = self.info['equilibrium_volume']

        sec_atoms = sec_system.m_create(Atoms)
        if symbols_positions_cell is not None:
            sec_atoms.labels = symbols_positions_cell[0]
            sec_atoms.positions = symbols_positions_cell[1]
            sec_atoms.lattice_vectors = symbols_positions_cell[2]
        sec_atoms.periodic = [True, True, True]
        sec_system.x_elastic_space_group_number = self.info['space_group_number']
        sec_system.x_elastic_unit_cell_volume = volume

        sec_method = sec_run.m_create(Method)

        sec_scc = sec_run.m_create(Calculation)
        sec_scc.calculations_path = self.get_references_to_calculations()

        fit_input = self.get_input()
        if fit_input is not None:
            sec_fit_par = sec_method.m_create(x_elastic_section_fitting_parameters)
            sec_fit_par.x_elastic_fitting_parameters_eta = fit_input[0]
            sec_fit_par.x_elastic_fitting_parameters_polynomial_order = fit_input[1]

        sec_scc.method_ref = sec_method
        sec_scc.system_ref = sec_system

        sec_workflow = self.archive.m_create(Workflow)
        sec_workflow.workflow_type = 'elastic'
        sec_elastic = sec_workflow.m_create(Elastic)
        sec_elastic.energy_stress_calculator = self.info['code_name']
        sec_elastic.calculation_method = self.info['calculation_method'].lower()
        sec_elastic.elastic_constants_order = self.info['order']
        sec_elastic.strain_maximum = self.info['max_strain']
        sec_elastic.n_strains = self.info['n_strains']

        deformation_types = self.get_deformation_types()
        sec_elastic.n_deformations = len(self.deformation_dirs)
        sec_elastic.deformation_types = deformation_types

        workflow = workflow2.Elastic(
            method=workflow2.ElasticMethod(), results=workflow2.ElasticResults())
        workflow.method.energy_stress_calculator = self.info['code_name']
        workflow.method.calculation_method = self.info['calculation_method'].lower()
        workflow.method.elastic_constants_order = self.info['order']
        workflow.method.strain_maximum = self.info['max_strain']
        workflow.results.n_strains = self.info['n_strains']
        workflow.results.n_deformations = len(self.deformation_dirs)
        workflow.results.deformation_types = deformation_types
        self.archive.workflow2 = workflow

        self.parse_strain()
        self.parse_elastic_constant()
