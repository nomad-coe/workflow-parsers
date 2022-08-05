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
import numpy as np            # pylint: disable=unused-import
import typing                 # pylint: disable=unused-import
from nomad.metainfo import (  # pylint: disable=unused-import
    MSection, MCategory, Category, Package, Quantity, Section, SubSection, SectionProxy,
    Reference
)
from nomad.datamodel.metainfo import simulation


m_package = Package()


class x_elastic_section_strain_diagrams(MSection):
    '''
    section collecting the data of the strain diagrams
    '''

    m_def = Section(validate=False)

    x_elastic_strain_diagram_values = Quantity(
        type=np.dtype(np.float64),
        shape=['x_elastic_number_of_deformations', 'x_elastic_strain_diagram_number_of_eta'],
        description='''
        Values of the energy(units:J)/d2E(units:Pa)/cross-validation (depending on the
        value of x_elastic_strain_diagram_type)
        ''')

    x_elastic_strain_diagram_eta_values = Quantity(
        type=np.dtype(np.float64),
        shape=['x_elastic_number_of_deformations', 'x_elastic_strain_diagram_number_of_eta'],
        description='''
        eta values used the strain diagrams
        ''')

    x_elastic_strain_diagram_number_of_eta = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        Number of strain values used in the strain diagram
        ''')

    x_elastic_strain_diagram_type = Quantity(
        type=str,
        shape=[],
        description='''
        Kind of strain diagram. Possible values are: energy; cross-validation (cross-
        validation error); d2E (second derivative of the energy wrt the strain)
        ''')

    x_elastic_strain_diagram_polynomial_fit_order = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        Order of the polynomial fit
        ''')


class x_elastic_section_fitting_parameters(MSection):
    '''
    section collecting the fitting parameters used to calculate the elastic constants
    '''

    m_def = Section(validate=False)

    x_elastic_fitting_parameters_eta = Quantity(
        type=np.dtype(np.float64),
        shape=['x_elastic_number_of_deformations'],
        description='''
        eta values used to calculate the elastic constants
        ''')

    x_elastic_fitting_parameters_polynomial_order = Quantity(
        type=np.dtype(np.int32),
        shape=['x_elastic_number_of_deformations'],
        description='''
        polynomial order used to fit the Energy vs. volume curve and to calculate the
        elastic constants
        ''')


class Method(simulation.method.Method):

    m_def = Section(validate=False, extends_base_section=True)

#     x_elastic_elastic_constant_order = Quantity(
#         type=np.dtype(np.int32),
#         shape=[],
#         description='''
#         Order of the elastic constant
#         ''',
#         a_legacy=LegacyDefinition(name='x_elastic_elastic_constant_order'))

#     x_elastic_number_of_deformations = Quantity(
#         type=np.dtype(np.int32),
#         shape=[],
#         description='''
#         number of deformed structures equally spaced in strain, which are generated
#         between the maximum negative strain and the maximum positive one
#         ''',
#         a_legacy=LegacyDefinition(name='x_elastic_number_of_deformations'))

#     x_elastic_deformation_types = Quantity(
#         type=np.dtype(np.int32),
#         shape=['x_elastic_number_of_deformations', 6],
#         description='''
#         deformation types
#         ''',
#         a_legacy=LegacyDefinition(name='x_elastic_deformation_types'))

#     x_elastic_calculation_method = Quantity(
#         type=str,
#         shape=[],
#         description='''
#         Method of calculation
#         ''',
#         a_legacy=LegacyDefinition(name='x_elastic_calculation_method'))

#     x_elastic_code = Quantity(
#         type=str,
#         shape=[],
#         description='''
#         Code used for the calculation of the elastic constants
#         ''',
#         a_legacy=LegacyDefinition(name='x_elastic_code'))

#     x_elastic_max_lagrangian_strain = Quantity(
#         type=np.dtype(np.float64),
#         shape=[],
#         description='''
#         Maximum lagrangian strain used to calculate the elastic constants
#         ''',
#         a_legacy=LegacyDefinition(name='x_elastic_max_lagrangian_strain'))

#     x_elastic_number_of_distorted_structures = Quantity(
#         type=np.dtype(np.int32),
#         shape=[],
#         description='''
#         Number of distorted structures used to calculate the elastic constants
#         ''',
#         a_legacy=LegacyDefinition(name='x_elastic_number_of_distorted_structures'))

    x_elastic_section_fitting_parameters = SubSection(
        sub_section=SectionProxy('x_elastic_section_fitting_parameters'),
        repeats=True)


# class section_single_configuration_calculation(public.section_single_configuration_calculation):

#     m_def = Section(validate=False, extends_base_section=True)

#     x_elastic_2nd_order_constants_notation_matrix = Quantity(
#         type=np.dtype(np.int32),
#         shape=[6, 6],
#         description='''
#         Symmetry of the second-order elastic constant matrix in Voigt notation
#         ''',
#         a_legacy=LegacyDefinition(name='x_elastic_2nd_order_constants_notation_matrix'))

#     x_elastic_2nd_order_constants_matrix = Quantity(
#         type=np.dtype(np.float64),
#         shape=[6, 6],
#         unit='pascal',
#         description='''
#         2nd order elastic constant (stiffness) matrix in GPa
#         ''',
#         a_legacy=LegacyDefinition(name='x_elastic_2nd_order_constants_matrix'))

#     x_elastic_3rd_order_constants_matrix = Quantity(
#         type=np.dtype(np.float64),
#         shape=[6, 6, 6],
#         unit='pascal',
#         description='''
#         3rd order elastic constant (stiffness) matrix in GPa
#         ''',
#         a_legacy=LegacyDefinition(name='x_elastic_3rd_order_constants_matrix'))

#     x_elastic_2nd_order_constants_compliance_matrix = Quantity(
#         type=np.dtype(np.float64),
#         shape=[6, 6],
#         unit='1 / pascal',
#         description='''
#         Elastic compliance matrix in 1/GPa
#         ''',
#         a_legacy=LegacyDefinition(name='x_elastic_2nd_order_constants_compliance_matrix'))

#     x_elastic_Voigt_bulk_modulus = Quantity(
#         type=np.dtype(np.float64),
#         shape=[],
#         unit='pascal',
#         description='''
#         Voigt bulk modulus
#         ''',
#         a_legacy=LegacyDefinition(name='x_elastic_Voigt_bulk_modulus'))

#     x_elastic_Voigt_shear_modulus = Quantity(
#         type=np.dtype(np.float64),
#         shape=[],
#         unit='pascal',
#         description='''
#         Voigt shear modulus
#         ''',
#         a_legacy=LegacyDefinition(name='x_elastic_Voigt_shear_modulus'))

#     x_elastic_Reuss_bulk_modulus = Quantity(
#         type=np.dtype(np.float64),
#         shape=[],
#         unit='pascal',
#         description='''
#         Reuss bulk modulus
#         ''',
#         a_legacy=LegacyDefinition(name='x_elastic_Reuss_bulk_modulus'))

#     x_elastic_Reuss_shear_modulus = Quantity(
#         type=np.dtype(np.float64),
#         shape=[],
#         unit='pascal',
#         description='''
#         Reuss shear modulus
#         ''',
#         a_legacy=LegacyDefinition(name='x_elastic_Reuss_shear_modulus'))

#     x_elastic_Hill_bulk_modulus = Quantity(
#         type=np.dtype(np.float64),
#         shape=[],
#         unit='pascal',
#         description='''
#         Hill bulk modulus
#         ''',
#         a_legacy=LegacyDefinition(name='x_elastic_Hill_bulk_modulus'))

#     x_elastic_Hill_shear_modulus = Quantity(
#         type=np.dtype(np.float64),
#         shape=[],
#         unit='pascal',
#         description='''
#         Hill shear modulus
#         ''',
#         a_legacy=LegacyDefinition(name='x_elastic_Hill_shear_modulus'))

#     x_elastic_Voigt_Young_modulus = Quantity(
#         type=np.dtype(np.float64),
#         shape=[],
#         unit='pascal',
#         description='''
#         Voigt Young modulus
#         ''',
#         a_legacy=LegacyDefinition(name='x_elastic_Voigt_Young_modulus'))

#     x_elastic_Voigt_Poisson_ratio = Quantity(
#         type=np.dtype(np.float64),
#         shape=[],
#         description='''
#         Voigt Poisson ratio
#         ''',
#         a_legacy=LegacyDefinition(name='x_elastic_Voigt_Poisson_ratio'))

#     x_elastic_Reuss_Young_modulus = Quantity(
#         type=np.dtype(np.float64),
#         shape=[],
#         unit='pascal',
#         description='''
#         Reuss Young modulus
#         ''',
#         a_legacy=LegacyDefinition(name='x_elastic_Reuss_Young_modulus'))

#     x_elastic_Reuss_Poisson_ratio = Quantity(
#         type=np.dtype(np.float64),
#         shape=[],
#         description='''
#         Reuss Poisson ratio
#         ''',
#         a_legacy=LegacyDefinition(name='x_elastic_Reuss_Poisson_ratio'))

#     x_elastic_Hill_Young_modulus = Quantity(
#         type=np.dtype(np.float64),
#         shape=[],
#         unit='pascal',
#         description='''
#         Hill Young modulus
#         ''',
#         a_legacy=LegacyDefinition(name='x_elastic_Hill_Young_modulus'))

#     x_elastic_Hill_Poisson_ratio = Quantity(
#         type=np.dtype(np.float64),
#         shape=[],
#         description='''
#         Hill Poisson ratio
#         ''',
#         a_legacy=LegacyDefinition(name='x_elastic_Hill_Poisson_ratio'))

#     x_elastic_eigenvalues = Quantity(
#         type=np.dtype(np.float64),
#         shape=[6],
#         unit='pascal',
#         description='''
#         Eigemvalues of the stiffness matrix
#         ''',
#         a_legacy=LegacyDefinition(name='x_elastic_eigenvalues'))

#     x_elastic_section_strain_diagrams = SubSection(
#         sub_section=SectionProxy('x_elastic_section_strain_diagrams'),
#         repeats=True,
#         a_legacy=LegacyDefinition(name='x_elastic_section_strain_diagrams'))


class System(simulation.system.System):

    m_def = Section(validate=False, extends_base_section=True)

    x_elastic_space_group_number = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        Space-group number of the system
        ''')

    x_elastic_unit_cell_volume = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        unit='m ** 3',
        description='''
        Volume of the equilibrium unit cell
        ''')
