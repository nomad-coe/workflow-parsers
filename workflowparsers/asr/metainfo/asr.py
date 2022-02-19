import numpy as np            # pylint: disable=unused-import
import typing                 # pylint: disable=unused-import
from nomad.metainfo import (  # pylint: disable=unused-import
    MSection, MCategory, Category, Package, Quantity, Section, SubSection, SectionProxy,
    Reference, MEnum, JSON)
from nomad.datamodel.metainfo import simulation


m_package = Package(
    name='None',
    description='None')


class x_asr_parameters(MSection):

    m_def = Section(
        validate=False)

    x_asr_tmp_atoms = Quantity(
        type=str,
        shape=[],
        description='''
        ''')

    x_asr_tmp_atoms_file = Quantity(
        type=str,
        shape=[],
        description='''
        ''')

    x_asr_fmax = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        ''')

    x_asr_calculator = Quantity(
        type=JSON,
        shape=[],
        description='''
        ''')

    x_asr_magstatecalculator = Quantity(
        type=JSON,
        shape=[],
        description='''
        ''')

    x_asr_rc = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        ''')

    x_asr_d = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        ''')

    x_asr_fsname = Quantity(
        type=str,
        shape=[],
        description='''
        ''')

    x_asr_sc = Quantity(
        type=np.dtype(np.float64),
        shape=[3],
        description='''
        ''')

    x_asr_dist_max = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        ''')


class x_asr_code(MSection):

    m_def = Section(
        validate=False)

    x_asr_package = Quantity(
        type=str,
        shape=[],
        description='''
        ''')

    x_asr_version = Quantity(
        type=str,
        shape=[],
        description='''
        ''')

    x_asr_git_hash = Quantity(
        type=str,
        shape=[],
        description='''
        ''')


class x_asr_codes(MSection):

    m_def = Section(
        validate=False)

    x_asr_code = SubSection(
        sub_section=SectionProxy('x_asr_code'),
        repeats=True)


class x_asr_run_specification(MSection):

    m_def = Section(
        validate=False)

    x_asr_name = Quantity(
        type=str,
        shape=[],
        description='''
        ''')

    x_asr_version = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        ''')

    x_asr_uid = Quantity(
        type=str,
        shape=[],
        description='''
        ''')

    x_asr_parameters = SubSection(
        sub_section=SectionProxy('x_asr_parameters'),
        repeats=False)

    x_asr_codes = SubSection(
        sub_section=SectionProxy('x_asr_codes'),
        repeats=False)


class x_asr_resources(MSection):

    m_def = Section(
        validate=False)

    x_asr_execution_start = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        ''')

    x_asr_execution_end = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        ''')

    x_asr_execution_duration = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        ''')

    x_asr_ncores = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        ''')


class x_asr_dependency(MSection):

    m_def = Section(
        validate=False)

    x_asr_uid = Quantity(
        type=str,
        shape=[],
        description='''
        ''')

    x_asr_revision = Quantity(
        type=str,
        shape=[],
        description='''
        ''')


class x_asr_dependencies(MSection):

    m_def = Section(
        validate=False)

    x_asr_dependency = SubSection(
        sub_section=SectionProxy('x_asr_dependency'),
        repeats=True)


class x_asr_metadata(MSection):

    m_def = Section(
        validate=False)

    x_asr_directory = Quantity(
        type=str,
        shape=[],
        description='''
        ''')

    x_asr_created = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        ''')

    x_asr_modified = Quantity(
        type=np.dtype(np.float64),
        shape=[],
        description='''
        ''')


class Run(simulation.run.Run):

    m_def = Section(validate=False, extends_base_section=True)

    x_asr_history = Quantity(
        type=str,
        shape=[],
        description='''
        ''')

    x_asr_name = Quantity(
        type=str,
        shape=[],
        description='''
        ''')

    x_asr_tags = Quantity(
        type=str,
        shape=[],
        description='''
        ''')

    x_asr_revision = Quantity(
        type=str,
        shape=[],
        description='''
        ''')

    x_asr_uid = Quantity(
        type=str,
        shape=[],
        description='''
        ''')

    x_asr_version = Quantity(
        type=np.dtype(np.int32),
        shape=[],
        description='''
        ''')

    x_asr_resources = SubSection(
        sub_section=SectionProxy('x_asr_resources'),
        repeats=False)

    x_asr_dependencies = SubSection(
        sub_section=SectionProxy('x_asr_dependencies'),
        repeats=False)

    x_asr_metadata = SubSection(
        sub_section=SectionProxy('x_asr_metadata'),
        repeats=False)

    x_asr_run_specification = SubSection(
        sub_section=SectionProxy('x_asr_run_specification'),
        repeats=False)


m_package.__init_metainfo__()
