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
from pydantic import Field
from typing import Optional

from nomad.config.models.plugins import ParserEntryPoint


class EntryPoint(ParserEntryPoint):
    parser_class_name: str = Field(
        description="""
        The fully qualified name of the Python class that implements the parser.
        This class must have a function `def parse(self, mainfile, archive, logger)`.
    """
    )
    code_name: Optional[str] = None
    code_homepage: Optional[str] = None
    code_category: Optional[str] = None
    metadata: Optional[dict] = Field(
        None,
        description="""
        Metadata passed to the UI. Deprecated. """
    )

    def load(self):
        from nomad.parsing import MatchingParserInterface

        return MatchingParserInterface(**self.dict())


aflow_parser_entry_point = EntryPoint(
    name='parsers/aflow',
    aliases=['parsers/aflow'],
    description='NOMAD parser for AFLOW.',
    python_package='workflowparsers.aflow',
    mainfile_contents_re=(
        r'^\s*\[AFLOW\] \*+\s*\[AFLOW\]\s*\[AFLOW\]                     .o.        .o88o.'
        r"oooo\s*\[AFLOW\]                    .888.       888 `` `888\s*\[AFLOW\]                   .8'888.     o888oo   888   .ooooo.  oooo"
        r"oooo    ooo\s*\[AFLOW\]                  .8' `888.     888     888  d88' `88b  `88."
        r"`88.  .8'\s*\[AFLOW\]                 .88ooo8888.    888     888  888   888   `88..]88..8'\s*\[AFLOW\]                .8'     `888.   888     888  888   888    `888'`888'\s*\[AFLOW\]               o88o     o8888o"
        r"o888o   o888o `Y8bod8P'     `8'  `8'  .in|^\s*\{\"aurl\"\:\"aflowlib\.duke\.edu\:AFLOWDATA"
    ),
    mainfile_mime_re='(application/json)|(text/.*)',
    mainfile_name_re=r'.*aflowlib\.json.*',
    mainfile_alternative=True,
    supported_compressions=['gz', 'bz2', 'xz'],
    parser_class_name='workflowparsers.aflow.AFLOWParser',
    code_name='AFLOW',
    code_homepage='http://www.aflowlib.org/',
    code_category='Workflow manager',
    metadata={
        'codeCategory': 'Workflow manager',
        'codeLabel': 'AFLOW',
        'codeLabelStyle': 'all capitals',
        'codeName': 'aflow',
        'codeUrl': 'http://www.aflowlib.org/',
        'parserDirName': 'dependencies/workflow/workflowparsers/aflow/',
        'parserGitUrl': 'https://github.com/nomad-coe/workflow-parsers.git',
        'parserSpecific': '',
        'preamble': '',
        'status': 'production',
        'tableOfFiles': '|Input Filename| Description|\n|--- | --- |\n|`aflowlib.json` | **Mainfile:** a json file containing the aflow output|\n|`aflow.ael.out`| plain text, elastic outputs|\n|`aflow.agl.out` | plain text, Debye model output|\n',
    },
)

asr_parser_entry_point = EntryPoint(
    name='parsers/asr',
    aliases=['parsers/asr'],
    description='NOMAD parser for ASR.',
    python_package='workflowparsers.asr',
    mainfile_contents_re='"name": "ASR"',
    mainfile_mime_re='(application/json)|(text/.*)',
    mainfile_name_re=r'.*archive_.*\.json',
    parser_class_name='workflowparsers.asr.ASRParser',
    code_name='ASR',
    code_homepage='https://asr.readthedocs.io/en/latest/index.html',
    code_category='Workflow manager',
    metadata={
        'codeCategory': 'Workflow manager',
        'codeLabel': 'ASR',
        'codeLabelStyle': 'all in capitals',
        'codeName': 'asr',
        'codeUrl': 'https://asr.readthedocs.io/en/latest/index.html',
        'parserDirName': 'dependencies/workflow/workflowparsers/asr/',
        'parserGitUrl': 'https://github.com/nomad-coe/workflow-parsers.git',
        'parserSpecific': '',
        'preamble': '',
        'status': 'production',
        'tableOfFiles': '|Input Filename| Description|\n|--- | --- |\n|`archive*.json` | **Mainfile:** a json file w/ **user-defined** name|\n',
    },
)

atomate_parser_entry_point = EntryPoint(
    name='parsers/atomate',
    aliases=['parsers/atomate'],
    description='NOMAD parser for ATOMATE.',
    python_package='workflowparsers.atomate',
    mainfile_contents_re='"pymatgen_version":',
    mainfile_mime_re='(application/json)|(text/.*)',
    mainfile_name_re=r'.*mp.+materials\.json',
    parser_class_name='workflowparsers.atomate.AtomateParser',
    code_name='Atomate',
    code_homepage='https://www.atomate.org/',
    code_category='Workflow manager',
    metadata={
        'codeCategory': 'Workflow manager',
        'codeLabel': 'Atomate',
        'codeLabelStyle': 'Capitals: A',
        'codeName': 'atomate',
        'codeUrl': 'https://www.atomate.org/',
        'parserDirName': 'dependencies/workflow/workflowparsers/mp/',
        'parserGitUrl': 'https://github.com/nomad-coe/workflow-parsers.git',
        'parserSpecific': '',
        'preamble': '',
        'status': 'production',
        'tableOfFiles': '|Input Filename| Description|\n|--- | --- |\n|`*materials.json` | **Mainfile:** a json file containing system info|\n|`*.json` | json files containing workflow results|\n',
    },
)

elastic_parser_entry_point = EntryPoint(
    name='parsers/elastic',
    aliases=['parsers/elastic'],
    description='NOMAD parser for ELASTIC.',
    python_package='workflowparsers.elastic',
    mainfile_contents_re=r'\s*Order of elastic constants\s*=\s*[0-9]+\s*',
    mainfile_name_re='.*/INFO_ElaStic',
    parser_class_name='workflowparsers.elastic.ElasticParser',
    code_name='ElaStic',
    code_homepage='http://exciting.wikidot.com/elastic',
    code_category='Workflow manager',
    metadata={
        'codeCategory': 'Workflow manager',
        'codeLabel': 'ElaStic',
        'codeLabelStyle': 'capitals: E, S. This is part of the exciting project',
        'codeName': 'elastic',
        'codeUrl': 'http://exciting.wikidot.com/elastic',
        'parserDirName': 'dependencies/workflow/workflowparsers/elastic/',
        'parserGitUrl': 'https://github.com/nomad-coe/workflow-parsers.git',
        'parserSpecific': '',
        'preamble': '',
        'status': 'production',
        'tableOfFiles': '',
    },
)

fhivibes_parser_entry_point = EntryPoint(
    name='parsers/fhivibes',
    aliases=['parsers/fhivibes'],
    description='NOMAD parser for FHIVIBES.',
    python_package='workflowparsers.fhivibes',
    mainfile_binary_header_re=b'^\\x89HDF',
    mainfile_contents_dict={'__has_all_keys': ['I', 'a', 'b']},
    mainfile_mime_re='(application/x-hdf)',
    mainfile_name_re=r'^.*\.(nc)$',
    parser_class_name='workflowparsers.fhivibes.FHIVibesParser',
    code_name='FHI-vibes',
    code_homepage='https://vibes.fhi-berlin.mpg.de/',
    code_category='Workflow manager',
    metadata={
        'codeCategory': 'Workflow manager',
        'codeLabel': 'FHI-vibes',
        'codeLabelStyle': 'Capitals: FHI, the rest in lowercase; use dash.',
        'codeName': 'fhi-vibes',
        'codeUrl': 'https://vibes.fhi-berlin.mpg.de/',
        'parserDirName': 'dependencies/workflow/workflowparsers/fhi-vibes/',
        'parserGitUrl': 'https://github.com/nomad-coe/workflow-parsers.git',
        'parserSpecific': '',
        'preamble': '',
        'status': 'production',
        'tableOfFiles': '|Input Filename| Description|\n|--- | --- |\n|`<hdf_file>` | **Mainfile**, binary hdf file w/ ext .nc` |\n',
    },
)

lobster_parser_entry_point = EntryPoint(
    name='parsers/lobster',
    aliases=['parsers/lobster'],
    description='NOMAD parser for LOBSTER.',
    python_package='workflowparsers.lobster',
    mainfile_contents_re=r'^LOBSTER\s*v[\d\.]+.*',
    mainfile_name_re='.*lobsterout.*',
    supported_compressions=['gz', 'bz2', 'xz'],
    parser_class_name='workflowparsers.lobster.LobsterParser',
    code_name='LOBSTER',
    code_homepage='http://schmeling.ac.rwth-aachen.de/cohp/',
    code_category='Workflow manager',
    metadata={
        'codeCategory': 'Workflow manager',
        'codeLabel': 'LOBSTER',
        'codeLabelStyle': 'All in capitals',
        'codeName': 'lobster',
        'codeUrl': 'http://schmeling.ac.rwth-aachen.de/cohp/',
        'parserDirName': 'dependencies/workflow/workflowparsers/lobster/',
        'parserGitUrl': 'https://github.com/nomad-coe/workflow-parsers.git',
        'preamble': '',
        'status': 'production',
        'tableOfFiles': '|Input Filename| Description|\n|--- | --- |\n|`lobsterout` | **Mainfile** in LOBSTER specific plain-text |\n',
    },
)

phonopy_parser_entry_point = EntryPoint(
    name='parsers/phonopy',
    aliases=['parsers/phonopy'],
    description='NOMAD parser for PHONOPY.',
    python_package='workflowparsers.phonopy',
    mainfile_name_re='(.*/phonopy-FHI-aims-displacement-0*1/control.in$)|(.*/phon[^/]+yaml)',
    parser_class_name='workflowparsers.phonopy.PhonopyParser',
    code_name='phonopy',
    code_homepage='https://phonopy.github.io/phonopy/',
    code_category='Workflow manager',
    metadata={
        'codeCategory': 'Workflow manager',
        'codeLabel': 'phonopy',
        'codeLabelStyle': 'all in lower case',
        'codeName': 'phonopy',
        'codeUrl': 'https://phonopy.github.io/phonopy/',
        'parserDirName': 'dependencies/workflow/workflowparsers/phonopy/',
        'parserGitUrl': 'https://github.com/nomad-coe/workflow-parsers.git',
        'parserSpecific': '',
        'preamble': '',
        'status': 'production',
        'tableOfFiles': '',
    },
)

quantum_espresso_epw_parser_entry_point = EntryPoint(
    name='parsers/quantum_espresso_epw',
    aliases=['parsers/quantum_espresso_epw'],
    description='NOMAD parser for QUANTUM_ESPRESSO_EPW.',
    python_package='workflowparsers.quantum_espresso_epw',
    mainfile_contents_re=r'Program EPW.+\s*This program is part of the open-source Quantum ESPRESSO suite',
    parser_class_name='workflowparsers.quantum_espresso_epw.QuantumEspressoEPWParser',
    code_name='QuantumEspressoEPW',
    code_homepage='https://www.quantum-espresso.org',
    code_category='Workflow manager',
    metadata={
        'codeCategory': 'Workflow manager',
        'codeLabel': 'QuantumEspressoEPW',
        'codeLabelStyle': 'Capitals: Q, E, E, P, W',
        'codeName': 'quantumespressoepw',
        'codeUrl': 'https://www.quantum-espresso.org',
        'parserDirName': 'dependencies/workflow/workflowparsers/quantum_espresso_epw',
        'parserGitUrl': 'https://github.com/nomad-coe/workflow-parsers.git',
        'parserSpecific': '',
        'preamble': '',
        'status': 'production',
        'tableOfFiles': '',
    },
)

quantum_espresso_phonon_parser_entry_point = EntryPoint(
    name='parsers/quantum_espresso_phonon',
    aliases=['parsers/quantum_espresso_phonon'],
    description='NOMAD parser for QUANTUM_ESPRESSO_PHONON.',
    python_package='workflowparsers.quantum_espresso_phonon',
    mainfile_contents_re=r'Program PHONON.+\s*This program is part of the open-source Quantum ESPRESSO suite',
    parser_class_name='workflowparsers.quantum_espresso_phonon.QuantumEspressoPhononParser',
    code_name='QuantumEspressPhonon',
    code_homepage='https://www.quantum-espresso.org',
    code_category='Workflow manager',
    metadata={
        'codeCategory': 'Workflow manager',
        'codeLabel': 'QuantumEspressPhonon',
        'codeLabelStyle': 'Capitals: Q, E, P',
        'codeName': 'quantumespressophonon',
        'codeUrl': 'https://www.quantum-espresso.org',
        'parserDirName': 'dependencies/workflow/workflowparsers/quantumespressophonon',
        'parserGitUrl': 'https://github.com/nomad-coe/workflow-parsers.git',
        'parserSpecific': '',
        'preamble': '',
        'status': 'production',
        'tableOfFiles': '',
    },
)

quantum_espresso_xspectra_parser_entry_point = EntryPoint(
    name='parsers/quantum_espresso_xspectra',
    aliases=['parsers/quantum_espresso_xspectra'],
    description='NOMAD parser for QUANTUM_ESPRESSO_XSPECTRA.',
    python_package='workflowparsers.quantum_espresso_xspectra',
    mainfile_contents_re=r'\s*Program XSpectra\s*',
    mainfile_mime_re='(application/.*)|(text/.*)',
    parser_class_name='workflowparsers.quantum_espresso_xspectra.QuantumEspressoXSpectraParser',
    code_name='QuantumESPRESSOXSpectra',
    code_homepage='https://www.quantum-espresso.org/Doc/INPUT_XSpectra.txt',
    code_category='Workflow manager',
    metadata={
        'codeCategory': 'Workflow manager',
        'codeLabel': 'QuantumESPRESSOXSpectra',
        'codeLabelStyle': 'Capitals: Q, ESPRESSO, X, S',
        'codeName': 'quantumespressoxspectra',
        'codeUrl': 'https://www.quantum-espresso.org/Doc/INPUT_XSpectra.txt',
        'parserDirName': 'dependencies/workflow/workflowparsers/quantumespressoxspectra',
        'parserGitUrl': 'https://github.com/nomad-coe/workflow-parsers.git',
        'parserSpecific': '',
        'preamble': '',
        'status': 'production',
        'tableOfFiles': '| Input Filename | Description |\n| --- | --- |\n| `*.out` | **Mainfile:** text output file |\n| `*.dat` | output data file with the Absorption Spectra |\n',
    },
)
