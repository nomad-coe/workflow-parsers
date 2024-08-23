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

from nomad.config.models.plugins import ParserEntryPoint


class EntryPoint(ParserEntryPoint):
    parser_class_name: str = Field(
        description="""
        The fully qualified name of the Python class that implements the parser.
        This class must have a function `def parse(self, mainfile, archive, logger)`.
    """
    )

    def load(self):
        from nomad.parsing import MatchingParserInterface
        from . import (
            aflow,
            asr,
            atomate,
            elastic,
            fhivibes,
            lobster,
            phonopy,
            quantum_espresso_epw,
            quantum_espresso_phonon,
            quantum_espresso_xspectra,
        )

        return MatchingParserInterface(self.parser_class_name)


aflow_parser_entry_point = EntryPoint(
    name='parsers/aflow',
    description='NOMAD parser for AFLOW.',
    python_package='workflowparsers.aflow',
    mainfile_contents_re=(r"^\s*\[AFLOW\] \*+\s*\[AFLOW\]\s*\[AFLOW\]                     .o.        .o88o."
        r"oooo\s*\[AFLOW\]                    .888.       888 `` `888\s*\[AFLOW\]                   .8'888.     o888oo   888   .ooooo.  oooo"
        r"oooo    ooo\s*\[AFLOW\]                  .8' `888.     888     888  d88' `88b  `88."
        r"`88.  .8'\s*\[AFLOW\]                 .88ooo8888.    888     888  888   888   `88..]88..8'\s*\[AFLOW\]                .8'     `888.   888     888  888   888    `888'`888'\s*\[AFLOW\]               o88o     o8888o"
        r"o888o   o888o `Y8bod8P'     `8'  `8'  .in|^\s*\{\"aurl\"\:\"aflowlib\.duke\.edu\:AFLOWDATA"),
    mainfile_mime_re='(application/json)|(text/.*)',
    mainfile_name_re=r'.*aflowlib\.json.*',
    mainfile_alternative=True,
    supported_compressions=['gz', 'bz2', 'xz'],
    parser_class_name='workflowparsers.aflow.AFLOWParser',
)

asr_parser_entry_point = EntryPoint(
    name='parsers/asr',
    description='NOMAD parser for ASR.',
    python_package='workflowparsers.asr',
    mainfile_contents_re='"name": "ASR"',
    mainfile_mime_re='(application/json)|(text/.*)',
    mainfile_name_re=r'.*archive_.*\.json',
    parser_class_name='workflowparsers.asr.ASRParser',
)

atomate_parser_entry_point = EntryPoint(
    name='parsers/atomate',
    description='NOMAD parser for ATOMATE.',
    python_package='workflowparsers.atomate',
    mainfile_contents_re='"pymatgen_version":',
    mainfile_mime_re='(application/json)|(text/.*)',
    mainfile_name_re=r'.*mp.+materials\.json',
    parser_class_name='workflowparsers.atomate.AtomateParser',
)

elastic_parser_entry_point = EntryPoint(
    name='parsers/elastic',
    description='NOMAD parser for ELASTIC.',
    python_package='workflowparsers.elastic',
    mainfile_contents_re=r'\s*Order of elastic constants\s*=\s*[0-9]+\s*',
    mainfile_name_re='.*/INFO_ElaStic',
    parser_class_name='workflowparsers.elastic.ElasticParser',
)

fhivibes_parser_entry_point = EntryPoint(
    name='parsers/fhivibes',
    description='NOMAD parser for FHIVIBES.',
    python_package='workflowparsers.fhivibes',
    mainfile_binary_header_re=b'^\\x89HDF',
    mainfile_contents_dict={'__has_all_keys': ['I', 'a', 'b']},
    mainfile_mime_re='(application/x-hdf)',
    mainfile_name_re=r'^.*\.(nc)$',
    parser_class_name='workflowparsers.fhivibes.FHIVibesParser',
)

lobster_parser_entry_point = EntryPoint(
    name='parsers/lobster',
    description='NOMAD parser for LOBSTER.',
    python_package='workflowparsers.lobster',
    mainfile_contents_re=r'^LOBSTER\s*v[\d\.]+.*',
    mainfile_name_re='.*lobsterout.*',
    supported_compressions=['gz', 'bz2', 'xz'],
    parser_class_name='workflowparsers.lobster.LobsterParser',
)

phonopy_parser_entry_point = EntryPoint(
    name='parsers/phonopy',
    description='NOMAD parser for PHONOPY.',
    python_package='workflowparsers.phonopy',
    mainfile_name_re='(.*/phonopy-FHI-aims-displacement-0*1/control.in$)|(.*/phon[^/]+yaml)',
    parser_class_name='workflowparsers.phonopy.PhonopyParser',
)

quantum_espresso_epw_parser_entry_point = EntryPoint(
    name='parsers/quantum_espresso_epw',
    description='NOMAD parser for QUANTUM_ESPRESSO_EPW.',
    python_package='workflowparsers.quantum_espresso_epw',
    mainfile_contents_re=r'Program EPW.+\s*This program is part of the open-source Quantum ESPRESSO suite',
    parser_class_name='workflowparsers.quantum_espresso_epw.',
)

quantum_espresso_phonon_parser_entry_point = EntryPoint(
    name='parsers/quantum_espresso_phonon',
    description='NOMAD parser for QUANTUM_ESPRESSO_PHONON.',
    python_package='workflowparsers.quantum_espresso_phonon',
    mainfile_contents_re=r'Program PHONON.+\s*This program is part of the open-source Quantum ESPRESSO suite',
    parser_class_name='workflowparsers.quantum_espresso_phonon.',
)

quantum_espresso_xspectra_parser_entry_point = EntryPoint(
    name='parsers/quantum_espresso_xspectra',
    description='NOMAD parser for QUANTUM_ESPRESSO_XSPECTRA.',
    python_package='workflowparsers.quantum_espresso_xspectra',
    mainfile_contents_re=r'\s*Program XSpectra\s*',
    mainfile_mime_re='(application/.*)|(text/.*)',
    parser_class_name='workflowparsers.quantum_espresso_xspectra.',
)
