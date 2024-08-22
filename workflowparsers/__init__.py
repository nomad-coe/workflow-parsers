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
from nomad.config.models.plugins import ParserEntryPoint


class EntryPoint(ParserEntryPoint):
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
    parser_class_name='workflowparsers.aflow.AFLOWParser',
)

asr_parser_entry_point = EntryPoint(
    name='parsers/asr',
    description='NOMAD parser for ASR.',
    parser_class_name='workflowparsers.asr.ASRParser',
)

atomate_parser_entry_point = EntryPoint(
    name='parsers/atomate',
    description='NOMAD parser for ATOMATE.',
    parser_class_name='workflowparsers.atomate.AtomateParser',
)

elastic_parser_entry_point = EntryPoint(
    name='parsers/elastic',
    description='NOMAD parser for ELASTIC.',
    parser_class_name='workflowparsers.elastic.ElasticParser',
)

fhivibes_parser_entry_point = EntryPoint(
    name='parsers/fhivibes',
    description='NOMAD parser for FHIVIBES.',
    parser_class_name='workflowparsers.fhivibes.FHIVibesParser',
)

lobster_parser_entry_point = EntryPoint(
    name='parsers/lobster',
    description='NOMAD parser for LOBSTER.',
    parser_class_name='workflowparsers.lobster.LobsterParser',
)

phonopy_parser_entry_point = EntryPoint(
    name='parsers/phonopy',
    description='NOMAD parser for PHONOPY.',
    parser_class_name='workflowparsers.phonopy.PhonopyParser',
)

quantum_espresso_epw_parser_entry_point = EntryPoint(
    name='parsers/quantum_espresso_epw',
    description='NOMAD parser for QUANTUM_ESPRESSO_EPW.',
    parser_class_name='workflowparsers.quantum_espresso_epw.',
)

quantum_espresso_phonon_parser_entry_point = EntryPoint(
    name='parsers/quantum_espresso_phonon',
    description='NOMAD parser for QUANTUM_ESPRESSO_PHONON.',
    parser_class_name='workflowparsers.quantum_espresso_phonon.',
)

quantum_espresso_xspectra_parser_entry_point = EntryPoint(
    name='parsers/quantum_espresso_xspectra',
    description='NOMAD parser for QUANTUM_ESPRESSO_XSPECTRA.',
    parser_class_name='workflowparsers.quantum_espresso_xspectra.',
)
