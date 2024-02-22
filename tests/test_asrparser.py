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

import pytest
import os
import json
import subprocess

from nomad.datamodel import EntryArchive
from workflowparsers.asr import asr_to_archives


# TODO this does not work for the current version of ASR

tests_path = 'tests/data/asr'


def approx(value, abs=0, rel=1e-6):
    return pytest.approx(value, abs=abs, rel=rel)


def clear_database():
    try:
        subprocess.Popen('rm -rf %s/.asr' % tests_path)
    except Exception:
        pass


@pytest.fixture(scope='module')
def test_database():
    # TODO not sure how to do this properly
    cwd = os.getcwd()
    try:
        os.chdir(tests_path)
        subprocess.Popen(
            ['asr', 'init'], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        subprocess.Popen(
            ['asr', 'run', 'asr.c2db.relax -a Si.json -c {"name":"emt"}'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except Exception:
        pass
    finally:
        os.chdir(cwd)


def test_parsing(test_database):
    asr_to_archives(tests_path)
    archive_files = [f for f in os.listdir(tests_path) if f.startswith('archive_')]
    # assert len(archive_files) > 0
    for f in archive_files:
        data = json.load(open(os.path.join(tests_path, f)))
        archive = EntryArchive()
        archive = archive.m_from_dict(data)
        assert len(archive.run) > 0
    clear_database()
