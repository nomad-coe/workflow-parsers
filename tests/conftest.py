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
import pytest
import json

from nomad.datamodel import EntryArchive, EntryMetadata

from nomad.utils.exampledata import ExampleData
from nomad import infrastructure
from nomad.datamodel.context import ServerContext
from nomad.utils import create_uuid


# Set up pytest to pass control to the debugger on an exception.
if os.getenv('_PYTEST_RAISE', '0') != '0':

    @pytest.hookimpl(tryfirst=True)
    def pytest_exception_interact(call):
        raise call.excinfo.value

    @pytest.hookimpl(tryfirst=True)
    def pytest_internalerror(excinfo):
        raise excinfo.value


@pytest.fixture(scope='session')
def upload_id():
    return 'test_upload'


@pytest.fixture(scope='session')
def main_author():
    from nomad.config import config  # noqa

    return infrastructure.user_management.get_user(username=config.client.user)


@pytest.fixture(scope='session')
def auth():
    from nomad.client import Auth  # noqa
    from nomad.config import config  # noqa

    return Auth(user=config.client.user, password=config.client.password, from_api=True)


@pytest.fixture(scope='session')
def upload_files():
    # add pre-parsed json achive files only to avoid loading parsers
    return {'tests/data/lobster/Fe/vasprun.archive.json': 'parsers/vasp'}


@pytest.fixture(scope='session')
def upload_archives(upload_files, main_author, upload_id):
    archives = []
    for mainfile, parser in upload_files.items():
        entry_id = create_uuid()
        archive = EntryArchive(metadata=EntryMetadata())
        with open(mainfile) as f:
            archive.m_update_from_dict(json.load(f))
        archive.metadata.main_author = main_author
        archive.metadata.parser_name = parser
        archive.metadata.upload_id = upload_id
        archive.metadata.entry_id = entry_id
        archive.metadata.mainfile = mainfile
        archives.append(archive)
    return archives


@pytest.fixture(scope='session')
def upload_data(upload_id, main_author, upload_archives):

    infrastructure.setup_mongo()
    infrastructure.setup_elastic()

    data = ExampleData(main_author=main_author)
    data.create_upload(upload_id=upload_id)
    for archive in upload_archives:
        archive = data.create_entry(
            entry_archive=archive,
            entry_id=archive.metadata.entry_id,
            upload_id=upload_id,
            mainfile=archive.metadata.mainfile,
        )
    data.save()


@pytest.fixture(scope='session')
def context(upload_data, upload_id, main_author):
    from nomad.app.v1.routers.uploads import get_upload_with_read_access  # noqa

    upload = get_upload_with_read_access(upload_id, main_author)
    return ServerContext(upload)
