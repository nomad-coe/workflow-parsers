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
import logging
import numpy as np
from nomad.utils import configure_logging
from nomad.datamodel.datamodel import EntryArchive
from nomad.metainfo import SchemaPackage, Quantity
from simulationworkflowschema.general import SerialSimulation

m_package = SchemaPackage()

configure_logging(console_log_level=logging.DEBUG)

class LOBSTERWorkflow(SerialSimulation):
    """
    A base section used to define LOBSTER workflows. These workflows are used to analyze bonds in materials.
    It involves a single point DFT run to generate wavefunction followed by projection of this wavefunction
    on a local atomic orbital basis with LOBSTER program. Depending on number of available projection basis,
    LOBSTER runs can be more than one.

    """

    name = Quantity(
        type=str,
        default='LOBSTER Workflow',
        description='Name of the workflow. Default set to `LOBSTER Workflow`.',
    )

    charge_spillings = Quantity(
        type=np.float64,
        shape=["*", "*"],
        description="""
        Absolute charge spilling of density (in occupied levels)
        when projecting from the original wave functions into the local basis
        for each of the LOBSTER runs of shape [spilling x number_of_spin_channels].
        """,
    )

    def extract_charge_spillings(
        self, logger: logging.Logger
    ):
        """
        Extracts the charge spillings from the task outputs of the LOBSTER workflow.

        Args:
            logger: The logger to log messages.

        Returns:
            : The charge spillings for each of the LOBSTER runs in the workflow.
        """

        # Append the charge spillings for each of the LOBSTER runs
        charge_spillings = []
        for output in self.outputs:
            if output.section.x_lobster_abs_charge_spilling is not None:
                charge_spillings.append(
                    output.section.x_lobster_abs_charge_spilling
                )
            else:
                charge_spillings.append(None)  # Handle missing values

        return charge_spillings

    def normalize(self, archive: EntryArchive, logger: logging.Logger) -> None:

        super().normalize(archive, logger)

        try:
            self.charge_spillings = self.extract_charge_spillings(
                logger=logger
            )
        except Exception:
            logger.error('Could not set LOBSTERWorkflow.charge_spillings.')


        # Extract system name from input structure (chemical composition from DFT run)
        try:
            system_name = self.tasks[0].inputs[0].section.chemical_composition_hill
        except (KeyError, IndexError, AttributeError):
            system_name = None

        # Dynamically set entry name
        archive.metadata.entry_type = 'LOBSTER Workflow'
        if system_name is not None:
            archive.metadata.entry_name = f'{system_name} LOBSTER Calculations'
        else:
            archive.metadata.entry_name = 'LOBSTER Calculations'

m_package.__init_metainfo__()
