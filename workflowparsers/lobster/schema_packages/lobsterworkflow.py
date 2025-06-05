from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nomad.datamodel.datamodel import EntryArchive
    from structlog.stdlib import BoundLogger

from nomad.config import config
from nomad.metainfo import Quantity, SchemaPackage, Quantity
from simulationworkflowschema.general import SimulationWorkflow

configuration = config.get_plugin_entry_point(
    'workflowparsers.lobster.schema_packages:nomad_lobster_workflows_plugin'
)

m_package = SchemaPackage()


class LOBSTERWorkflow(SimulationWorkflow):
    """
    A base section used to define LOBSTER workflows. These workflows are used to analyze bonds in materials. 
    It involves a single point DFT run to generate wavefunction followed by projection of this wavefunction 
    on a local atomic orbital basis with LOBSTER program. Depending on number of available projection basis,
    LOBSTER runs can be more than one.

    """

    name = Quantity(
        type=str,
        default='LOBSTER',
        description='Name of the workflow. Default set to `LOBSTER`.',
    )

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super().normalize(archive, logger)

        # Extract system name from input structure (chemical composition of first image)
        try:
            system_name = self.tasks[0].inputs[0].section.chemical_composition_hill
        except (KeyError, IndexError, AttributeError):
            system_name = None

        # Dynamically set entry name
        archive.metadata.entry_type = 'LOBSTER Workflow'
        if system_name is not None:
            archive.metadata.entry_name = f'{system_name} LOBSTER Calculation'
        else:
            archive.metadata.entry_name = 'LOBSTER Calculation'


m_package.__init_metainfo__()