import logging
from nomad.utils import configure_logging
from nomad.datamodel.datamodel import EntryArchive
from nomad.metainfo import SchemaPackage, Quantity
from simulationworkflowschema.general import SimulationWorkflow

m_package = SchemaPackage()

configure_logging(console_log_level=logging.DEBUG)

class LOBSTERWorkflow(SimulationWorkflow):
    """
    A base section used to define LOBSTER workflows. These workflows are used to analyze bonds in materials.
    It involves a single point DFT run to generate wavefunction followed by projection of this wavefunction
    on a local atomic orbital basis with LOBSTER program. Depending on number of available projection basis,
    LOBSTER runs can be more than one.

    """

    name = Quantity(
        type=str,
        default='LOBSTER Workflow',
        description='Name of the workflow. Default set to `LOBSTER`.',
    )

    def normalize(self, archive: EntryArchive, logger: logging.Logger) -> None:
        super().normalize(archive, logger)


m_package.__init_metainfo__()
