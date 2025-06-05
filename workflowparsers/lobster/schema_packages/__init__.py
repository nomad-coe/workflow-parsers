from nomad.config.models.plugins import SchemaPackageEntryPoint
from pydantic import Field


class NOMADLOBSTERWorkflowsEntryPoint(SchemaPackageEntryPoint):
    parameter: int = Field(0, description='Custom configuration parameter')

    def load(self):
        from workflowparsers.lobster.schema_packages.lobsterworkflow import m_package

        return m_package


nomad_lobster_workflows_plugin = NOMADLOBSTERWorkflowsEntryPoint(
    name='NOMADLOBSTERWorkflows',
    description='Schema package plugin for the NOMAD LOBSTER workflows definitions.',
)
