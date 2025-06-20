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
import yaml
from monty.os.path import zpath

def generate_vasp_lobster_workflow_yaml(parent_directory,
                                        filename="workflow.archive.yaml",
                                        output_path="."):
    """
    Helper function to generate workflow yaml files from LOBSTER calc directory.

    The calc directory contains both VASP and LOBSTER run files. If multiple
    directories are found in the provided parent_directory,
    all will be grouped together. Thus, please ensure that the directories contain calculation files
    of the same structure with different projection basis in LOBSTER runs
    to work as expected.
    """

    # Check if directory exists
    if not os.path.isdir(parent_directory):
        print(f"Directory {parent_directory} not found.")
        return

    # List all subdirectories inside the given directory
    dirs = [d for d in os.listdir(parent_directory) if os.path.isdir(os.path.join(parent_directory, d))]

    dirs.sort()

    if not dirs:
        print(f"No subdirectories found in {parent_directory}.")
        return

    tasks = []
    inputs = []
    outputs = []

    # Define a single DFT run task (using the first directory for paths)
    ref_dir = dirs[0]
    local_vasprun_path = zpath(os.path.join(parent_directory, ref_dir, "vasprun.xml"))

    if os.path.isfile(local_vasprun_path):
        vasprun_file_name = os.path.basename(local_vasprun_path)
    else:
        raise ValueError(f"vasprun.xml file not found in the {ref_dir} directory required"
                         " for generating the LOBSTERWorkflow yaml")

    vasprun_path = f"../upload/archive/mainfile/{ref_dir}/{vasprun_file_name}"

    dft_task = {
        'm_def': 'nomad.datamodel.metainfo.workflow.TaskReference',
        'task': f"{vasprun_path}#/workflow2",
        'name': 'DFT run',
        'inputs': [{
            'name': 'Input structure',
            'section': f"{vasprun_path}#/run/0/system/-1"
        }],
        'outputs': [{
            'name': 'Output DFT calculation',
            'section': f"{vasprun_path}#/run/0/calculation/-1"
        }]
    }
    tasks.append(dft_task)

    # Add the DFT input and output to top-level workflow
    inputs.append({
        'name': 'Structure',
        'section': f"{vasprun_path}#/run/0/system/-1"
    })

    # Add LOBSTER run tasks for each directory
    for i, d in enumerate(dirs):
        local_lobsterout_path = zpath(os.path.join(parent_directory, ref_dir, "lobsterout"))

        if os.path.isfile(local_lobsterout_path):
            lobsterout_file_name = os.path.basename(zpath(os.path.join(parent_directory, ref_dir, "lobsterout")))
        else:
            raise ValueError(f"lobsterout file not found in the {d} directory required"
                             " for generating the LOBSTERWorkflow yaml")

        lobsterout_path = f"../upload/archive/mainfile/{d}/{lobsterout_file_name}"

        lobster_task = {
            'task': f"{lobsterout_path}#/workflow2",
            'name': f"LOBSTER run {i+1}",
            'inputs': [{
                'name': 'Structure and Planewavefunction',
                'section': f"{vasprun_path}#/run/0/calculation/-1"
            }],
            'outputs': [{
                'name': 'Bonding analysis data',
                'section': f"{lobsterout_path}#/run/0/calculation/-1"
            }]
        }

        tasks.append(lobster_task)

        # Top-level workflow output for each LOBSTER run
        outputs.append({
            'name': f'LOBSTER Output {i+1}',
            'section': f"{lobsterout_path}#/run/0/calculation/-1"
        })

    # Set workflow YAML structure
    yaml_data = {
        'workflow2': {
            'm_def': 'workflowparsers.lobster.workflow.LOBSTERWorkflow',
            'inputs': inputs,
            'outputs': outputs,
            'tasks': tasks
        }
    }

    output_path = os.path.join(output_path, filename)

    # Write to YAML file
    with open(output_path, 'w') as f:
        yaml.dump(yaml_data, f, sort_keys=False, default_flow_style=False)
