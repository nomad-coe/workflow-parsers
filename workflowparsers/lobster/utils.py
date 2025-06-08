import os
import yaml

def generate_vasp_lobster_workflow_yaml(parent_directory, filename="workflow.archive.yaml",
                                        output_path="."):

    # Check if directory exists
    if not os.path.isdir(parent_directory):
        print(f"Directory '{parent_directory}' not found.")
        return

    # List all subdirectories inside the given directory
    dirs = [d for d in os.listdir(parent_directory) if os.path.isdir(os.path.join(parent_directory, d))]

    dirs.sort()

    if not dirs:
        print(f"No subdirectories found in '{parent_directory}'.")
        return

    tasks = []
    inputs = []
    outputs = []

    #if os.path.exists(get_lobster_file(f"{parent_directory}/{dirs[0]}/")

    # Define a single DFT run task (using the first directory for paths)
    ref_dir = dirs[0]
    vasprun_path = f"../upload/archive/mainfile/{ref_dir}/vasprun.xml.gz"

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
        lobsterout_path = f"../upload/archive/mainfile/{d}/lobsterout.gz"

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
