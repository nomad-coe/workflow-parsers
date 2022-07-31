This is a collection of the NOMAD parsers for the following workflow engines.
1. [AFLOW](http://www.aflowlib.org/)
2. [ASR](https://asr.readthedocs.io/en/latest/index.html)
3. [ElaStic](http://exciting-code.org/elastic)
4. [FHI-vibes](https://vibes.fhi-berlin.mpg.de/)
5. [LOBSTER](http://schmeling.ac.rwth-aachen.de/cohp/)
6. [MaterialsProject](https://www.materialsproject.org/)
7. [Phonopy](https://phonopy.github.io/phonopy/)
8. [QuantumExpressoPhonon](https://www.quantum-espresso.org)

Each of the parsers will read the relevant input and output files and provide all information in
NOMAD's unified Metainfo based Archive format.

## Preparing code input and output file for uploading to NOMAD

NOMAD accepts `.zip` and `.tar.gz` archives as uploads. Each upload can contain arbitrary
files and directories. NOMAD will automatically try to choose the right parser for you files.
For each parser (i.e. for each supported code) there is one type of file that the respective
parser can recognize. We call these files `mainfiles` as they typically are the main
output file a code. For each `mainfile` that NOMAD discovers it will create an entry
in the database that users can search, view, and download. NOMAD will associate all files
in the same directory as files that also belong to that entry. Parsers
might also read information from these auxillary files. This way you can add more files
to an entry, even if the respective parser/code might not directly support it.

To create an upload with all calculations in a directory structure:

```
zip -r <upload-file>.zip <directory>/*
```

Go to the [NOMAD upload page](https://nomad-lab.eu/prod/rae/gui/uploads) to upload files
or find instructions about how to upload files from the command line.

## Using the parser

You can use NOMAD's parsers and normalizers locally on your computer. You need to install
NOMAD's pypi package:

```
pip install nomad-lab
```

To parse code input/output from the command line, you can use NOMAD's command line
interface (CLI) and print the processing results output to stdout:

```
nomad parse --show-archive <path-to-file>
```

To parse a file in Python, you can program something like this:
```python
import sys
from nomad.cli.parse import parse, normalize_all

# match and run the parser
archive = parse(sys.argv[1])
# run all normalizers
normalize_all(archive)

# get the 'main section' section_run as a metainfo object
section_run = archive.section_run[0]

# get the same data as JSON serializable Python dict
python_dict = section_run.m_to_dict()
```

## Developing the parser

Create a virtual environment to install the parser in development mode:

```
pip install virtualenv
virtualenv -p `which python3` .pyenv
source .pyenv/bin/activate
```

Install NOMAD's pypi package:

```
pip install nomad-lab
```

Clone the workflow parsers project and install it in development mode:

```
git clone https://github.com/nomad-coe/workflow-parsers.git workflow-parsers
pip install -e workflow-parsers
```

