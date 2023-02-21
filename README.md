This is a collection of the NOMAD parsers for the following workflow codes:

1. [AFLOW](http://www.aflowlib.org/)
2. [ASR](https://asr.readthedocs.io/en/latest/index.html)
3. [Atomate](https://www.atomate.org/)
4. [ElaStic](http://exciting.wikidot.com/elastic)
5. [FHI-vibes](https://vibes.fhi-berlin.mpg.de/)
6. [LOBSTER](http://schmeling.ac.rwth-aachen.de/cohp/)
7. [MOFStructures]()
8. [phonopy](https://phonopy.github.io/phonopy/)
9. [QuantumEspressoEPW](https://www.quantum-espresso.org)
10. [QuantumEspressPhonon](https://www.quantum-espresso.org)
11. [QuantumEspressoXSpectra](https://www.quantum-espresso.org/Doc/INPUT_XSpectra.txt)

## Preparing code input and output file for uploading to NOMAD

An *upload* is basically a directory structure with files. If you have all the files locally
you can just upload everything as a `.zip` or `.tar.gz` file in a single step. While the upload is
in the *staging area* (i.e. before it is published) you can also easily add or remove files in the
directory tree via the web interface. NOMAD will automatically try to choose the right parser
for you files.

For each parser there is one type of file that the respective parser can recognize. We call
these files *mainfiles*. For each mainfile that NOMAD discovers it will create an *entry*
in the database, which users can search, view, and download. NOMAD will consider all files
in the same directory as *auxiliary files* that also are associated with that entry. Parsers
might also read information from these auxillary files. This way you can add more files
to an entry, even if the respective parser/code might not use them. However, we strongly
recommend to not have multiple mainfiles in the same directory. For CMS calculations, we
recommend having a separate directory for each code run.

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

Clone the parser project and install it in development mode:

```
git clone https://github.com/nomad-coe/workflow-parsers.git workflow-parsers
pip install -e workflow-parsers
```

Running the parser now, will use the parser's Python code from the clone project.
