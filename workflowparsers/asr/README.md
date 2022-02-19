This is a NOMAD parser for [ASR](https://asr.readthedocs.io/en/latest/index.html). It will read Atomic Simulation Recipes input and
output files and provide all information in NOMAD's unified Metainfo based Archive format.

For ABINIT please provide at least the files from this table if applicable to your
calculations (remember that you can provide more files if you want):

|Input Filename| Description|
|--- | --- |
|`*.out` | **Mainfile:** a plain text file w/ **user-defined** name|
|`*.files`|plain text; user-defined filenames |
|`*.in`| plain text, input parameters|
|`*_o_DDB`|binary file, Derivative DataBases of total energy|
|`*_o_DEN`|binary file, charge density|
|`*_o_EIG`|text file, eigenvalues|
|`*_o_WFK`|binary file, wavefunction|
|`log` | plain text, redirection of screen output (`stdout`)|


