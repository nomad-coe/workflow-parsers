This is a NOMAD parser for [Quantum Espresso BANDS](https://www.quantum-espresso.org/) module.
It reads Quantum Espresso BANDS output files and provide all information in NOMAD's unified Metainfo based Archive format.

It also scans for a **single** representative PWSCF file that indicates a "Self-consistent Calculation".
Failure to provide any of these two files will result in either

- the band structure entry not being created.
- the band structure entry remaining mostly empty.