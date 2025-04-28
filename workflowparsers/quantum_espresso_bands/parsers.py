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
import re
import glob
import logging
import numpy as np
from typing import Optional, List, Dict, Any, Tuple

from nomad.units import ureg
from nomad.parsing.file_parser.text_parser import TextParser, Quantity
from runschema.run import Run, Program, TimeRun
from runschema.method import Method, Electronic, DFT, XCFunctional
from runschema.system import System, Atoms
from runschema.calculation import Calculation, BandEnergies, BandStructure, BandGapDeprecated, Energy, EnergyEntry
from simulationworkflowschema import SinglePoint

# Regular expressions
re_f = r'[-+]?\d+\.\d*(?:[Ee][-+]\d+)?'


class MainfileParser(TextParser):
    """Parser for PWSCF output file to extract data related to band structure calculations."""
    
    def init_quantities(self):
        # Define quantities to extract from PWSCF output
        self._quantities = [
            Quantity(
                'program_version',
                r'Program\s*(\w+\s*v\.\S+\s*(?:\(svn rev\.\s*\d+\))*)',
            ),
            Quantity(
                'start_date_time',
                r'starts (?:on|\.\.\.\s*Today is)\s*(\w+)\s*at\s*([\d: ]+)',
                flatten=False,
            ),
            # Lattice information
            Quantity(
                'alat',
                rf'lattice parameter \((?:alat|a_0)\)\s*=\s*({re_f})',
                unit='bohr',
                dtype=float,
            ),
            Quantity(
                'simulation_cell',
                r'a\(1\) = \(([\-\d\. ]+)\)\s*a\(2\) = \(([\-\d\. ]+)\)\s*a\(3\) = \(([\-\d\. ]+)\)\s*',
                dtype=float,
                shape=(3, 3),
            ),
            Quantity(
                'reciprocal_cell',
                r'b\(1\) = \(([\-\d\. ]+)\)\s*b\(2\) = \(([\-\d\. ]+)\)\s*b\(3\) = \(([\-\d\. ]+)\)\s*',
                dtype=float,
                shape=(3, 3),
            ),
            # Electronic structure information
            Quantity(
                'number_of_electrons',
                rf'number of electrons\s*=\s*({re_f})',
                dtype=float,
            ),
            Quantity(
                'number_of_bands',
                r'number of Kohn-Sham states\s*=\s*(\d+)',
                dtype=int,
            ),
            Quantity(
                'fermi_energy',
                rf'the\s+Fermi\s+energy\s+is\s+({re_f})\s+ev',
                dtype=float,
                unit='eV',
            ),
            Quantity(
                'highest_occupied',
                rf'highest\s+occupied\s+level\s+\(ev\):\s+({re_f})',
                dtype=float,
                unit='eV',
            ),
            Quantity(
                'lowest_unoccupied',
                rf'lowest\s+unoccupied\s+level\s+\(ev\):\s+({re_f})',
                dtype=float,
                unit='eV',
            ),
            Quantity(
                'band_gap',
                rf'the\s+gap\s+is\s+({re_f})\s+ev',
                dtype=float,
                unit='eV',
            ),
            # Positions and atomic information
            Quantity(
                'n_atoms',
                r'number of atoms/cell\s*=\s*(\d+)',
                dtype=int,
            ),
            Quantity(
                'atom_labels_positions',
                r'site n.\s+atom\s+positions \(alat units\)([\s\S]+?)\n\s*\n',
                sub_parser=TextParser(
                    quantities=[
                        Quantity(
                            'labels', 
                            r'\d+\s+(\w+)', 
                            repeats=True
                        ),
                        Quantity(
                            'positions',
                            rf'tau\(\s*\d+\s*\)\s*=\s*\(\s*({re_f})\s+({re_f})\s+({re_f})',
                            repeats=True,
                            dtype=float,
                        ),
                    ]
                ),
            ),
            # XC functional information
            Quantity(
                'xc_functional',
                r'Exchange-correlation\s*=\s*(.+)',
                flatten=False,
            ),
        ]


class BandsFileParser(TextParser):
    """Parser for Quantum ESPRESSO BANDS output files."""
    
    def init_quantities(self):
        self._quantities = [
            Quantity(
                'program_version',
                r'Program BANDS v\.(\S+)',
                dtype=str,
            ),
            Quantity(
                'kpoint',
                rf'xk=\(\s*({re_f}),\s*({re_f}),\s*({re_f})\s*\)',
                repeats=True,
                dtype=float,
            ),
            Quantity(
                'symmetry',
                r'Band symmetry, ([\w_]+)\s*\(.*\)\s+point group:',
                repeats=True,
                dtype=str,
            ),
            Quantity(
                'band',
                r'point group:([\s\S]+?)\n\n',
                repeats=True,
                sub_parser=TextParser(
                    quantities=[
                        Quantity(
                            'energy',
                            r'e\(\s*\d+ -\s*\d+\) =\s*([\-\d\.]+)\s+eV',
                            repeats=True,
                            dtype=float,
                        ),
                        Quantity(
                            'mult',
                            r'eV\s*(\d+)\s*-->',
                            repeats=True,
                            dtype=int,
                        ),
                    ],
                ),
            ),
        ]

    @staticmethod
    def scan_dir_for_files(directory: str, pattern: str = '*.out') -> list[str]:
        """Find all files matching the pattern in the given directory."""
        return glob.glob(os.path.join(directory, pattern))

    @staticmethod
    def read_header(filepath: str, num_lines: int = 2) -> str:
        """Read the first few lines of a file."""
        with open(filepath, 'r') as file:
            header = ''.join([file.readline() for _ in range(num_lines)])
        return header

    @staticmethod
    def is_bands_file(content: str) -> bool:
        """Check if the content is from a BANDS calculation output."""
        pattern = re.compile(r'Program BANDS v\.\d+\.\d+')
        return bool(pattern.search(content))

    @staticmethod
    def is_pwscf_file(content: str) -> bool:
        """Check if the content is from a PWSCF calculation output."""
        pattern = re.compile(r'Program PWSCF v\.\d+\.\d+')
        return bool(pattern.search(content))
        
    @staticmethod
    def determine_calculation_type(filepath: str) -> str:
        """
        Determine the type of calculation in a QE output file.
        
        Returns:
            str: 'scf' for self-consistent calculation, 'bands' for band structure,
                 'other' for other calculation types, 'unknown' if can't determine
        """
        # Patterns to identify different calculation types
        scf_pattern = re.compile(r'Self-consistent Calculation')
        bands_pattern = re.compile(r'Band Structure Calculation')
        other_calc_patterns = [
            re.compile(r'Geometry Optimization'),
            re.compile(r'Molecular Dynamics'),
            re.compile(r'Post-processing Calculation')
        ]
        
        with open(filepath, 'r') as f:
            # Read file in chunks to avoid loading entire large files
            chunk_size = 100000  # 100KB chunks
            
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                    
                # Check for SCF calculation
                if scf_pattern.search(chunk):
                    return 'scf'
                
                # Check for band structure calculation
                if bands_pattern.search(chunk):
                    return 'bands'
                
                # Check for other calculation types - if found, no need to read further
                for pattern in other_calc_patterns:
                    if pattern.search(chunk):
                        return 'other'
        
        return 'unknown'

    @staticmethod
    def points_to_segments(kpoints: list, symmetries: list) -> list[list[list[float]]]:
        """Split the kpoints by segment based on differing symmetry group."""

        def shift_window(window: tuple, elem) -> tuple:
            return window[1:] + (elem,)

        previous_point: Optional[list[float]] = None
        symmetry_window: tuple[Optional[str]] = (None,) * 3

        segments: list[list[np.ndarray[float]]] = [[]]
        for point, symmetry in zip(kpoints, symmetries):
            symmetry_window = shift_window(symmetry_window, symmetry)
            # case enumeration for `symmetry_window`:
            # 1. (None, None, None) -> not possible
            # 2. (None, None, X) -> first step (add to initial bucket)
            # 3. (X, None, None) -> not possible
            # 4. (X, Y, None) -> end reached
            # 5. (None, X, Y) -> add to initial bucket
            # 6. (X, X, Y) -> add Y to latest bucket
            # 7. (X, Y, Y) -> add Y to latest bucket
            # 8. (X, Y, X) -> add Y and X a new, latest bucket
            # 9. (X, Y, Z) -> add Y and Z a new, latest bucket
            if (None not in symmetry_window) and all(
                [symmetry_window[i] != symmetry_window[i + 1] for i in range(2)]
            ):
                segments.append([previous_point])
            segments[-1].append(point)
            previous_point = point
        return segments

    @staticmethod
    def apply_multiplicity(
        energies: list[list[float]], multiplicity: list[int]
    ) -> list[list[float]]:
        """Apply band multiplicity to energies."""
        return [
            [e for e, m in zip(energy, mult) for _ in range(m)]
            for energy, mult in zip(energies, multiplicity)
        ]


class QuantumEspressoBandsParser:
    """Parser for Quantum ESPRESSO BANDS calculations."""

    def __init__(self):
        self.bands_parser = BandsFileParser()
        self.pwscf_parser = MainfileParser()
        self.logger = None
        self.archive = None
        self.filepath = None
        self.maindir = None
        self.pwscf_file = None
        
    def _init_parsers(self):
        """Initialize parser objects and reset state."""
        self.bands_parser = BandsFileParser()
        self.pwscf_parser = MainfileParser()
        
    def _find_files(self):
        """Find BANDS and PWSCF files in the directory."""
        out_files = self.bands_parser.scan_dir_for_files(self.maindir)
        
        bands_files = []
        pwscf_files = []
        scf_pwscf_files = []
        
        for filepath in out_files:
            with open(filepath, 'r') as f:
                content = f.read(5000)  # Read first 5KB, enough for program identification
                
            if self.bands_parser.is_bands_file(content):
                bands_files.append(filepath)
            elif self.bands_parser.is_pwscf_file(content):
                # Determine calculation type more efficiently
                calc_type = self.bands_parser.determine_calculation_type(filepath)
                
                if calc_type == 'scf':
                    scf_pwscf_files.append(filepath)
                
                # Keep all PWSCF files as fallback
                pwscf_files.append(filepath)
        
        # If multiple SCF files are found, raise an error
        if len(scf_pwscf_files) > 1:
            raise ValueError(
                f"Multiple PWSCF files with self-consistent calculations found in {self.maindir}. "
                f"Cannot determine which one to use for electronic structure information. "
                f"Files: {', '.join(scf_pwscf_files)}"
            )
            
        # Return the SCF files first if available
        return bands_files, scf_pwscf_files if scf_pwscf_files else pwscf_files
        
    def parse(self, filepath, archive, logger):
        """
        Parse both PWSCF output and BANDS output files.
        
        Args:
            filepath: Path to the main output file
            archive: NOMAD archive where results will be stored
            logger: Logger for output messages
        """
        self.filepath = os.path.abspath(filepath)
        self.maindir = os.path.dirname(self.filepath)
        self.logger = logger if logger is not None else logging
        self.archive = archive
        
        self._init_parsers()
        
        # Find BANDS and PWSCF files
        try:
            bands_files, pwscf_files = self._find_files()
        except ValueError as e:
            self.logger.error(str(e))
            raise
        
        if not bands_files:
            self.logger.warning("No BANDS output files found.")
            return
            
        # Set the mainfile to the BANDS file (as it's the distinguishing file)
        self.bands_parser.mainfile = bands_files[0]
        self.bands_parser.parse()
        
        # Try to find and parse a corresponding PWSCF file
        if pwscf_files:
            self.logger.info(f"Using PWSCF file: {pwscf_files[0]} for electronic structure data")
            self.pwscf_file = pwscf_files[0]
            self.pwscf_parser.mainfile = self.pwscf_file
            self.pwscf_parser.parse()
        else:
            self.logger.warning("No PWSCF file with self-consistent calculation found. "
                              "Electronic structure information will be incomplete.")
        
        # Process the data and create the archive structure
        self._process_data()
    
    def _process_data(self):
        """Process the parsed data and create the archive structure."""
        # Create main run section
        sec_run = Run()
        self.archive.run.append(sec_run)
        
        # Set program information
        program_version = self.bands_parser.get('program_version')
        sec_run.program = Program(
            name='Quantum ESPRESSO BANDS',
            version=program_version if program_version else None
        )
        
        # Create system section (if PWSCF data available)
        if self.pwscf_parser.results:
            self._create_system_section(sec_run)
        
        # Create method section
        self._create_method_section(sec_run)
        
        # Create calculation section with band structure
        self._create_calculation_section(sec_run)
        
        # Set workflow
        self.archive.workflow2 = SinglePoint()
    
    def _create_system_section(self, sec_run):
        """Create the system section from PWSCF data."""
        sec_system = System()
        sec_run.system.append(sec_system)
        
        # Create atoms section
        sec_atoms = Atoms()
        sec_system.atoms = sec_atoms
        
        # Set lattice vectors (if available)
        alat = self.pwscf_parser.get('alat')
        cell = self.pwscf_parser.get('simulation_cell')
        if alat is not None and cell is not None:
            sec_atoms.lattice_vectors = cell * alat
        
        # Set reciprocal lattice vectors (if available)
        reciprocal_cell = self.pwscf_parser.get('reciprocal_cell')
        if reciprocal_cell is not None:
            # Calculate proper reciprocal cell if not directly available
            if alat is not None:
                sec_atoms.lattice_vectors_reciprocal = reciprocal_cell * (2 * np.pi / alat)
        
        # Add atom positions and labels (if available)
        atom_data = self.pwscf_parser.get('atom_labels_positions')
        if atom_data:
            labels = atom_data.get('labels')
            positions = atom_data.get('positions')
            if labels and positions:
                sec_atoms.labels = labels
                if alat is not None:
                    sec_atoms.positions = positions * alat
                else:
                    sec_atoms.positions = positions
        
        # Set number of atoms
        n_atoms = self.pwscf_parser.get('n_atoms')
        if n_atoms is not None:
            sec_system.atoms.n_atoms = n_atoms
        
        # Set periodic boundary conditions
        sec_atoms.periodic = [True, True, True]
    
    def _create_method_section(self, sec_run):
        """Create the method section."""
        sec_method = Method()
        sec_run.method.append(sec_method)
        
        # Set electronic structure method
        sec_electronic = Electronic()
        sec_method.electronic = sec_electronic
        sec_electronic.method = 'DFT'
        
        # Set number of electrons (if available)
        n_electrons = self.pwscf_parser.get('number_of_electrons')
        if n_electrons is not None:
            sec_electronic.n_electrons = n_electrons
        
        # Set number of bands (if available)
        n_bands = self.pwscf_parser.get('number_of_bands')
        if n_bands is not None:
            sec_electronic.n_bands = n_bands
        
        # Set XC functional (if available)
        xc_functional = self.pwscf_parser.get('xc_functional')
        if xc_functional is not None:
            sec_method.dft = DFT(xc_functional=XCFunctional(name=xc_functional))
    
    def _extract_reference_energy(self, sec_energy, band_energies=None):
        """
        Extract a reference energy for band structure plots.
        
        Following the QE documentation and best practices:
        1. First try to find Fermi energy (available for metals with smearing)
        2. If not found, use highest occupied level (always printed for systems with gap)
        3. If neither is found, calculate from band energies and number of electrons
        
        Args:
            sec_energy: Energy section to populate
            band_energies: Band energies list if available (for fallback calculation)
        
        Returns:
            reference_energy: The reference energy value in eV (or None if unavailable)
            reference_type: String identifying the type of reference ('fermi', 'homo', 'calculated')
        """
        reference_energy = None
        reference_type = None
        
        # Try to get Fermi energy first (for metals with smearing)
        fermi_energy = self.pwscf_parser.get('fermi_energy')
        if fermi_energy is not None:
            self.logger.info("Found Fermi energy in PWSCF output")
            reference_energy = fermi_energy
            reference_type = 'fermi'
            sec_energy.fermi = fermi_energy * ureg.eV
        
        # If no Fermi energy, try highest occupied level (for insulators)
        elif (highest_occupied := self.pwscf_parser.get('highest_occupied')) is not None:
            self.logger.info("Using highest occupied level as reference (Fermi energy not available)")
            reference_energy = highest_occupied
            reference_type = 'homo'
            sec_energy.highest_occupied = highest_occupied * ureg.eV
            
            # Add lowest unoccupied if available
            if (lowest_unoccupied := self.pwscf_parser.get('lowest_unoccupied')) is not None:
                sec_energy.lowest_unoccupied = lowest_unoccupied * ureg.eV
                
                # Calculate band gap
                band_gap = lowest_unoccupied - highest_occupied
                if band_gap > 0:
                    sec_energy.band_gap = [EnergyEntry(value=band_gap * ureg.eV)]
        
        # Last resort: calculate from band energies and number of electrons
        elif band_energies is not None and (n_electrons := self.pwscf_parser.get('number_of_electrons')) is not None:
            # Get number of bands from PWSCF output or calculate based on QE conventions
            n_bands = self.pwscf_parser.get('number_of_bands')
            n_occupied = int(n_electrons / 2)  # Integer division for number of occupied bands
            
            # Determine if this is likely a metal or insulator based on available data
            is_metal = False
            
            # Check if we have band information to infer metal/insulator
            if n_bands is not None:
                # If nbnd significantly exceeds valence bands, it's likely a metal
                # QE uses 20% more (min 4 more) bands for metals
                min_metal_bands = max(n_occupied + 4, int(n_occupied * 1.2))
                is_metal = n_bands >= min_metal_bands
                
                # Additional checks for metallic nature
                if fermi_energy is not None:
                    # If Fermi energy is specified, it's definitely a metal
                    is_metal = True
            
            # For a metal, Fermi level is somewhere within the bands
            # For an insulator, highest occupied level is the last occupied band
            if is_metal:
                # Need to estimate Fermi level - this is approximate
                if isinstance(band_energies, list) and len(band_energies) >= n_occupied:
                    # Sort band energies and take the n_occupied'th value
                    sorted_energies = sorted(band_energies)
                    estimated_fermi = sorted_energies[n_occupied-1]
                    reference_energy = estimated_fermi
                    reference_type = 'calculated_fermi'
                    sec_energy.fermi = estimated_fermi * ureg.eV
                    self.logger.info(
                        "System appears to be metallic; estimating Fermi energy "
                        "based on number of electrons and available bands"
                    )
            else:
                # For insulator, highest occupied is the band at n_occupied
                if isinstance(band_energies, list) and len(band_energies) >= n_occupied:
                    sorted_energies = sorted(band_energies)
                    homo = sorted_energies[n_occupied-1]
                    reference_energy = homo
                    reference_type = 'calculated_homo'
                    sec_energy.highest_occupied = homo * ureg.eV
                    self.logger.info(
                        "System appears to be an insulator; using calculated highest occupied level"
                    )
                    
                    # If there are bands above this, can estimate LUMO too
                    if len(sorted_energies) > n_occupied:
                        lumo = sorted_energies[n_occupied]
                        sec_energy.lowest_unoccupied = lumo * ureg.eV
                        
                        # Calculate band gap
                        band_gap = lumo - homo
                        if band_gap > 0:
                            sec_energy.band_gap = [EnergyEntry(value=band_gap * ureg.eV)]
        
        return reference_energy, reference_type
        
        # Extract band structure data
        kpoints = self.bands_parser.get('kpoint', [])
        symmetries = self.bands_parser.get('symmetry', [])
        bands = self.bands_parser.get('band', [])
        
        if kpoints and symmetries and bands:
            # Create band structure segments
            band_segments = []
            kpoint_segments = self.bands_parser.points_to_segments(kpoints, symmetries)
            
            for kpath in kpoint_segments:
                band_split = len(kpath)
                band_selection, bands = bands[:band_split], bands[band_split-1:]
                
                # Apply multiplicity to energies
                desymm_energies = self.bands_parser.apply_multiplicity(
                    [b.get('energy', []) * ureg.eV for b in band_selection],
                    [b.get('mult', []) for b in band_selection],
                )
                
                # Create band energies for this segment
                band_energy = BandEnergies(
                    kpoints=kpath,
                    energies=[desymm_energies],
                )
                band_segments.append(band_energy)
            
            # Add band structure to calculation
            if sec_run.system and sec_run.system[-1].atoms.lattice_vectors_reciprocal is not None:
                sec_calc.band_structure_electronic.append(
                    BandStructure(
                        segment=band_segments,
                        reciprocal_cell=sec_run.system[-1].atoms.lattice_vectors_reciprocal,
                    )
                )
                
                # Add reference to Fermi energy
                if fermi_energy is not None:
                    sec_calc.band_structure_electronic[-1].fermi = fermi_energy * ureg.eV
            else:
                self.logger.warning("Missing reciprocal lattice vectors for band structure.")