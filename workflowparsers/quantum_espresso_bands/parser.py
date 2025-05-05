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
from typing import Optional, Any
from typing_extensions import TypeAlias

from nomad.units import ureg
from nomad.parsing.file_parser.text_parser import TextParser, Quantity
from nomad.search import search
from nomad.app.v1.models import MetadataRequired
from runschema.run import Run, Program
from runschema.method import Method, Electronic, DFT, XCFunctional
from runschema.system import System, Atoms
from runschema.calculation import (
    Calculation,
    BandEnergies,
    BandStructure,
    BandGapDeprecated,
    EnergyEntry,
)
from simulationworkflowschema import SinglePoint

# Use numpy types as a parent type of pint
KPoint: TypeAlias = np.ndarray[Any, np.dtype[np.float64]]

re_f = r'[-+]?\d*\.\d+|\d+'


class MainfileParser(TextParser):
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
            Quantity(
                'failed_symmetry',
                r'(zone border point and non-symmorphic group)',
                repeats=True,
                dtype=str,
                default='',
            ),
        ]

    @staticmethod
    def scan_dir_for_files(directory: str, pattern: str = '*.out') -> list[str]:
        """Find all files matching the pattern in the given directory."""
        return glob.glob(os.path.join(directory, pattern))

    @staticmethod
    def points_to_segments(kpoints: list, symmetries: list) -> list[list[KPoint]]:
        """Split the k-points by segment based on differing symmetry group."""

        def shift_window(window: tuple, elem) -> tuple:
            return window[1:] + (elem,)

        previous_kpoint: Optional[KPoint] = None
        symmetry_window: tuple[Optional[str], Optional[str], Optional[str]] = (
            None,
        ) * 3

        segments: list[list[KPoint]] = [[]]  # segments are lists of k-point paths
        for kpoint, symmetry in zip(kpoints, symmetries):
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
                segments.append([previous_kpoint])
            segments[-1].append(kpoint)
            previous_kpoint = kpoint
        return segments

    @staticmethod
    def apply_multiplicity(
        energies: list[list[float]], mults: list[list[int]]
    ) -> list[list[float]]:
        """Apply band multiplicity to energies."""
        return [
            [e for e, m in zip(energy, mult) for _ in range(m)]
            for energy, mult in zip(energies, mults)
        ]


class QuantumEspressoBandsParser:
    """Parser for Quantum ESPRESSO BANDS calculations."""

    def __init__(self):
        self.bands_parser = MainfileParser()
        self.pwscf_entry_id = None

    def get_mainfile_keys(self, filename: str, decoded_buffer: str) -> bool | list[str]:
        pw_patt = re.compile(r'Program PWSCF')
        scf_patt = re.compile(r'Self-consistent Calculation')
        band_patt = re.compile(r'Band Structure Calculation')

        if pw_patt.search(decoded_buffer):
            if scf_patt.search(decoded_buffer):
                return ['bs_analysis_pwscf']
            elif band_patt.search(decoded_buffer):
                return ['bs_analysis_bands']
                # this step is currently not analyzed, but covers some edge cases:
                # 1. no analysis was performed on the band structure
                # 2. the symmetry analysis failed
        return False

    def parse(self, filepath, archive, logger):
        """
        Parse both PWSCF output and BANDS output files.

        Args:
            filepath: Path to the main output file
            archive: NOMAD archive where results will be stored
            logger: Logger for output messages
        """
        self.bands_parser.mainfile = filepath
        self.maindir = os.path.dirname(filepath)
        self.logger = logger if logger is not None else logging
        self.archive = archive

        self.bands_parser.parse()

        if self.bands_parser.get('failed_symmetry'):
            self.logger.warning(
                'Failed symmetry analysis detected for some k-points.'
                'This may lead to missing k-points or even k-path segments.'
            )

        # Search NOMAD for matching SCF entry
        upload_id = self.archive.metadata.upload_id
        search_hits_data = search(
            owner='visible',
            user_id=self.archive.metadata.main_author.user_id,
            query={'upload_id': upload_id},
            required=MetadataRequired(include=['entry_id', 'mainfile']),
        ).data
        try:
            self.pwscf_entry_id = search_hits_data[0]['entry_id']
            search_hit = self.archive.m_context.load_archive(
                self.pwscf_entry, upload_id, None
            )
        except (IndexError, AttributeError):
            self.logger.warning(
                'No matching SCF entry found in the same entry.'
                'Parsing of the band structure will be incomplete.'
            )
            return None
        # TODO: handle multiple hits

        sec_run = Run(program=Program(name='Quantum ESPRESSO BANDS'))
        sec_run.program.version = self.bands_parser.get('program_version')
        self.archive.run.append(sec_run)

        self._create_calculation_section(sec_run, search_hit)
        self._create_system_section(sec_run)
        self._create_method_section(sec_run)

    def _create_system_section(self, sec_run):
        """Create the system section from PWSCF data."""
        sec_system = System()
        sec_run.system.append(sec_system)

        sec_atoms = Atoms()
        sec_system.atoms = sec_atoms
        sec_atoms.periodic = [True] * 3

        alat = self.pwscf_parser.get('alat')
        cell = self.pwscf_parser.get('simulation_cell')
        if alat is not None and cell is not None:
            sec_atoms.lattice_vectors = cell * alat
        reciprocal_cell = self.pwscf_parser.get('reciprocal_cell')
        if reciprocal_cell is not None:
            if alat is not None:
                sec_atoms.lattice_vectors_reciprocal = reciprocal_cell * (
                    2 * np.pi / alat
                )

    def _create_method_section(self, sec_run):
        """Create the method section."""
        sec_method = Method()
        sec_run.method.append(sec_method)

        sec_electronic = Electronic()
        sec_method.electronic = sec_electronic

        n_electrons = self.pwscf_parser.get('number_of_electrons')
        if n_electrons is not None:
            sec_electronic.n_electrons = n_electrons

        n_bands = self.pwscf_parser.get('number_of_bands')
        if n_bands is not None:
            sec_electronic.n_bands = n_bands

    def _create_calculation_section(self, sec_run: Run, search_hit: ArchiveSection):
        """Create the calculation section with band structure data."""
        sec_calc = Calculation()
        sec_run.calculation.append(sec_calc)

        # Reference the system and method
        if sec_run.system:
            sec_calc.system_ref = sec_run.system[-1]
        if sec_run.method:
            sec_calc.method_ref = sec_run.method[-1]

        # Extract band structure data
        kpoints = self.bands_parser.get('kpoint', [])
        symmetries = self.bands_parser.get('symmetry', [])
        if len(kpoints) == len(symmetries):
            kpoint_segments = self.bands_parser.points_to_segments(kpoints, symmetries)
        else:
            self.logger.error(
                'Mismatch between number of k-points and symmetry labels.'
                'Unable to create band structure segments.'
            )
            return None

        band_segments = []
        bands = self.bands_parser.get('band', [])
        for kpath in kpoint_segments:
            band_split = len(kpath)
            band_selection, bands = bands[:band_split], bands[band_split - 1 :]

            desymm_energies = self.bands_parser.apply_multiplicity(
                [b.get('energy', []) * ureg.eV for b in band_selection],
                [b.get('mult', []) for b in band_selection],
            )  # Apply multiplicity to energies

            band_energy = BandEnergies(
                kpoints=kpath,
                energies=[desymm_energies],
            )
            band_segments.append(band_energy)

        sec_band_structure = BandStructure(segment=band_segments, band_gap=[])
        sec_calc.band_structure_electronic.append(sec_band_structure)

        # Add reciprocal cell and alignment energies from reference SCF
        try:
            sec_band_structure.reciprocal_cell = search_hit.run.system[
                    -1
                ].atoms.lattice_vectors_reciprocal
        except (IndexError, AttributeError):
            self.logger.warning(f'No reciprocal lattice found in reference SCF (entry_id: {self.pwscf_entry_id}).')

        try:
            # TODO: handle multiple band gaps
            search_hit_bg = search_hit.results.properties.electronic.band_gap[0]
        except (IndexError, AttributeError):
            self.logger.error(f'No energy alignment data found in reference SCF (entry_id: {self.pwscf_entry_id}).')

        if (eho := search_hit_bg.energy_highest_occupied) is not None:
            sec_band_structure.energy_fermi = eho
            sec_band_gap = BandGapDeprecated(energy_highest_occupied=eho)
            sec_band_structure.band_gap.append(sec_band_gap)
        else:
            self.logger.error(f'No energy alignment data found in reference SCF (entry_id: {self.pwscf_entry_id}).')

        if (elu := search_hit_bg.energy_lowest_unoccupied) is not None:
            sec_band_gap.energy_lowest_unoccupied = elu
            sec_band_gap.value = sec_band_gap.energy_highest_occupied - sec_band_gap.energy_lowest_unoccupied
        else:
            self.logger.warning(f'No LUMO found in reference SCF (entry_id: {self.pwscf_entry_id}).')

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
            self.logger.info('Found Fermi energy in PWSCF output')
            reference_energy = fermi_energy
            reference_type = 'fermi'
            sec_energy.fermi = fermi_energy * ureg.eV

        # If no Fermi energy, try highest occupied level (for insulators)
        elif (
            highest_occupied := self.pwscf_parser.get('highest_occupied')
        ) is not None:
            self.logger.info(
                'Using highest occupied level as reference (Fermi energy not available)'
            )
            reference_energy = highest_occupied
            reference_type = 'homo'
            sec_energy.highest_occupied = highest_occupied * ureg.eV

            # Add lowest unoccupied if available
            if (
                lowest_unoccupied := self.pwscf_parser.get('lowest_unoccupied')
            ) is not None:
                sec_energy.lowest_unoccupied = lowest_unoccupied * ureg.eV

                # Calculate band gap
                band_gap = lowest_unoccupied - highest_occupied
                if band_gap > 0:
                    sec_energy.band_gap = [EnergyEntry(value=band_gap * ureg.eV)]

        # Last resort: calculate from band energies and number of electrons
        elif (
            band_energies is not None
            and (n_electrons := self.pwscf_parser.get('number_of_electrons'))
            is not None
        ):
            n_bands = self.pwscf_parser.get('number_of_bands')
            n_occupied = int(n_electrons / 2)

            # Determine if this is likely a metal or insulator based on available data
            is_metal = False
            if n_bands is not None:
                # If nbnd significantly exceeds valence bands, it's likely a metal
                # QE uses 20% more (min 4 more) bands for metals
                min_metal_bands = max(n_occupied + 4, int(n_occupied * 1.2))
                is_metal = n_bands >= min_metal_bands
                if fermi_energy is not None:
                    is_metal = True

            # For a metal, Fermi level is somewhere within the bands
            # For an insulator, highest occupied level is the last occupied band
            if is_metal:
                if isinstance(band_energies, list) and len(band_energies) >= n_occupied:
                    # Sort band energies and take the n_occupied'th value
                    sorted_energies = sorted(band_energies)
                    estimated_fermi = sorted_energies[n_occupied - 1]
                    reference_energy = estimated_fermi
                    reference_type = 'calculated_fermi'
                    sec_energy.fermi = estimated_fermi * ureg.eV
                    self.logger.info(
                        'System appears to be metallic; estimating Fermi energy '
                        'based on number of electrons and available bands'
                    )
            else:
                # For insulator, highest occupied is the band at n_occupied
                if isinstance(band_energies, list) and len(band_energies) >= n_occupied:
                    sorted_energies = sorted(band_energies)
                    homo = sorted_energies[n_occupied - 1]
                    reference_energy = homo
                    reference_type = 'calculated_homo'
                    sec_energy.highest_occupied = homo * ureg.eV
                    self.logger.info(
                        'System appears to be an insulator; using calculated highest occupied level'
                    )

                    # If there are bands above this, can estimate LUMO too
                    if len(sorted_energies) > n_occupied:
                        lumo = sorted_energies[n_occupied]
                        sec_energy.lowest_unoccupied = lumo * ureg.eV

                        # Calculate band gap
                        band_gap = lumo - homo
                        if band_gap > 0:
                            sec_energy.band_gap = [
                                EnergyEntry(value=band_gap * ureg.eV)
                            ]

        return reference_energy, reference_type
