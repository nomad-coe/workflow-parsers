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
import numpy as np
import re
from fractions import Fraction

from ase import lattice as aselattice
from ase.cell import Cell
from ase.dft.kpoints import special_paths, parse_path_string, get_special_points

from phonopy.phonon.band_structure import BandStructure
from phonopy.units import EvTokJmol, VaspToTHz


def generate_kpath_parameters(points, paths, npoints):
    k_points = []
    for p in paths:
        k_points.append([points[k] for k in p])
        for index in range(len(p)):
            if p[index] == 'G':
                p[index] = 'Γ'
    parameters = []
    for h, seg in enumerate(k_points):
        for i, path in enumerate(seg):
            parameter = {}
            parameter['npoints'] = npoints
            parameter['startname'] = paths[h][i]
            if i == 0 and len(seg) > 2:
                parameter['kstart'] = path
                parameter['kend'] = seg[i + 1]
                parameter['endname'] = paths[h][i + 1]
                parameters.append(parameter)
            elif i == (len(seg) - 2):
                parameter['kstart'] = path
                parameter['kend'] = seg[i + 1]
                parameter['endname'] = paths[h][i + 1]
                parameters.append(parameter)
                break
            else:
                parameter['kstart'] = path
                parameter['kend'] = seg[i + 1]
                parameter['endname'] = paths[h][i + 1]
                parameters.append(parameter)
    return parameters


def read_kpath(filename):
    with open(filename) as f:
        string = f.read()

        labels = re.search(r'BAND_LABELS\s*=\s*(.+)', string)
        try:
            labels = labels.group(1).strip().split()
        except Exception:
            return

        points = re.search(r'BAND\s*=\s*(.+)', string)
        try:
            points = points.group(1)
            points = [float(Fraction(p)) for p in points.split()]
            points = np.reshape(points, (len(labels), 3))
            points = {labels[i]: points[i] for i in range(len(labels))}
        except Exception:
            return

        npoints = re.search(r'BAND_POINTS\s*\=\s*(\d+)', string)
        if npoints is not None:
            npoints = int(npoints.group(1))
        else:
            npoints = 100

    return generate_kpath_parameters(points, [labels], npoints)


def generate_kpath_ase(cell, symprec):
    try:
        lattice = aselattice.get_lattice_from_canonical_cell(Cell(cell))
        paths = parse_path_string(lattice.special_path)
        points = lattice.get_special_points()
    except Exception:
        paths = None
        points = None
    if paths is None:
        paths = special_paths['orthorhombic']
    if points is None:
        try:
            points = get_special_points(cell)
        except Exception:
            return []

    return generate_kpath_parameters(points, paths, 100)


class PhononProperties():
    def __init__(self, phonopy_obj, logger, **kwargs):
        self.logger = logger
        self.phonopy_obj = phonopy_obj
        self.t_max = kwargs.get('t_max', 1000)
        self.t_min = kwargs.get('t_min', 0)
        self.t_step = kwargs.get('t_step', 100)
        self.band_conf = kwargs.get('band_conf')

        self.n_atoms = len(phonopy_obj.unitcell)

        k_mesh = kwargs.get('k_mesh', 30)
        mesh_density = (2 * k_mesh ** 3) / self.n_atoms
        mesh_number = np.round(mesh_density**(1. / 3.))
        self.mesh = [mesh_number, mesh_number, mesh_number]

        self.n_atoms_supercell = len(phonopy_obj.supercell)

    def get_bandstructure(self):
        phonopy_obj = self.phonopy_obj

        frequency_unit_factor = VaspToTHz
        is_eigenvectors = False

        unit_cell = phonopy_obj.unitcell.get_cell()
        sym_tol = phonopy_obj.symmetry.tolerance
        if self.band_conf is not None:
            parameters = read_kpath(self.band_conf)
        else:
            parameters = generate_kpath_ase(unit_cell, sym_tol)
        if not parameters:
            return None, None, None

        # Distances calculated in phonopy.band_structure.BandStructure object
        # are based on absolute positions of q-points in reciprocal space
        # as calculated by using the cell which is handed over during instantiation.
        # Fooling that object by handing over a "unit cell" diag(1,1,1) instead clashes
        # with calculation of non-analytical terms.
        # Hence generate appropriate distances and special k-points list based on fractional
        # coordinates in reciprocal space (to keep backwards compatibility with previous
        # FHI-aims phonon implementation).
        bands = []
        bands_distances = []
        distance = 0.0
        bands_special_points = [distance]
        bands_labels = []
        label = parameters[0]["startname"]
        for b in parameters:
            kstart = np.array(b["kstart"])
            kend = np.array(b["kend"])
            npoints = b["npoints"]
            dk = (kend - kstart) / (npoints - 1)
            bands.append([(kstart + dk * n) for n in range(npoints)])
            dk_length = np.linalg.norm(dk)

            for n in range(npoints):
                bands_distances.append(distance + dk_length * n)

            distance += dk_length * (npoints - 1)
            bands_special_points.append(distance)
            label = [b["startname"], b["endname"]]
            bands_labels.append(label)

        bs_obj = BandStructure(
            bands, phonopy_obj.dynamical_matrix, with_eigenvectors=is_eigenvectors,
            factor=frequency_unit_factor)

        freqs = bs_obj.get_frequencies()

        return np.array(freqs), np.array(bands), np.array(bands_labels)

    def get_dos(self):
        phonopy_obj = self.phonopy_obj
        mesh = self.mesh

        phonopy_obj.set_mesh(mesh, is_gamma_center=True)
        q_points = phonopy_obj.get_mesh()[0]
        phonopy_obj.set_qpoints_phonon(q_points, is_eigenvectors=False)

        frequencies = phonopy_obj.get_qpoints_phonon()[0]
        self.frequencies = np.array(frequencies)
        min_freq = min(np.ravel(frequencies))
        max_freq = max(np.ravel(frequencies)) + max(np.ravel(frequencies)) * 0.05

        phonopy_obj.set_total_DOS(
            freq_min=min_freq, freq_max=max_freq, tetrahedron_method=True)
        f, dos = phonopy_obj.get_total_DOS()

        return f, dos

    def get_thermodynamical_properties(self):
        phonopy_obj = self.phonopy_obj

        phonopy_obj.set_mesh(self.mesh, is_gamma_center=True)
        phonopy_obj.set_thermal_properties(
            t_step=self.t_step, t_max=self.t_max, t_min=self.t_min)
        T, fe, entropy, cv = phonopy_obj.get_thermal_properties()
        kJmolToEv = 1.0 / EvTokJmol
        fe = fe * kJmolToEv
        JmolToEv = kJmolToEv / 1000
        cv = JmolToEv * cv
        return T, fe, entropy, cv
