"""
The submodule structure contains the main molcraft classes.

It is created to work with input and output files. It is also used to extract
information from a molecular system. It defines the three main classes: BULK,
MOL and ATOM.

"""

import re
import pandas as pd
import numpy as np
import networkx as nx
import itertools as it
from scipy.spatial.distance import cdist
from molcraft.exceptions import MoleculeDefintionError

Elements = {  # g/mol
    'H': {'mass': 1.0080, 'num': 1},
    'C': {'mass': 12.011, 'num': 6},
    'N': {'mass': 14.0067, 'num': 7},
    'O': {'mass': 16.0000, 'num': 8}

}


def load_xyz(file, warning=True):
    """
    Read a file xyz.

    Parameters
    ----------
    file : str
        File path.

    warning : bool
        Display or no warning when reading a file.

    Returns
    -------
    DataFrame
        The element symbols, mass, atom number and coordinates (in angstroms).

    """
    # Regular expression that extracts matrix XYZ.
    atoms = re.compile(r"""
            ^\s+
            (?P<atsb>[A-Za-z]+\d?\d?)\s+      # Atom name.
            (?P<x>[+-]?\d+\.\d+)\s+           # Orthogonal coordinates for X.
            (?P<y>[+-]?\d+\.\d+)\s+           # Orthogonal coordinates for Y.
            (?P<z>[+-]?\d+\.\d+)\s+           # Orthogonal coordinates for Z.
            """, re.X)

    xyz = []
    with open(file, "r") as XYZ:
        for line in XYZ:
            if atoms.match(line):
                m = atoms.match(line)
                xyz.append(m.groupdict())

    coord = pd.DataFrame(xyz)
    coord = coord.astype({
        "x": np.float64,
        "y": np.float64,
        "z": np.float64
    })

    try:
        coord["mass"] = coord["atsb"].apply(lambda at: Elements[at]["mass"])
        coord["num"] = coord["atsb"].apply(lambda at: Elements[at]["num"])
    except KeyError:
        if warning:
            print("Careful! the atomic symbols present were not recognized.")
        else:
            pass
    # This file has no partial charges .
    coord["charge"] = 0.0

    return coord


def load_mol2(file):
    """
    Obtain the geometry and partial charges from mol2 file.

    Parameters:
    -----------
    file : str
        Input file name

    Returns
    -------
    DataFrame
        The element symbols, mass, atom number and coordinates (in angstroms).

    """
    atoms = re.compile(r"""
        ^\s+(?P<atid>\d+)\s+              # Atom serial number.
        (?P<atsb>[A-Za-z]+)\d?\d?\s+      # Atom name.
        (?P<x>[+-]?\d+\.\d+)\s+           # Orthogonal coordinates for X.
        (?P<y>[+-]?\d+\.\d+)\s+           # Orthogonal coordinates for Y.
        (?P<z>[+-]?\d+\.\d+)\s+           # Orthogonal coordinates for Z.
        \w+\s+\d\s+\w+\s+
        (?P<charge>[+-]?\d+\.\d+)         # Charges
        """, re.X)

    dat = list()
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            if atoms.match(line):
                m = atoms.match(line)
                dat.append(m.groupdict())

    dfatoms = pd.DataFrame(dat)

    dfatoms = dfatoms.astype({
        "x": np.float,
        "y": np.float,
        "z": np.float,
        "charge": np.float,
        "atid": np.integer})

    # dfatoms = dfatoms.set_index('atid')

    """ Adding mass """
    dfatoms["mass"] = dfatoms["atsb"].apply(lambda at: Elements[at]["mass"])

    """ Adding atnum """
    dfatoms["num"] = dfatoms["atsb"].apply(lambda at: Elements[at]["num"])

    return dfatoms


def load_pdb(file):
    """
    Obtain the geometry of the system from pdb file.

    Parameters:
    -----------
    file : str
        Input file name

    Returns
    -------
    DataFrame
        The element symbols, mass, atom number and coordinates (in angstroms).

    dict
        Connectivity.

    """
    coord = re.compile(r"""
            \w+\s+
            (?P<atid>\d+)\s+           # Atom id.
            (?P<atsb>\w+)\s+           # Atomic number.
            \w+\s+
            \d+\s+
            (?P<x>[+-]?\d+\.\d+)\s+    # Orthogonal coordinates for X.
            (?P<y>[+-]?\d+\.\d+)\s+    # Orthogonal coordinates for Y.
            (?P<z>[+-]?\d+\.\d+)\s+    # Orthogonal coordinates for Z.
            """, re.X)

    data = list()
    ndx_conect = dict()
    with open(file, 'r') as INPUT:
        for line in INPUT:
            if coord.match(line):
                m = coord.match(line)
                data.append(m.groupdict())

            if "CONECT" in line:
                """ ndx_conect"""
                line = line.split()
                if len(line) > 2:
                    ndx_conect[int(line[1]) - 1] = [int(i) - 1 for i in line[2:]]

    coord = pd.DataFrame(data)
    coord = coord.astype({
        'atid': np.int64,
        'x': np.float64,
        'y': np.float64,
        'z': np.float64})
    # coord = coord.set_index('atid')

    coord["mass"] = coord["atsb"].apply(lambda at: Elements[at]["mass"])
    coord["num"] = coord["atsb"].apply(lambda at: Elements[at]["num"])
    # This file has no partial charges .
    coord["charge"] = 0.0

    return coord, ndx_conect


def minImagenC(q1, q2, L):
    """Return the one-dimensional distance using the minimun image criterion."""
    dq = q2 - q1
    if dq > L * 0.5:
        dq -= L
    
    if dq <= -L * 0.5:
        dq += L
    
    return dq


class connectivity(nx.DiGraph):
    """Building a class connectivity from directed graphs."""

    def __init__(self):
        """Initiate with the superclass of graphs."""
        super().__init__()

    def get_connectivity(self, coord):
        """
        Build connectivity from coordinates using nodes like atoms.

        Add nodes using atoms and symbols and coordinates. The edges are bonds.
        """
        self.define_atoms(coord)

        pairs, m = self._neighboring_pairs(coord)
        for i, j in pairs:
            self.add_edge(i, j)
            self.add_edge(j, i)

            self.edges[i, j]['dist'] = m[i, j]
            self.edges[j, i]['dist'] = m[i, j]

    def _neighboring_pairs(self, coord, cutoff=1.7):
        """Return neighboring pairs."""
        xyz = coord.loc[:, ['x', 'y', 'z']].values.astype(np.float64)

        # compute distance
        m = cdist(xyz, xyz, 'euclidean')
        m = np.triu(m)

        # compute pairs
        indexs = np.where((m > 0.) & (m <= cutoff))
        pairs = map(lambda in0, in1: (in0, in1), indexs[0], indexs[1])

        return pairs, m

    @property
    def atoms_map(self):
        """Return a dict the indices as key and values to which it's bonded."""
        return {at: list(self.neighbors(at)) for at in self.nodes}

    def reset_nodes(self):
        """Reset le count of node from 0 to new sizes nodes."""
        mapping = {value: count for count, value in enumerate(self.nodes, start=0)}

        # return nx.relabel_nodes(self, mapping, copy=True)
        self = nx.relabel_nodes(self, mapping, copy=True)

    def nbonds(self, inode):
        """Return number of atoms in connected to iat."""
        return int(self.degree[inode] / 2)

    def at_connect(self, inode):
        """Return the atoms symbols conectec to i node."""
        return '-'.join(
            [self.nodes[a]['atsb'] for a in list(self.neighbors(inode))]
        )

    def get_df(self):
        """Return a coordinate dataframe from connectivity."""
        indexs = list(self.nodes)
        rows = list()

        for i in self.nodes:
            rows.append({
                'atsb': self.nodes[i]['atsb'],
                'x': self.nodes[i]['xyz'][0],
                'y': self.nodes[i]['xyz'][1],
                'z': self.nodes[i]['xyz'][2]
            })

        df = pd.DataFrame(rows, index=indexs)

        return df

    def define_atoms(self, coord):
        """Define the atoms of the system like nodes."""
        for i in coord.index:
            self.add_node(
                i,
                xyz=coord.loc[i, ['x', 'y', 'z']].values,
                atsb=coord.loc[i, 'atsb']
            )

    def read_dict(self, connect):
        """Read connectivity from a dictionary."""
        # Add edges like bonds
        for i in connect:
            for ai, aj in it.product([i], connect[i]):
                self.add_edge(ai, aj)
                self.add_edge(aj, ai)
                pos_i = self.nodes[ai]['xyz']
                pos_j = self.nodes[aj]['xyz']

                # save distance ij
                m = np.linalg.norm(pos_j - pos_i)

                self.edges[ai, aj]['dist'] = m
                self.edges[aj, ai]['dist'] = m

    def get_interactions_list(self):
        """List of interactions are generated."""
        all_length = dict(nx.algorithms.all_pairs_shortest_path_length(self))
        all_paths = []

        for s in all_length.keys():
            for e in all_length[s].keys():
                if all_length[s][e] == 1:
                    all_paths += list(
                        nx.algorithms.all_simple_paths(self, s, e, cutoff=1))
                elif all_length[s][e] == 2:
                    all_paths += list(
                        nx.algorithms.all_simple_paths(self, s, e, cutoff=2))
                elif all_length[s][e] == 3:
                    all_paths += list(
                        nx.algorithms.all_simple_paths(self, s, e, cutoff=3))

        # print(list(nx.simple_cycles(self)))
        # For cycles
        # cycles = [cycle for cycle in nx.recursive_simple_cycles(mol.connect) if len(cycle) > 2]
        # [mol.atoms[0].hyb for i in cycles[0]] hybridation
        # [mol.atoms[0].atsb for i in cycles[0]] atoms symbols
        # print(all_length)

        return all_paths

    @property
    def atomsMOL(self):
        """
        Indexes of atoms per molecule.

        Property that returns a dictionary with molecule indices as key and
        atom indices as values.

        """
        atoms_MOL = nx.weakly_connected_components(self)
        # Dict imol : Natoms, indexs
        bulk = dict()
        ipol = 0

        for mol in atoms_MOL:
            mol = list(sorted(mol))
            bulk[ipol] = dict()
            bulk[ipol]["Natoms"] = len(mol)
            bulk[ipol]["index"] = mol
            ipol += 1

        return bulk

    def sub_connect(self, atoms):
        """
        Extract the connectivity for a particular group of atoms.

        Parameters
        ----------
        atoms : list
            List with the indices of the atoms.
        """
        return nx.subgraph(self, atoms)

    @property
    def connection_paths(self):
        """
        Atomic interconnection paths.

        Used for a molecule, returns a dictionary with the connection pathways.
        """
        return dict(nx.shortest_path_length(self))

    @property
    def spine_atoms(self):
        """
        Longest path of connection between atoms.

        This connection constitutes the backbone of the molecule, it is used to
        reconstruct a molecule broken by periodic boundary conditions.
        """
        paths = self.connection_paths

        length = 0
        source = 0
        target = 0

        for i in paths:
            for c in paths[i]:
                # i : atom
                # c : bond to
                # pth[i][c] : length
                if paths[i][c] > length:
                    length = paths[i][c]
                    source = i
                    target = c

        return nx.shortest_path(self, source, target)

    def update_coordinates(self, coord):
        """
        Update atomic coordinates.

        At the moment it is made to work with DataFrames. This is because in
        the DataFrame the indices must correspond to the connectivity atoms.
        """
        if type(coord) == np.ndarray:
            raise Exception("Be careful! you must use a DataFrame")

        for n in self.nodes:
            try:
                self.nodes[n]["xyz"] = coord.loc[n, ["x", "y", "z"]].values.astype(np.float64)
            except KeyError:
                pass

    def noPBC(self, box):
        """
        Reconstruct a molecule that is split due to PBC.

        At the moment it is used in one molecule.

        Parameters
        ----------
        box : numpy.array (3x1)
            Vector with the length of the box.
        """
        spine_atoms = self.spine_atoms
        ref_atoms = []
        
        # Fisrt atoms in spine.
        # The molecule is rebuilt from the longest atom bond.
        for at1 in spine_atoms:
            for at2 in self[at1]:
                if at2 not in ref_atoms and at2 in spine_atoms:
                    r1 = self.nodes[at1]["xyz"]
                    r2 = self.nodes[at2]["xyz"]

                    # Assuming a system with 3D dimensions
                    nr2 = np.zeros(3)
                    for i in range(3):
                        nr2[i] = minImagenC(r1[i], r2[i], box[i]) + r1[i]
                    nd12 = distance(r1, nr2)
                    self.nodes[at2]["xyz"] = nr2
                    self.edges[at1, at2]['dist'] = nd12
                    self.edges[at2, at1]['dist'] = nd12

            ref_atoms.append(at1)

        # Then the rest of the atoms.
        # Using the longest bond of atoms, the complete molecule is
        # reconstructed.
        for at1 in self.nodes():
            if at1 not in ref_atoms:
                for at2 in self[at1]:
                    r1 = self.nodes[at1]["xyz"]
                    r2 = self.nodes[at2]["xyz"]
                    
                    if at2 in ref_atoms:
                        nr1 = np.zeros(3)
                        for i in range(3):
                            nr1[i] = minImagenC(r2[i], r1[i], box[i]) + r2[i]
                        nd12 = distance(r2, nr1)
                        self.nodes[at1]["xyz"] = nr1
                    else:
                        nr2 = np.zeros(3)
                        for i in range(3):
                            nr2[i] = minImagenC(r1[i], r2[i], box[i]) + r1[i]
                        nd12 = distance(r1, nr2)
                        self.nodes[at2]["xyz"] = nr2

                    self.edges[at1, at2]['dist'] = nd12
                    self.edges[at2, at1]['dist'] = nd12
                    
            else:
                continue


class MOL:
    """
    Class used to represent a molecule.

    Attributes
    ----------
    dfatoms : DataFrame
        Table with all the information of the atoms of the molecule, symbol,
        mass and x and z coordinates.

    Methods
    -------
    load_file : str
        Reads a molecular file.


    """

    # Atoms
    dfatoms = pd.DataFrame()
    atoms = list()

    # Connectivity
    connect = connectivity()

    # Bonds
    bonds_list = []
    dfbonds = pd.DataFrame()

    # Angles
    angles_list = []
    dfangles = pd.DataFrame()

    # Dihedrals
    dihedrals_list = []
    dfdih = pd.DataFrame()

    def __init__(self, file=None, **kwargs):
        """
        Initialize the molecule.

        Creates a dictionary that indicates that information about the molecule
        is loaded.

        If a file is indicated the atomic coordinates will be loaded.

        Parameters
        ----------
        file : str
            File path.

        """
        resume = {
            "dfatoms": False
        }

        self.resume = resume
        self.atoms_count = {"C": 0, "N": 0, "H": 0}

        if file:
            self.load_file(file, **kwargs)

    def load_file(self, file, res=None, connectivity=True):
        """Extract information from file."""
        connect = None
        if file.endswith("xyz"):
            MOL.dfatoms = load_xyz(file)
        elif file.endswith("mol2"):
            MOL.dfatoms = load_mol2(file)
        elif file.endswith("pdb"):
            MOL.dfatoms, connect = load_pdb(file)
        else:
            raise MoleculeDefintionError(0)

        # RES name from file name
        if res:
            self.res
        else:
            self.res = file.split('/')[-1].split('.')[0].upper()

        # Count the atoms present.
        self.atoms_count.update(dict(MOL.dfatoms["atsb"].value_counts()))

        # Search connectivity
        if connectivity:
            if not connect:
                self._connectivity()
            else:
                self.connect.define_atoms(MOL.dfatoms)
                self.connect.read_dict(connect)

            # Atoms information
            self._get_atoms_info()

            # Intramolecular information
            self._intramol_list()

    @property
    def DBE(self):
        """Return the `DBE` (Double Bond Equivalent)."""
        atoms = self.atoms_count
        return int((2 * atoms["C"] + atoms["N"] + 2 - atoms["H"]) / 2)

    @property
    def formule(self):
        """Return the molecular formule."""
        formule = ""
        atoms = self.atoms_count
        for at in sorted(atoms):
            if atoms[at] > 0:
                if atoms[at] == 1:
                    formule += at

                else:
                    formule += at
                    formule += str(atoms[at])

        return formule

    def _connectivity(self):
        """Generate the connectivity object of the molecule."""
        MOL.connect.get_connectivity(MOL.dfatoms)

    def search_connectivity(self):
        """Search connectivity."""
        if len(MOL.dfatoms) > 0:
            self._connectivity()
            self._get_atoms_info()
            self._intramol_list()
        else:
            raise MoleculeDefintionError(1)

    def _get_atoms_info(self):
        """Generate a composite list of class atom."""
        for i in MOL.dfatoms.index:
            at = MOL.dfatoms.loc[i, "atsb"]
            atom = ATOM(at, i)
            MOL.atoms.append(atom)

    def _intramol_list(self):
        """Get intramolecular interactions."""
        all_paths = self.connect.get_interactions_list()

        # BONDS, list, types
        bonds_list = [tuple(p) for p in all_paths if len(set(p)) == 2]
        for iat, jat in bonds_list:
            if iat < jat:
                MOL.bonds_list.append((iat, jat))

        # ANGLES, list, types
        angles_list = [tuple(p) for p in all_paths if len(set(p)) == 3]
        for iat, jat, kat in angles_list:
            if iat < kat:
                MOL.angles_list.append((iat, jat, kat))

        # DIHEDRALS, list, types
        dihedrals_list = [tuple(p) for p in all_paths if len(set(p)) == 4]
        for iat, jat, kat, lat in dihedrals_list:
            if iat < lat:
                # Remove dihedrals -ca-ca-
                # if MOL.dftypes.loc[jat, "type"] != "ca" and MOL.dftypes.loc[kat, "type"] != "ca":
                #    MOL.dihedrals_list.append((iat, jat, kat, lat))
                MOL.dihedrals_list.append((iat, jat, kat, lat))


class ATOM(MOL):
    """Subclass to represent an atom of the parent class `MOL`."""

    def __init__(self, at, n):
        """Initialize with the atomic symbol and the index in the molecule."""
        self.atsb = at
        self.n = n

    @property
    def atoms_connect(self):
        """
        Connect atom.

        Returns a string with the symbols of the connected atoms and their
        hybridization.
        """
        hyb_atoms_connect = ""

        if self.atsb == "H":
            # HYDROGEN
            at_i = list(MOL.connect.neighbors(self.n))[0]
            sb = MOL.connect.nodes[at_i]["atsb"]
            nbonds = len(list(MOL.connect.neighbors(at_i)))

            if sb == 'C' and nbonds == 3:
                """ H, aromatic hydrogen """
                hyb_atoms_connect += "Csp2"

            elif sb == 'C' and nbonds == 4:
                """ H, aliphatic hydrogen """
                hyb_atoms_connect += "Csp3"

        elif self.atsb in ["C", "N"]:
            # CARBON
            # atoms bonded to carbon
            for at_i in MOL.connect.neighbors(self.n):
                sb = MOL.connect.nodes[at_i]["atsb"]
                nbonds = len(list(MOL.connect.neighbors(at_i)))

                if sb == "C" and nbonds == 3:
                    hyb_atoms_connect += "Csp2"

                elif sb == "C" and nbonds == 4:
                    hyb_atoms_connect += "Csp3"

                elif sb == "N" and nbonds == 2:
                    hyb_atoms_connect += "Nsp2"

                elif sb == "H":
                    hyb_atoms_connect = "H" + hyb_atoms_connect

        else:
            raise MoleculeDefintionError(2).format(
                MOL.dfatoms.loc[self.n, "atsb"])

        return hyb_atoms_connect

    @property
    def hyb(self):
        """Return of atomic hybridization."""
        atms_hyb = {
            "H": {1: "s"},
            "C": {4: "sp3", 3: "sp2"},
            "N": {2: "sp2", 3: "sp3"}
        }
        try:
            return atms_hyb[self.atsb][len(self.connect[self.n])]
        except KeyError:
            raise MoleculeDefintionError(3).format(
                self.n, self.atsb)


def distance(v1, v2):
    """
    Compute the distance between two vectors.

    Parameters
    ----------
    v1,v2 : numpy.array
         Vector in space.
    """
    return np.linalg.norm(v1 - v2)


def pbcboxs():
    """Return a generator with periodical vectors."""
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            for k in [-1, 0, 1]:
                yield (i, j, k)


"""Functions for saving information."""


def save_xyz(coord, name='file'):
    """
    Save an xyz file of coordinates.

    Parameters:
    -----------
    coord : DataFrame

    name : str

    """
    nat = len(coord)
    xyz = "%s.xyz" % name
    lines = ''
    lines += '%d\n' % nat
    lines += '%s\n' % name
    for i in coord.index:
        line = (coord.atsb[i], coord.x[i], coord.y[i], coord.z[i])
        lines += '%3s%8.3f%8.3f%8.3f\n' % line
    # writing all
    with open(xyz, "w") as f:
        f.write(lines)
    print(f'Name of xyz file: {xyz}')
