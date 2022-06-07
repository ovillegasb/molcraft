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
    'N': {'mass': 14.0067, 'num': 7}
}


def load_xyz(file):
    """
    Read a file xyz.

    Parameters
    ----------
    file : str
        File path

    Returns
    -------
    DataFrame
        The element symbols, mass, atom number and coordinates (in angstroms).

    """
    coord = pd.read_csv(
        file,
        sep=r'\s+',
        skiprows=2,
        header=None,
        names=['atsb', 'x', 'y', 'z'],
        dtype={'x': np.float64, 'y': np.float64, 'z': np.float64}
    )

    coord["mass"] = coord["atsb"].apply(lambda at: Elements[at]["mass"])
    coord["num"] = coord["atsb"].apply(lambda at: Elements[at]["num"])
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
        if file.endswith("xyz"):
            MOL.dfatoms = load_xyz(file)
        elif file.endswith("mol2"):
            MOL.dfatoms = load_mol2(file)
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
            self._connectivity()

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
