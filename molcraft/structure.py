"""
The submodule structure contains the main molcraft classes.

It is created to work with input and output files. It is also used to extract
information from a molecular system. It defines the three main classes: BULK,
MOL and ATOM.

"""

import pandas as pd
import numpy as np
import networkx as nx
from scipy.spatial.distance import cdist


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
        Atomic coordinates information

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


class connectivity(nx.DiGraph):
    """Building a class connectivity from directed graphs."""

    def __init__(self):
        """Initiate with the superclass of graphs."""
        super().__init__()

    def get_connectivity(self, coord):
        """
        Build connectivity from coordinates using nodes like atoms.

        Add nodes using atoms andsymbols and coordinates. The edges are bonds.
        """
        for i in coord.index:
            self.add_node(
                i,
                xyz=coord.loc[i, ['x', 'y', 'z']].values,
                atsb=coord.loc[i, 'atsb']
            )

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

    # Connectivity
    connect = connectivity()

    def __init__(self, file=None):
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
            self.load_file(file)

    def load_file(self, file, res=None, connectivity=True):
        """Extract information from file."""
        if file.endswith("xyz"):
            MOL.dfatoms = load_xyz(file)

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
