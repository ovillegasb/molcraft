"""MolCraft submodule containing functions to efficiently transform molecular formats."""


# from rdkit import Chem
from rdkit.Chem import AllChem


def smi2xyz(smi):
    """Convert smile format to xyz format."""
    m = AllChem.MolFromSmiles(smi)
    m = AllChem.AddHs(m)
    _ = AllChem.EmbedMolecule(m)
    _ = AllChem.UFFOptimizeMolecule(m)

    return AllChem.MolToXYZBlock(m)
