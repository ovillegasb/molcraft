#!/bin/env python
# -*- coding: utf-8 -*-

"""Register MolCraft functions to implement your tools from command line."""

import argparse

TITLE = """\033[1;36m
  __  __       _  _____            __ _
 |  \\/  |     | |/ ____|          / _| |
 | \\  / | ___ | | |     _ __ __ _| |_| |_
 | |\\/| |/ _ \\| | |    | '__/ _` |  _| __|
 | |  | | (_) | | |____| | | (_| | | | |_
 |_|  |_|\\___/|_|\\_____|_|  \\__,_|_|  \\__|
\033[m
Python module used to manage molecular systems for molecular modeling and calculations.

Author: Orlando VILLEGAS
Date: 2024-03-21
"""


def options():
    """Generate command line interface."""
    parser = argparse.ArgumentParser(
        prog="molcraft",
        usage="%(prog)s [-options]",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Enjoy the program!"
    )

    return vars(parser.parse_args())

#===TEST ZONE
from rdkit import Chem
from rdkit.Chem import AllChem

#===
def main():
    """Run main function."""
    print(TITLE)
    args = options()
    print(args)
    m = Chem.MolFromSmiles('Cc1ccccc1')
    print(m.GetNumAtoms())
    print(Chem.MolToSmiles(m))
    print(Chem.MolToMolBlock(m))
    m_H = Chem.AddHs(m)
    print(Chem.MolToMolBlock(m_H))
    AllChem.EmbedMolecule(m_H, randomSeed=0xf00d)   # optional random seed for reproducibility)
    print(Chem.MolToMolBlock(m_H))

    print(Chem.MolToXYZBlock(m_H))
    xyz = Chem.MolToXYZBlock(m_H)

    with open("test.xyz", "w") as F:
        F.write(xyz)


if __name__ == '__main__':
    main()
