"""
File that lists all exceptions and errors that can occur using MolCraft.

    MoleculeDefinitionError:
        Exceptions found during the work with molecules.

"""


class MoleculeDefintionError(Exception):
    """Define errors related to the definition of molecules."""

    def __init__(self, error_number):
        """Initialize the class by assinging a message."""
        # types of files sopported
        typesfiles = ["xyz", "pdb", "mol2"]

        # Defining error messages
        errorMessages = {
            0: """
            \033[1;31mMolecule format is not recognized,
            files supported to date: {}
            \033[m""".format(", ".join(typesfiles)),
            1: """
            \033[1;31mNo structures have been loaded yet.
            \033[m""",
            2: """
            \033[1;31mATOM is not configured to work with the atom: {}
            \033[m""",
            3: """
            \033[1;31mHybridation for {} - {} not found"
            \033[m""",
        }
        self.message = errorMessages[error_number]
        super().__init__(self.message)

    def __str__(self):
        """Return message."""
        return self.message
