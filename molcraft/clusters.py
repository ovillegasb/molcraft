"""Module dedicated to the analysis of molecular agglomerates."""

import numpy as np
from scipy.constants import N_A as avogadro
from scipy.spatial.distance import cdist


class GyrationTensor:
    """Class to represent a Gyration Tensor and compute related quantities."""

    def __init__(self, coords, masses, box, pbc=True):
        """
        Initialize calling coordinates, masses and box dimnetion.

        Arguments:
        ----------
            matrix (3x3): 3x3 matrix of the gyration tensor
        """
        self.coords = coords

        self.masses = masses
        # Assuming cubic box
        self.box = box

        self._pbc = pbc

        # Traslation using center of mass like reference poit

        newcoords = list()
        center_of_mass = self.center_of_mass

        if pbc:
            for i in coords:
                r_i = list()
                for j in range(len(i)):
                    if np.abs(i[j] - center_of_mass[j]) < self.box[j] / 2:
                        # x position
                        r_i.append(i[j])
                    elif np.abs(i[j] - self.box[j] - center_of_mass[j]) < self.box[j] / 2:
                        # y position
                        r_i.append(i[j] - self.box[j])
                    elif np.abs(i[j] + self.box[j] - center_of_mass[j]) < self.box[j] / 2:
                        # z position
                        r_i.append(i[j] + self.box[j])
                newcoords.append(r_i)
            newcoords = np.array(newcoords)
            self.coords = newcoords

        # weighted coords
        self.wcoords = self.coords * self.masses[:, np.newaxis]

    @property
    def center_of_mass(self):
        """Compute the center of mass, the mass weighterd barycenter."""
        if self._pbc:
            theta_i = self.coords / self.box * 2 * np.pi
            xi_i = np.cos(theta_i)
            eta_i = np.sin(theta_i)
            xi_m = np.sum(xi_i * self.masses[:, np.newaxis], axis=0) / self.masses.sum()
            eta_m = np.sum(eta_i * self.masses[:, np.newaxis], axis=0) / self.masses.sum()
            theta_m = np.arctan2(-eta_m, -xi_m) + np.pi
            return self.box * theta_m / 2 / np.pi

        else:
            return np.sum(self.coords * self.masses[:, np.newaxis], axis=0) / self.masses.sum()

    @property
    def matrix(self):
        r"""
        Compute the gyration tensor.

        The diagonal terms are:

        1 / N \sum_i (x_i - x_com)^2

        for x, y and z, with com meaning center of mass and N the number of
        atoms.

        The off diagonal terms are:

        1 / N \sum_i (x_i - x_com) (y_i - y_com)
        """
        # mass weighted deviation with respect to the center of mass
        # x_i - x_com
        deviation = self.coords - self.center_of_mass
        # \sum_i (x_i - x_com)^2
        deviation2 = np.sum(deviation**2, axis=0)

        # off diagonal terms
        xy = np.dot(deviation[:, 0], deviation[:, 1])
        xz = np.dot(deviation[:, 0], deviation[:, 2])
        yz = np.dot(deviation[:, 1], deviation[:, 2])

        S = np.array([[deviation2[0], xy,            xz],
                      [xy,            deviation2[1], yz],
                      [xz,            yz,            deviation2[2]]])
        S /= self.masses.size

        return S

    @property
    def iso_rg(self):
        r"""
        Compute the radius of gyration.

        R_g assuming an isotrop system as:

        R_g = \sqrt{1 / N \sum_i (r_i - r_com)^2}
        """
        natom = self.masses.size
        Rg2 = 1 / natom * np.sum((self.coords - self.center_of_mass)**2)

        return np.sqrt(Rg2)

    @property
    def iso_w_rg(self):
        r"""
        Compute the radius of gyration, R_g.

        assuming an isotrop system as:

        R_g = \sqrt{1 / N \sum_i (r_i - r_com)^2}
        """
        dr2 = (self.coords - self.center_of_mass)**2 * self.masses[:, np.newaxis]
        Rg2 = np.sum(dr2) / self.masses.sum()

        return np.sqrt(Rg2)

    @property
    def principal_moments(self):
        """Return the sorted eigenvalue of the gyration tensor in decreasing order."""
        eigval, _ = np.linalg.eig(self.matrix)
        eigval = np.flip(np.sort(eigval))
        return eigval

    @property
    def rg(self):
        r"""
        Compute the radius of gyration from the eigenvalues of the gyration
        tensor from:

        R_g = \sqrt{\lambda_1 + \lambda_2 + \lambda_3}

        where the \lambda_i are the eigenvaleus of the gyration tensor.
        """
        return np.sqrt(np.sum(self.principal_moments))

    @property
    def shape_anisotropy(self):
        """
        Compute the shape anisotropy from:

        k2 = 1 - 3 (lambda_1 lambda_2 + lambda2 lambda3 + lambda3 lambda1) / (lambda1 + lambda2 + lambda3)^2
        """
        p1, p2, p3 = self.principal_moments
        return 1 - 3 * (p1 * p2 + p2 * p3 + p3 * p1) / (p1 + p2 + p3)**2

    @property
    def volume(self):
        r"""
        Compute the volume of an ellipsoid that would have its principal
        moments a, b and c equals to \sqrt(5 \lambda_i).
        """
        return 4 / 3 * np.pi * np.sqrt(5 ** 3 * self.principal_moments.prod())

    @property
    def density(self):
        """
        Compute the density assuming an effective ellipsoidal volume. The volume
        is assumed to be in nm^3 and the mass in g.mol-1. Thus the density is
        computed in kg.m-3
        """
        rho = np.sum(self.masses) / self.volume / avogadro / 1e-27 * 1e-3
        return rho

    @property
    def total_mass(self):
        """
        Compute the mass of aggregate in g.mol-1.
        """
        tmass = np.sum(self.masses)
        return tmass

    @property
    def max_distance(self):
        """
        Return the longest distance between an atom and the center of mass.
        """
        # Symmetrical distance matrix
        m = cdist(self.coords, self.coords, 'euclidean')

        # upper triangle matrix
        m = np.triu(m)
        # distances = np.sum((self.coords - self.center_of_mass)**2, axis=1)
        # return np.sqrt(np.max(distances))
        return np.max(m)

    @staticmethod
    def get_data_header():
        """
        Return an header according to the data returned by get_data
        """
        lines = "# Gyration tensor calculations.\n"
        lines += "# distance units depends on the units of the trajectory.\n"
        lines += "# (L) is the distance unit\n"
        lines += "# column 1: Rg ([L])\n"
        lines += "# column 2: Rg isotrop ([L])\n"
        lines += "# column 3: Rg isotrop mass weighted ([L])\n"
        lines += "# column 4: k^2, shape anisotropy (au) \n"
        lines += "# column 5: volume ([L]^3)\n"
        lines += "# column 5: molar mass ([g/mol])\n"
        lines += "# column 6: largest distance between center of mass and atoms ([L])\n"
        lines += "# column 7: density (g.L-1) grams per litters if [L] = nm.\n"
        # lines += "# column 8: Dipolar moment [D].\n"
        return lines

    def get_data(self):
        """
        Returns all data computed from the gyration tensor with 10.4f format.
        """
        lines = f"{self.rg:10.4f}"
        lines += f"{self.iso_rg:10.4f}"
        lines += f"{self.iso_w_rg:10.4f}"
        lines += f"{self.shape_anisotropy:10.4f}"
        lines += f"{self.volume:10.4f}"
        lines += f"{self.total_mass:12.4f}"
        lines += f"{self.max_distance:10.4f}"
        lines += f"{self.density:12.4e}"
        # lines += f"{self.mu:12.4e}"

        return lines
