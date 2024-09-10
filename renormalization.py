import numpy as np
from scipy.misc import derivative

class RenormalizationGroup:

    def __init__(self, dimension):

        self.b = 2 # Decimation
        self.d = dimension

        # Set Migdal-Kadanoff bond-moving multiplier
        if dimension == 1:
            self.m = 1
            
        elif dimension >= 2:
            self.m = self.b ** (dimension - 1)
            
        else:
            print("Dimension should be a positive integer.")

    def _pderivative(self, function, variable=0, point=[]):
        arguments = point[:]
        def wraps(x):
            arguments[variable] = x
            return function(*arguments)
        return derivative(wraps, point[variable], dx=np.sqrt(np.finfo(float).eps))

    def _lnR1(self, j, h):

        e1 = 2*j + 4*h
        e2 = -2*j
        emax = np.amax([e1, e2])

        return emax + np.log(np.exp(e1 - emax) + np.exp(e2 - emax))

    def _lnR2(self, j, h):

        e1 = -2*j
        e2 = 2*j - 4*h
        emax = np.amax([e1, e2])

        return emax + np.log(np.exp(e1 - emax) + np.exp(e2 - emax))

    def _lnR3(self, j, h):

        e1 = 2*h
        e2 = -2*h
        emax = np.amax([e1, e2])

        return emax + np.log(np.exp(e1 - emax) + np.exp(e2 - emax))

    def J(self, interaction, field):
        j = self.m * interaction
        h = self.m * field
        return (1/4) * (self._lnR1(j, h) + self._lnR2(j, h) - 2*self._lnR3(j, h))

    def H(self, interaction, field):
        j = self.m * interaction
        h = self.m * field
        return (1/4) * (self._lnR1(j, h) - self._lnR2(j, h))

    def G(self, interaction, field):
        j = self.m * interaction
        h = self.m * field
        return (1/4) * (self._lnR1(j, h) + self._lnR2(j, h) + 2*self._lnR3(j, h))

    def _recursion_matrix(self, j, h):

        eigen = self.b ** self.d

        X = self._pderivative(self.G, 0, [j, h])
        Y = self._pderivative(self.J, 0, [j, h])
        Z = self._pderivative(self.H, 0, [j, h])

        A = self._pderivative(self.G, 1, [j, h])
        B = self._pderivative(self.J, 1, [j, h])
        C = self._pderivative(self.H, 1, [j, h])

        return np.array([[eigen, X,   A],
                         [0,     Y,   B],
                         [0,     Z,   C]])

    def flow(self, interaction, field, n, output=False):

        j, h = interaction, field

        if output:
            output = [[j, h]]
        else:
            print("k,   J         H\n------------------")
            print(0, "  ", j, "    ", h)

        for i in range(n):

            j, h = self.J(j, h), self.H(j, h)
            if output:
                output.append([j, h])
            else:
                print(i+1, "  ", j, "    ", h)

        if output:
            return np.array(output)

    def critical_point(self, decimal_precision=5):

        if self.d == 1:
            print("There is no phase transition in 1D.")
            return None
        else:
            pass

        p = 0.01
        jc = 0.02

        for n in range(decimal_precision + 1):
            jc = jc - p
            p = 1 / 10**(n + 1)

            while True:
                jx = jc
                jx = self.J(jx, 0)
                if jx < jc:
                    jc = jc + p
                else:
                    break

        return round(jc, decimal_precision)

    def densities(self, field=0):
    # Densities under fixed H

            temp_list = []
            M_results = []

            j_initial = 0.01

            while j_initial < 30:

                h = field
                j = j_initial
                temp_list.append(1 / j)

                if h > 0:
                    Mn = [1, 0, 1]

                if h < 0:
                    Mn = [1, 0, -1]

                if h == 0:
                    Mn = [1, 0, 0]

                U = np.identity(3)
                for i in range(15):
                    U = (self.b ** (-self.d)) * np.dot(self._recursion_matrix(j, h), U)
                    j, h = self.J(j, h), self.H(j, h)

                M = np.dot(Mn, U)

                M_results.append(M)

                j_initial = j_initial + 0.01

            return np.array(temp_list), np.array(M_results)