from abc import ABC, abstractmethod
import fenics as fe
# import solid_mechanics.kinnematics as kin
from . import kinnematics as kin


class ConstitutiveModelBase(ABC):

    def __init__(self, parameters: dict, history_terms: dict = dict({})):
        self._parameters = parameters
        self._history_terms = history_terms

    @abstractmethod
    def strain_energy(self, u):
        raise NotImplementedError

    def stress(self, u):
        u_var = fe.variable(u)
        C = fe.variable(kin.right_cauchy_green(u_var))
        return 2 * fe.diff(self.strain_energy(u_var), C)

    # @abstractmethod
    # def tangent(self):
    #     raise NotImplementedError

class IsoConstitutiveModelBase(ConstitutiveModelBase):
    pass

class IsoStVenant(ConstitutiveModelBase):
    """Blah blah

        :math:`\\mathbb{A}\\bm{a}`

        .. math:: \\bm{a} = \\bm{b} + \\bm{c}
           :label: a

        .. math:: \\bm{c} = \\begin{bmatrix} 1 & 2 \\\\ \
        2 & 2 \\\\ \
        3 & 2 \\\\ \
        \\end{bmatrix}
           :label: c



    """
    def __init__(self, parameters):
        super().__init__(parameters)

    def strain_energy(self, u):
        pass

    def stress(self, u):
        pass

    def iso_stress(self, u):
        E = kin.iso_green_lagrange_strain(u)

        return 2 * self._parameters['mu'] * E

    def p(self, u):
        return None



class StVenant(ConstitutiveModelBase):
    """Blah blah

        :math:`\\mathbb{A}\\bm{a}`

        .. math:: \\bm{a} = \\bm{b} + \\bm{c}
           :label: a

        .. math:: \\bm{c} = \\begin{bmatrix} 1 & 2 \\\\ \
        2 & 2 \\\\ \
        3 & 2 \\\\ \
        \\end{bmatrix}
           :label: c



    """
    def __init__(self, parameters):
        super().__init__(parameters)

    def strain_energy(self, u):
        pass

    def stress(self, u):
        E = kin.green_lagrange_strain(u)
        I = kin.identity(u)

        return self._parameters['lambda'] * fe.tr(E) * I + 2 * self._parameters['mu'] * E


# class NeoHookean(ConstitutiveModelBase):
#
#     def __init__(self, parameters):
#         super().__init__(parameters)
#
#     def strain_energy(self, u):
#         C = kin.right_cauchy_green(u)
#         I1 = fe.tr(C)
#         return self._parameters['mu'] * (I1 - fe.Constant(3.))
#
#     def stress(self, u):
#         E = kin.green_lagrange_strain(u)
#         I = kin.identity(u)
#
#         return self._parameters['lambda'] * fe.tr(E) * I + 2 * self._parameters['mu'] * E

