from abc import ABC, abstractmethod
import fenics as fe
from . import kinnematics as kin


class ConstitutiveModelBase(ABC):

    def __init__(self, parameters: dict, history_terms: dict = dict({})):
        self._parameters = parameters
        self._history_terms = history_terms

    @abstractmethod
    def strain_energy(self, u):
        raise NotImplementedError

    def stress(self, u):
        C = kin.right_cauchy_green(u)
        return 2 * fe.diff(self.strain_energy(u), C)

    # @abstractmethod
    # def tangent(self):
    #     raise NotImplementedError


class NeoHookean(ConstitutiveModelBase):

    def __init__(self, parameters):
        super().__init__(parameters)

    def strain_energy(self, u):
        C = kin.right_cauchy_green(u)
        I1 = fe.tr(C)
        return self._parameters['mu'] * (I1 - fe.Constant(3.))
