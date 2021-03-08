from abc import ABC, abstractmethod


class ConstitutiveModelBase(ABC):

    def __init__(self, parameters: dict, history_terms: dict = dict({})):
        self._parameters = parameters
        self._history_terms = history_terms

    @abstractmethod
    def strain_energy(self):
        raise NotImplementedError

    @abstractmethod
    def stress(self, F):
        #TODO: Type F correctly
        raise NotImplementedError

    @abstractmethod
    def tangent(self):
        raise NotImplementedError
