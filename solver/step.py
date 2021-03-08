import fenics as fe
from abc import ABC


# TODO: Think about if dictionary is best for bcs


class Step(ABC):

    def __init__(self, dt0: fe.Expression, bcs: dict):
        self._dt = dt0
        self._bcs = bcs

    @property
    def dt(self):
        return self._dt

    @property
    def bcs(self):
        return self._bcs


class ImplicitStep(Step):

    def __init__(self,
                 dt0: fe.Expression,
                 bcs: dict,
                 alpha: fe.Constant,
                 u0: fe.Expression = None,
                 v0: fe.Expression = None,
                 a0: fe.Expression = None):

        super().__init__(dt0, bcs)

        self._alpha = alpha
        self._beta = (1 - self._alpha) ** 2 / 4.
        self._gamma = 1. / 2. - self._alpha

    @property
    def beta(self):
        return self._beta

    @property
    def gamma(self):
        return self._gamma
