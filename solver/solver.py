import fenics as fe
from .problems import StationaryProblem
from .step import Step
from ..solid_mechanics.constiutive_model import ConstitutiveModelBase
from ..solid_mechanics import kinnematics as kin


class SolidMechanicsSolver(object):

    def __init__(self, steps: [Step]):

        self._steps = steps
        self._current_step = 0

    def solve(self):
        for step in self._steps:
            step.run()
