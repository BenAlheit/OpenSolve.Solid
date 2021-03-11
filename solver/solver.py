from .step import Step


class SolidMechanicsSolver(object):

    def __init__(self, steps: [Step]):

        self._steps = steps
        self._current_step = 0

    def solve(self):
        for step in self._steps:
            step.run()
