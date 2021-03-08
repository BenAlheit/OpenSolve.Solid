import fenics as fe
from .step import ImplicitStep
from ..solid_mechanics.constiutive_model import ConstitutiveModelBase
from ..solid_mechanics import kinnematics as kin


class SolidMechanicsSolver(object):

    def __init__(self,
                 mesh: fe.Mesh,
                 density: fe.Expression,
                 constitutive_model: ConstitutiveModelBase,
                 steps: [ImplicitStep]):

        self._mesh = mesh
        self._density = density
        self._constitutive_model = constitutive_model
        self._steps = steps

        self._current_step = 0

        # Build function space
        element_v = fe.VectorElement("P", mesh.ufl_cell(), 1)
        element_s = fe.FiniteElement("P", mesh.ufl_cell(), 1)
        mixed_element = fe.MixedElement([element_v, element_v, element_s])
        W = fe.FunctionSpace(mesh, mixed_element)

        # Unknowns, values at previous step and test functions
        w = fe.Function(W)
        self._u, self._v, self._p = fe.split(w)
        w0 = fe.Function(W)
        self._u0, self._v0, self._p0 = fe.split(w0)
        self._ut, self._vt, self._pt = fe.TestFunctions(W)

        self._F = kin.F(self._u)
        self._F0 = kin.F(self._u0)

        self._functionals = []

    def create_step_functionals(self):
        self._functionals = []
        step = self._steps[self._current_step]
        a = fe.inner(self._u, self._ut) + \
            fe.inner(kin.F(self._u) * self._constitutive_model.stress(self._F), fe.grad(self._ut)) \
            * pow(step.dt, 2) * step.beta / self._density

        L =

    def apply_step(self):
        pass