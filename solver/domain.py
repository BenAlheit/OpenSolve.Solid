import fenics as fe
from solid_mechanics.constiutive_model import ConstitutiveModelBase
from abc import abstractmethod, ABC
from solid_mechanics import kinnematics as kin

class Domain(ABC):
    def __init__(self,
                 mesh: fe.Mesh,
                 constitutive_model: ConstitutiveModelBase,
                 function_space: fe.FunctionSpace,
                 bf: fe.Expression = fe.Expression('0', degree=0)):

        self._mesh = mesh
        self._constitutive_model = constitutive_model
        self.V = function_space
        self._bf = bf

    @property
    def constitutive_model(self):
        return self._constitutive_model

    @abstractmethod
    def update_values(self):
        raise NotImplementedError


class UDomain(Domain):

    def __init__(self,
                 mesh: fe.Mesh,
                 constitutive_model: ConstitutiveModelBase,
                 density: fe.Expression = fe.Expression('0', degree=0),
                 bf: fe.Expression = fe.Expression('0', degree=0)):

        self._density = density

        W = fe.VectorFunctionSpace(mesh, "P", 1)

        # Unknowns, values at previous step and test functions
        self.w = fe.Function(W)
        self.u, self.u0 = self.w, fe.Function(W)
        self.v, self.v0 = fe.Function(W), fe.Function(W)
        self.a, self.a0 = fe.Function(W), fe.Function(W)

        # self.a0 = fe.Function(fe.FunctionSpace(mesh, element_v))

        self.ut = fe.TestFunction(W)

        self.F = kin.def_grad(self.u)
        self.F0 = kin.def_grad(self.u0)

        super().__init__(mesh, constitutive_model, W, bf)

    def update_values(self):

        self.u0.assign(self.u)
        self.v0.assign(self.v)
        self.a0.assign(self.a)

        self.F0 = kin.def_grad(self.u0)

    @property
    def density(self):
        return self._density

    @property
    def constitutive_model(self):
        return self._constitutive_model
