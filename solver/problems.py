import fenics as fe
from ..solid_mechanics.constiutive_model import ConstitutiveModelBase
from ..solid_mechanics import kinnematics as kin
from abc import ABC, abstractmethod


class StationaryProblem(ABC):

    def __init__(self,
                 mesh: fe.Mesh,
                 density: fe.Expression,
                 constitutive_model: ConstitutiveModelBase,
                 bf: fe.Expression = fe.Expression('0', degree=0)):
        self._mesh = mesh
        self._density = density
        self._constitutive_model = constitutive_model
        self.bf = bf

    @property
    def density(self):
        return self._density

    @property
    def constitutive_model(self):
        return self._constitutive_model

    @abstractmethod
    def update_values(self):
        raise NotImplementedError


class UStationaryProblem(StationaryProblem):

    def __init__(self,
                 mesh: fe.Mesh,
                 density: fe.Expression,
                 constitutive_model: ConstitutiveModelBase,
                 bf: fe.Expression = fe.Expression('0', degree=0)):

        super().__init__(mesh, density, constitutive_model, bf)

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


class UPStationaryProblem(object):

    def __init__(self,
                 mesh: fe.Mesh,
                 density: fe.Expression,
                 constitutive_model: ConstitutiveModelBase,
                 bf: fe.Expression = fe.Expression('0', degree=0)):

        self._mesh = mesh
        self._density = density
        self._constitutive_model = constitutive_model
        self.bf = bf

        element_v = fe.VectorElement("P", mesh.ufl_cell(), 1)
        element_s = fe.FiniteElement("P", mesh.ufl_cell(), 1)
        mixed_element = fe.MixedElement([element_v, element_v, element_s])
        W = fe.FunctionSpace(mesh, mixed_element)

        # Unknowns, values at previous step and test functions
        w = fe.Function(W)
        self.u, self.v, self.p = fe.split(w)

        w0 = fe.Function(W)
        self.u0, self.v0, self.p0 = fe.split(w0)
        self.a0 = fe.Function(fe.FunctionSpace(mesh, element_v))

        self.ut, self.vt, self.pt = fe.TestFunctions(W)

        self.F = kin.def_grad(self.u)
        self.F0 = kin.def_grad(self.u0)

    @property
    def density(self):
        return self._density

    @property
    def constitutive_model(self):
        return self._constitutive_model


# class StationaryProblem(ABC):
#
#     def __init__(self,
#                  mesh: fe.Mesh,
#                  density: fe.Expression,
#                  constitutive_model: ConstitutiveModelBase,
#                  bf: fe.Expression = fe.Expression('0', degree=0)):
#         self._mesh = mesh
#         self._density = density
#         self._constitutive_model = constitutive_model
#         self.bf = bf
#
#         element_v = fe.VectorElement("P", mesh.ufl_cell(), 1)
#         element_s = fe.FiniteElement("P", mesh.ufl_cell(), 1)
#         mixed_element = fe.MixedElement([element_v, element_v, element_s])
#         W = fe.FunctionSpace(mesh, mixed_element)
#
#         # Unknowns, values at previous step and test functions
#         self.w = fe.Function(W)
#         self.u, self.v, self.p = fe.split(self.w)
#
#         w0 = fe.Function(W)
#         self.u0, self.v0, self.p0 = fe.split(w0)
#         self.a0 = fe.Function(fe.FunctionSpace(mesh, element_v))
#
#         self.ut, self.vt, self.pt = fe.TestFunctions(W)
#
#         self.F = kin.F(self.u)
#         self.F0 = kin.F(self.u0)
#
#     @property
#     def density(self):
#         return self._density
#
#     @property
#     def constitutive_model(self):
#         return self._constitutive_model
#
#     @abstractmethod
#     def update_values(self):
#         raise NotImplementedError
