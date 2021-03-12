import fenics as fe
from solid_mechanics.constiutive_model import ConstitutiveModelBase
from abc import abstractmethod, ABC
from solid_mechanics import kinnematics as kin
from enum import Enum, auto
from . import job_name


class Outputs(Enum):
    stress = auto()
    strain = auto()
    displacement = auto()


class Domain(ABC):
    DEFAULT_OUTPUTS: [Outputs] = [Outputs.stress,
                                  Outputs.strain,
                                  Outputs.displacement]

    def __init__(self,
                 mesh: fe.Mesh,
                 constitutive_model: ConstitutiveModelBase,
                 function_space: fe.FunctionSpace,
                 bf: fe.Expression = fe.Expression('0', degree=0),
                 outputs: [Outputs] = DEFAULT_OUTPUTS):
        self._mesh = mesh
        self._constitutive_model = constitutive_model
        self.V = function_space
        self.T = fe.TensorFunctionSpace(mesh, "P", 1)
        self.S = fe.FunctionSpace(mesh, "P", 1)

        self._bf = bf
        self.w = None
        self._outputs = outputs

        self._output_files = {output: fe.XDMFFile(f'./{job_name}/{output.name}.xdmf') for output in self._outputs}

    @property
    def constitutive_model(self):
        return self._constitutive_model

    @abstractmethod
    def update_values(self):
        raise NotImplementedError

    @abstractmethod
    def write_outputs(self, t):
        raise NotImplementedError


class UDomain(Domain):

    def __init__(self,
                 mesh: fe.Mesh,
                 constitutive_model: ConstitutiveModelBase,
                 density: fe.Expression = fe.Expression('0', degree=0),
                 bf: fe.Expression = fe.Expression('0', degree=0),
                 user_output_fn: callable = None):

        W = fe.VectorFunctionSpace(mesh, "P", 1)
        super().__init__(mesh, constitutive_model, W, bf)
        self._density = density

        self.user_output_fn = user_output_fn

        # Unknowns, values at previous step and test functions
        self.w = fe.Function(W)
        self.u, self.u0 = self.w, fe.Function(W)
        self.v, self.v0 = fe.Function(W), fe.Function(W)
        self.a, self.a0 = fe.Function(W), fe.Function(W)

        # self.a0 = fe.Function(fe.FunctionSpace(mesh, element_v))

        self.ut = fe.TestFunction(W)

        self.F = kin.def_grad(self.u)
        self.F0 = kin.def_grad(self.u0)

        self.output_fn_map = {Outputs.stress: self.write_stress,
                              Outputs.strain: self.write_strain,
                              Outputs.displacement: self.write_u}

    def update_values(self):
        self.u0.assign(self.u)
        self.v0.assign(self.v)
        self.a0.assign(self.a)

        self.F0 = kin.def_grad(self.u0)

    def write_outputs(self, t):

        if self.user_output_fn is not None:
            self.user_output_fn(self, t)

        for output in self._outputs:
            self.output_fn_map[output](self._output_files[output], t)

    def write_stress(self, file: fe.XDMFFile, t):
        s = self.constitutive_model.stress(self.u)
        file.write(fe.project(s), t)

    def write_strain(self, file: fe.XDMFFile, t):
        E = kin.green_lagrange_strain(self.u)
        file.write(fe.project(E), t)

    def write_u(self, file: fe.XDMFFile, t):
        file.write(self.u, t)

    @property
    def density(self):
        return self._density

    @property
    def constitutive_model(self):
        return self._constitutive_model


class PDomain(Domain):

    def __init__(self,
                 mesh,
                 constiutive_model):
        pass


class UPDomain(UDomain, PDomain):

    def __init__(self, mesh: fe.Mesh, constitutive_model: ConstitutiveModelBase):
        super().__init__(mesh, constitutive_model)
