import fenics as fe
from ..solid_mechanics.constiutive_model import ConstitutiveModelBase


class Domain:
    def __init__(self,
                 mesh: fe.Mesh,
                 density: fe.Expression,
                 constitutive_model: ConstitutiveModelBase,
                 function_space: fe.FunctionSpace,
                 bf: fe.Expression = fe.Expression('0', degree=0)):

        self._mesh = mesh
        self._density = density
        self._constitutive_model = constitutive_model
        self._w = function_space
        self._bf = bf
