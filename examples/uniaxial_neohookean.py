import fenics as fe

from solid_mechanics.constiutive_model import NeoHookean

from solver.domain import Domain
from solver.step import ImplicitStep
from solver.space_generators import Vector
from solver.solver import SolidMechanicsSolver

n = 1
mesh = fe.UnitCubeMesh(n, n, n)
domain = Domain(mesh, fe.Constant(0.), NeoHookean({'mu': 1}), Vector(mesh).V)

step1 = ImplicitStep(
    domain=domain,
    bcs={'': ''},
    t_start=0,
    t_end=1.,
    dt0=fe.Expression('dt', dt=0.1),)

solver = SolidMechanicsSolver()

# TODO sort out BCS
# TODO sort out saving of solution data
# TODO complete solver really
# TODO Inevitably debug a ton of shit
