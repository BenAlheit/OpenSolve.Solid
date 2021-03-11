import fenics as fe

from solid_mechanics.constiutive_model import NeoHookean

from solver.domain import UDomain
from solver.step import StaticStep
from solver.solver import SolidMechanicsSolver
from solver.boundary_condition import DirichletBC

n = 2
mesh = fe.UnitCubeMesh(n, n, n)
domain = UDomain(mesh, NeoHookean({'mu': 1,
                                   'lambda': 1}))

lam = 2

zero = fe.Constant(0)
ex = fe.Expression('time*(lam - 1.)', time=0, lam=lam, degree=1)
step1 = StaticStep(
    domain=domain,
    dbcs=[
        DirichletBC(domain.V.sub(0), zero, 'on_boundary && near(x[0], 0)'),
        DirichletBC(domain.V.sub(1), zero, 'on_boundary && near(x[1], 0)'),
        DirichletBC(domain.V.sub(2), zero, 'on_boundary && near(x[2], 0)'),
        DirichletBC(domain.V.sub(0),
                       ex,
                       # fe.Expression('(lam - 1.)', time=0, lam=lam, degree=1),
                       'on_boundary && near(x[0], 1.)'),
    ],
    t_start=0,
    t_end=1.,
    dt0=fe.Expression('dt', dt=0.1, degree=1), ex=ex)

solver = SolidMechanicsSolver([step1])
solver.solve()

# TODO fix time parameter issue
# TODO fix automatic differentiation of strain energy functions
# TODO Implement UP
