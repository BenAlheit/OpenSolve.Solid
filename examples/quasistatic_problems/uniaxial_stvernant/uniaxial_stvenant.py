import fenics as fe

from solid_mechanics.constiutive_model import StVenant

from solver.domain import UDomain
from solver.step import StaticStep
from solver.solver import SolidMechanicsSolver

import matplotlib.pyplot as plt

n = 2
mesh = fe.UnitCubeMesh(n, n, n)
stress = []
time = []
lams = []


def output_fn(domain: UDomain, t):
    time.append(t)
    lams.append(fe.project(domain.F[0, 0], domain.S).vector().get_local().mean())
    stress.append(fe.project(domain.constitutive_model.stress(domain.u)[0, 0], domain.S).vector().get_local().mean())


domain = UDomain(mesh,
                 StVenant({'mu': 1,
                           'lambda': 1}),
                 user_output_fn=output_fn)

lam = 2

zero = fe.Constant(0)
pull = fe.Expression('time*(lam - 1.)', time=0, lam=lam, degree=1)
exs = [pull]

step1 = StaticStep(
    domain=domain,
    dbcs=[
        fe.DirichletBC(domain.V.sub(0), zero, 'on_boundary && near(x[0], 0)'),
        fe.DirichletBC(domain.V.sub(1), zero, 'on_boundary && near(x[1], 0)'),
        fe.DirichletBC(domain.V.sub(2), zero, 'on_boundary && near(x[2], 0)'),
        fe.DirichletBC(domain.V.sub(0), pull, 'on_boundary && near(x[0], 1.)'),
    ],
    t_start=0,
    t_end=1.,
    dt0=fe.Expression('dt', dt=0.1, degree=1),
    expressions=[pull])

solver = SolidMechanicsSolver([step1])
solver.solve()

plt.plot(time, stress)
plt.show()


# TODO Implement UP
# TODO Add viscoelasticity

# TODO Traction bcs
# TODO body forces
# TODO fix automatic differentiation of strain energy functions
# TODO Verify StVernant material model

# TODO Add anisotropy
# TODO restarts at some point in the future?
# TODO Think about starting to add tests
