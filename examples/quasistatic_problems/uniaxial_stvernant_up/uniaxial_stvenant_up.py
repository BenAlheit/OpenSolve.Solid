import fenics as fe

from solid_mechanics.constiutive_model import IsoStVenant

from solver.domain import UPDomain
from solver.step import StaticStep
from solver.solver import SolidMechanicsSolver

import matplotlib.pyplot as plt

n = 2
mesh = fe.UnitCubeMesh(n, n, n)
stress = []
time = []
lams = []


def output_fn(domain: UPDomain, t):
    pass
    # time.append(t)
    # lams.append(fe.project(domain.F[0, 0], domain.S).vector().get_local().mean())
    # stress.append(fe.project(domain.constitutive_model.stress(domain.u)[0, 0], domain.S).vector().get_local().mean())


def main():
    domain = UPDomain(mesh,
                      IsoStVenant({'mu': 1})
                      # 'lambda': 1}),
                      # user_output_fn=output_fn
                      )

    lam = 2

    zero = fe.Constant(0)
    pull = fe.Expression('time*(lam - 1.)', time=0, lam=lam, degree=1)

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

    plt.plot(lams, stress)
    plt.show()


if __name__ == '__main__':
    main()

# TODO Create UP from scratch in one file
# TODO Implement in framework

