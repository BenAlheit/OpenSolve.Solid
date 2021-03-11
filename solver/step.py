import fenics as fe
from abc import ABC, abstractmethod
from .problems import StationaryProblem
from .domain import Domain
from .boundary_condition import DirichletBC

# TODO: Think about if dictionary is best for bcs


class Step(ABC):

    def __init__(self, dt0: fe.Expression, dbcs: [DirichletBC], domain: Domain):
        self._dt = dt0
        self._fedbcs = list(map(lambda x: x.feDBC, dbcs))
        self._dbcs = dbcs
        self._domain = domain

    @abstractmethod
    def run(self):
        raise NotImplementedError

    @property
    def dt(self):
        return self._dt

    @property
    def bcs(self):
        return self._bcs

    def update_time(self, t):
        for dbc in self._dbcs:
            try:
                dbc.time = t

            except:
                pass


class StaticStep(Step):

    def __init__(self,
                 domain: Domain,
                 dbcs: [fe.DirichletBC],
                 t_start: float,
                 t_end: float,
                 dt0: fe.Expression,
                 ex = None,
                 u0: fe.Expression = None):
        super().__init__(dt0, dbcs, domain)

        self.t_start = t_start
        self.t_end = t_end

        self.ex = ex
        self._sp: StationaryProblem = domain
        self.d_LHS = None
        self.d_RHS = None
        self.solver: fe.NonlinearVariationalSolver = None

    def construct_functionals(self):
        pass

    @property
    def stationary_problem(self):
        return self._sp

    @stationary_problem.setter
    def stationary_problem(self, value: StationaryProblem):
        self._sp: StationaryProblem = value

    def create_step_functionals(self):
        u = self._sp.u
        # u0 = self._sp.u0
        # bf = self._sp.bf
        ut = self._sp.ut
        F, F0 = self._sp.F, self._sp.F0

        constitutive_model = self._sp.constitutive_model

        self.d_LHS = fe.inner(F * constitutive_model.stress(u), fe.grad(ut)) * fe.dx

        self.d_RHS = (fe.inner(fe.Constant((0., 0., 0.)), ut) * fe.dx)
        # self.d_RHS = (fe.inner(bf, ut) * fe.dx)
        # + fe.inner(self.bcs['trac'], ut) * fe.ds)

    # def create_step_functionals(self):
    #     u, v, p = self._sp.u, self._sp.v, self._sp.p
    #     u0, v0, p0 = self._sp.u0, self._sp.v0, self._sp.p0
    #     a0 = self._sp.a0
    #     bf = self._sp.bf
    #     ut, vt, pt = self._sp.ut, self._sp.vt, self._sp.pt
    #     F, F0 = self._sp.F, self._sp.F0
    #     density = self._sp.density
    #
    #     constitutive_model = self._sp.constitutive_model
    #
    #     self.d_LHS = fe.inner(u, ut) * density * fe.dx \
    #                  - pow(self._dt, 2) * self._beta * fe.inner(F * constitutive_model.stress(F), fe.grad(ut)) * fe.dx
    #
    #     self.d_RHS = density * (fe.inner(u0, ut) * fe.dx
    #                             + self._dt * fe.inner(v0, ut) * fe.dx
    #                             + pow(self._dt, 2) * (0.5 - self._beta) * fe.inner(a0, ut) * fe.dx) \
    #                  - pow(self._dt, 2) * self._beta * (fe.inner(bf, ut) * fe.dx
    #                                                     + fe.inner(self.bcs['trac'], ut) * fe.ds)

    def setup_solver(self):
        F = self.d_LHS - self.d_RHS
        J = fe.derivative(F, self._sp.w)

        # Initialize solver
        problem = fe.NonlinearVariationalProblem(F, self._sp.w, bcs=self._fedbcs, J=J)
        self.solver = fe.NonlinearVariationalSolver(problem)
        self.solver.parameters['newton_solver']['relative_tolerance'] = 1e-6
        self.solver.parameters['newton_solver']["linear_solver"] = "cg"
        self.solver.parameters['newton_solver']["preconditioner"] = "ilu"

    def run(self):
        self.create_step_functionals()
        self.setup_solver()
        t = self.t_start
        while t < self.t_end:
            t += self.dt.dt
            # g_time.time += self.dt.dt
            self.ex.time += self.dt.dt
            self.update_time(t)
            self.solver.solve()
            self._sp.update_values()


class ImplicitStep(Step):
    def __init__(self,
                 domain: Domain,
                 bcs: dict,
                 t_start: float,
                 t_end: float,
                 dt0: fe.Expression,
                 alpha: fe.Constant = fe.Constant(0.),
                 u0: fe.Expression = None,
                 v0: fe.Expression = None,
                 a0: fe.Expression = None):

        super().__init__(dt0, bcs, domain)

        self.t_start = t_start
        self.t_end = t_end

        self._alpha = alpha
        self._beta = (1 - self._alpha) ** 2 / 4.
        self._gamma = 1. / 2. - self._alpha

        self._sp: StationaryProblem = None
        self.d_LHS = None
        self.d_RHS = None
        self.solver: fe.NonlinearVariationalSolver = None

    def construct_functionals(self):
        pass

    @property
    def stationary_problem(self):
        return self._sp

    @stationary_problem.setter
    def stationary_problem(self, value: StationaryProblem):
        self._sp: StationaryProblem = value

    def create_step_functionals(self):
        u, v = self._sp.u, self._sp.v
        u0, v0 = self._sp.u0, self._sp.v0
        a0 = self._sp.a0
        bf = self._sp.bf
        ut = self._sp.ut
        F, F0 = self._sp.F, self._sp.F0
        density = self._sp.density

        constitutive_model = self._sp.constitutive_model

        self.d_LHS = fe.inner(u, ut) * density * fe.dx \
                     - pow(self._dt, 2) * self._beta * fe.inner(F * constitutive_model.stress(F), fe.grad(ut)) * fe.dx

        self.d_RHS = density * (fe.inner(u0, ut) * fe.dx
                                + self._dt * fe.inner(v0, ut) * fe.dx
                                + pow(self._dt, 2) * (0.5 - self._beta) * fe.inner(a0, ut) * fe.dx) \
                     - pow(self._dt, 2) * self._beta * (fe.inner(bf, ut) * fe.dx)
        # + fe.inner(self.bcs['trac'], ut) * fe.ds)

    # def create_step_functionals(self):
    #     u, v, p = self._sp.u, self._sp.v, self._sp.p
    #     u0, v0, p0 = self._sp.u0, self._sp.v0, self._sp.p0
    #     a0 = self._sp.a0
    #     bf = self._sp.bf
    #     ut, vt, pt = self._sp.ut, self._sp.vt, self._sp.pt
    #     F, F0 = self._sp.F, self._sp.F0
    #     density = self._sp.density
    #
    #     constitutive_model = self._sp.constitutive_model
    #
    #     self.d_LHS = fe.inner(u, ut) * density * fe.dx \
    #                  - pow(self._dt, 2) * self._beta * fe.inner(F * constitutive_model.stress(F), fe.grad(ut)) * fe.dx
    #
    #     self.d_RHS = density * (fe.inner(u0, ut) * fe.dx
    #                             + self._dt * fe.inner(v0, ut) * fe.dx
    #                             + pow(self._dt, 2) * (0.5 - self._beta) * fe.inner(a0, ut) * fe.dx) \
    #                  - pow(self._dt, 2) * self._beta * (fe.inner(bf, ut) * fe.dx
    #                                                     + fe.inner(self.bcs['trac'], ut) * fe.ds)

    def setup_solver(self):
        F = self.d_LHS - self.d_RHS
        J = fe.derivative(F, self._sp.w)

        # Initialize solver
        problem = fe.NonlinearVariationalProblem(F, self._sp.w, bcs=self.bcs, J=J)
        self.solver = fe.NonlinearVariationalSolver(problem)
        self.solver.parameters['newton_solver']['relative_tolerance'] = 1e-6

    def run(self):
        t = self.t_start
        while t < self.t_end:
            t += self.dt.dt
            self.solver.solve()
            self._sp.update_values()
