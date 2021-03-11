import fenics as fe


def identity(u):
    return fe.Identity(u.ufl_shape[0])


def def_grad(u):
    # TODO: Check dim is correct
    I = fe.Identity(u.ufl_shape[0])
    return I + fe.grad(u)


def right_cauchy_green(u):
    F = def_grad(u)
    return F.T * F


def green_lagrange_strain(u):
    I = identity(u)
    C = right_cauchy_green(u)
    return (C - I) / 2