import fenics as fe


def identity(u):
    return fe.Identity(u.ufl_shape[0])


def def_grad(u):
    I = fe.Identity(u.ufl_shape[0])
    return I + fe.grad(u)


def right_cauchy_green(u):
    F = def_grad(u)
    return F.T * F


def green_lagrange_strain(u):
    I = identity(u)
    C = right_cauchy_green(u)
    return (C - I) / 2


def iso_def_grad(u):
    F = def_grad(u)
    F /= fe.det(F) ** (1./3.)
    return F


def iso_right_cauchy_green(u):
    iso_F = iso_def_grad(u)
    return iso_F.T * iso_F


def iso_green_lagrange_strain(u):
    I = identity(u)
    iso_C = iso_right_cauchy_green(u)
    return (iso_C - I) / 2
