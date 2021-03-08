import fenics as fe


def F(u):
    # TODO: Check dim is correct
    I = fe.Identity(u.cell().dim)
    return I + fe.grad(u)
