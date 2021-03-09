import fenics as fe


class Vector:

    def __init__(self, mesh: fe.Mesh, family="P", order=1):

        self.V = fe.VectorFunctionSpace(mesh, family, order)

