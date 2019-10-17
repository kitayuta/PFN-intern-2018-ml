import vector


class Matrix:

    def __init__(self, rows):
        """Initialize an instance of Matrix.

        Args:
            rows (list(Vector)): Row vectors of the matrix.

        """
        self.rows = rows

    @property
    def shape(self):
        return len(self.rows), len(self.rows[0])

    @property
    def T(self):
        """Matrix: The transpose."""
        rows = [vector.Vector([self.rows[j][i] for j in range(self.shape[0])]) for i in range(self.shape[1])]
        return Matrix(rows)


def mat_vec_prod(m, v):
    """Calculate the matrix-vector product."""
    if m.shape[1] != v.shape[0]:
        raise ValueError
    l = [vector.dot_prod(m.rows[i], v) for i in range(m.shape[0])]
    return vector.Vector(l)
