import vector
import matrix


class Predictor:

    def __init__(self, params):
        self.w1 = params[0][0]
        self.w1T = self.w1.T
        self.b1 = params[0][1]
        self.w2 = params[1][0]
        self.w2T = self.w2.T
        self.b2 = params[1][1]
        self.w3 = params[2][0]
        self.w3T = self.w3.T
        self.b3 = params[2][1]

    def forward(self, x):
        """Do the forward computation.

        Args:
            x (Vector): Vector of the pixels of a picture.

        Returns:
            Vector: Vector of the probabilities.

        """
        self.a1 = matrix.mat_vec_prod(self.w1, x) + self.b1
        h1 = vector.ReLU(self.a1)
        self.a2 = matrix.mat_vec_prod(self.w2, h1) + self.b2
        h2 = vector.ReLU(self.a2)
        y = matrix.mat_vec_prod(self.w3, h2) + self.b3
        self.fx = vector.softmax(y)
        return self.fx

    def backward(self, label):
        """Do the backward computation.

        Args:
            label (int): True label of the picture.

        Returns:
            Vector: Gradient of the loss function.

        """
        dy = vector.Vector([-1.0 if t == label else 0.0 for t in range(len(self.b3))]) + self.fx
        dh2 = matrix.mat_vec_prod(self.w3T, dy)
        da2 = vector.ReLU_backward(dh2, self.a2)
        dh1 = matrix.mat_vec_prod(self.w2T, da2)
        da1 = vector.ReLU_backward(dh1, self.a1)
        dx = matrix.mat_vec_prod(self.w1T, da1)
        return dx


def read_params(filename):
    """Read parameters of a predictor."""

    def read_matrix(m, n):
        rows = []
        for _ in range(m):
            line = f.readline()
            r = vector.Vector(map(float, line.split()))
            assert len(r) == n  # validate
            rows.append(r)
        return matrix.Matrix(rows)

    def read_vector(n):
        line = f.readline()
        v = vector.Vector(map(float, line.split()))
        assert len(v) == n  # validate
        return v

    H, N, C = 256, 1024, 23
    with open(filename) as f:
        w1 = read_matrix(H, N)
        b1 = read_vector(H)
        w2 = read_matrix(H, H)
        b2 = read_vector(H)
        w3 = read_matrix(C, H)
        b3 = read_vector(C)

    return (w1, b1), (w2, b2), (w3, b3)
