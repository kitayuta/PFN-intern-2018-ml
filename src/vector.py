import math


class Vector:

    def __init__(self, initializer):
        self.list = list(initializer)

    def __len__(self):
        return len(self.list)

    @property
    def shape(self):
        return len(self),

    def __getitem__(self, index):
        return self.list[index]

    def __add__(self, other):
        """Calculate the sum of two vectors."""
        if self.shape != other.shape:
            raise ValueError
        l = [self[i] + other[i] for i in range(len(self))]
        return Vector(l)


def scalar_mul(a, v):
    """Calculate scalar multiplication of a vector."""
    l = [a * v[i] for i in range(len(v))]
    return Vector(l)


def dot_prod(v, w):
    """Calculate the dot product of two vectors."""
    if v.shape != w.shape:
        raise ValueError
    p = 0.0
    for i in range(len(v)):
        p += v[i] * w[i]
    return p


def ReLU(v):
    """Calculate ReLU function over a vector."""
    l = [max(0.0, v[i]) for i in range(len(v))]
    return Vector(l)


def ReLU_backward(p, q):
    """Calculate the backward for ReLU function."""
    if p.shape != q.shape:
        raise ValueError
    l = [p[i] if q[i] > 0.0 else 0.0 for i in range(len(p))]
    return Vector(l)


def softmax(v):
    """Calculate softmax function over a vector."""
    exp_x = [math.exp(v[i]) for i in range(len(v))]
    exp_sum = sum(exp_x)
    l = [exp_x[i] / exp_sum for i in range(len(exp_x))]
    return Vector(l)


def argmax(v):
    """Calculate argmax of a vector."""
    return v.list.index(max(v.list))
