import unittest
import vector
import matrix
from tests.utils import assert_allclose


class TestMatrix(unittest.TestCase):

    def setUp(self):
        self.m = matrix.Matrix([
            vector.Vector([0.0, 0.5, 1.0]),
            vector.Vector([1.5, 2.0, 2.5]),
        ])

    def test_init(self):
        self.assertEqual(self.m.rows[0].list, [0.0, 0.5, 1.0])
        self.assertEqual(self.m.rows[1].list, [1.5, 2.0, 2.5])

    def test_shape(self):
        self.assertEqual(self.m.shape, (2, 3))

    def test_T(self):
        mt = self.m.T
        self.assertEqual(mt.shape, (3, 2))
        self.assertEqual(mt.rows[0].list, [0.0, 1.5])
        self.assertEqual(mt.rows[1].list, [0.5, 2.0])
        self.assertEqual(mt.rows[2].list, [1.0, 2.5])


class TestMatrixFunctions(unittest.TestCase):

    def setUp(self):
        self.m = matrix.Matrix([
            vector.Vector([0.0, 0.5, 1.0]),
            vector.Vector([1.5, 2.0, 2.5]),
        ])
        self.v = vector.Vector([0.0, 1.5, 3.0])

    def test_mat_vec_prod(self):
        res = matrix.mat_vec_prod(self.m, self.v)
        assert_allclose(self, res, vector.Vector([3.75, 10.5]))

        with self.assertRaises(ValueError):
            matrix.mat_vec_prod(self.m, vector.Vector([1.0]))
