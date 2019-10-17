import unittest
import vector
from tests.utils import assert_allclose


class TestVector(unittest.TestCase):

    def setUp(self):
        self.v = vector.Vector([0.5, 1.5, 2.5])

    def test_init(self):
        self.assertEqual(self.v.list, [0.5, 1.5, 2.5])

    def test_len(self):
        self.assertEqual(len(self.v.list), 3)

    def test_shape(self):
        self.assertEqual(self.v.shape, (3,))

    def test_getitem(self):
        self.assertEqual(self.v[1], 1.5)

    def test_add(self):
        w = vector.Vector([0.5, -0.5, -1.5])
        s = self.v + w
        self.assertIsInstance(s, vector.Vector)
        assert_allclose(self, self.v + w, vector.Vector([1.0, 1.0, 1.0]))

        with self.assertRaises(ValueError):
            self.v + vector.Vector([1.0])


class TestVectorFunctions(unittest.TestCase):

    def setUp(self):
        self.v = vector.Vector([-0.5, 1.5, 2.5])

    def test_scalar_mul(self):
        res = vector.scalar_mul(1.5, self.v)
        assert_allclose(self, res, vector.Vector([-0.75, 2.25, 3.75]))

    def test_dot_prod(self):
        p = vector.dot_prod(self.v, self.v)
        self.assertAlmostEqual(p, 8.75)

        with self.assertRaises(ValueError):
            vector.dot_prod(self.v, vector.Vector([1.0]))

    def test_ReLU(self):
        res = vector.ReLU(self.v)
        self.assertIsInstance(res, vector.Vector)
        assert_allclose(self, res, vector.Vector([0.0, 1.5, 2.5]))

    def test_ReLU_backward(self):
        w = vector.Vector([2.5, 1.5, -0.5])
        res = vector.ReLU_backward(self.v, w)
        self.assertIsInstance(res, vector.Vector)
        assert_allclose(self, res, vector.Vector([-0.5, 1.5, 0.0]))

        with self.assertRaises(ValueError):
            vector.ReLU_backward(self.v, vector.Vector([1.0]))

    def test_softmax(self):
        res = vector.softmax(self.v)
        self.assertIsInstance(res, vector.Vector)
        self.assertAlmostEqual(sum(res.list), 1.0)
        assert_allclose(self, res, vector.Vector([0.03511903, 0.25949646, 0.70538451]))

    def test_argmax(self):
        self.assertEqual(vector.argmax(self.v), 2)
