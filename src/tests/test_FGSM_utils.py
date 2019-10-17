import unittest
import vector
import FGSM_utils


class TestFGSMUtils(unittest.TestCase):

    def test_sign(self):
        res = FGSM_utils.sign(vector.Vector([0.5, 1.5, -0.5]))
        self.assertEqual(res, [1, 1, -1])

    def test_clip(self):
        self.assertEqual(FGSM_utils.clip(-100), 0)
        self.assertEqual(FGSM_utils.clip(0), 0)
        self.assertEqual(FGSM_utils.clip(100), 100)
        self.assertEqual(FGSM_utils.clip(255), 255)
        self.assertEqual(FGSM_utils.clip(300), 255)

    def test_perturb(self):
        res = FGSM_utils.perturb([0, 100, 200, 0, 255], [1, -1, 1, -1, 1], 0.1)
        self.assertEqual(res, [25, 75, 225, 0, 255])
