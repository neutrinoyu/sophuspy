import unittest

import numpy as np
from sophuspy import *

class TestQuaternion(unittest.TestCase):
    def test_creat(self):
        q = Quaternion()
        self.assertTrue(np.isclose(q.data, [0,0,0,1]).all())


if __name__ == "__main__":
    unittest.main()