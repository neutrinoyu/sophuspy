import unittest

import numpy as np
from sophuspy import *
from scipy.spatial.transform import Rotation as R

class TestSE3(unittest.TestCase):

    def test_log(self):
        for i in range(1):
            se_log = np.random.rand(6)
            sohpus_se3 = SE3.exp([se_log[0],se_log[1],se_log[2],se_log[3],se_log[4],se_log[5]])
            print(sohpus_se3.log())
            print(se_log)
            self.assertTrue(np.allclose(sohpus_se3.log(), se_log))