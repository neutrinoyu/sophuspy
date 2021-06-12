import unittest

import numpy as np
from sophuspy import *
from scipy.spatial.transform import Rotation as R

class TestQuaternion(unittest.TestCase):
    def test_creat(self):
        q = Quaternion()
        self.assertTrue(np.array_equal(q.data, [0,0,0,1]))

    def test_print(self):
        q = Quaternion()
        self.assertTrue(q.__str__() == "[0. 0. 0. 1.]", f"output is {q.__str__()}")

    def test_equal(self):
        q1 = Quaternion(0.1,0.2,0.3,0.4)
        q2 = Quaternion(0.1,0.2,0.3,0.4)
        self.assertTrue(np.array_equal(q1.data,q2.data))

    def test_mul(self):
        q1 = Quaternion(0.1,0.2,0.3,0.4)
        q2 = Quaternion(0.1,0.2,0.3,0.4)
        q3 = q1*q2

        scipy_q1 = R.from_quat([0.1,0.2,0.3,0.4])
        scipy_q2 = R.from_quat([0.1,0.2,0.3,0.4])
        scipy_q3 = scipy_q1*scipy_q2
        # print(q3.data)
        # print(scipy_q3.as_quat())
        self.assertTrue(np.allclose(q3.data, scipy_q3.as_quat()))

    def test_mul2(self):
        for i in range(100):
            quat = np.random.rand(4)
            q1 = Quaternion(quat[0],quat[1],quat[2],quat[3])
            q2 = Quaternion(quat[0],quat[1],quat[2],quat[3])
            q3 = q1*q2

            scipy_q1 = R.from_quat([quat[0],quat[1],quat[2],quat[3]])
            scipy_q2 = R.from_quat([quat[0],quat[1],quat[2],quat[3]])
            scipy_q3 = scipy_q1*scipy_q2
            self.assertTrue(np.allclose(q3.data, scipy_q3.as_quat()))

    def test_as_matrix(self):
        for i in range(1):
            quat = np.random.rand(4)
            q1 = Quaternion(quat[0],quat[1],quat[2],quat[3])
            scipy_q1 = R.from_quat([quat[0],quat[1],quat[2],quat[3]])
            print(q1.as_matrix())
            print(scipy_q1.as_matrix())
            self.assertTrue(np.allclose(q1.as_matrix(), scipy_q1.as_matrix()))

    def test_apply_vector(self):
        for i in range(100):
            quat = np.random.rand(4)
            v = np.random.rand(3)
            q1 = Quaternion(quat[0],quat[1],quat[2],quat[3])
            scipy_q1 = R.from_quat([quat[0],quat[1],quat[2],quat[3]])
            v2 = q1@v
            v3 = scipy_q1.apply(v)
            print(v2)
            print(v3)
            self.assertTrue(np.allclose(v2,v3))

    def test_inverse(self):
        for i in range(100):
            quat = np.random.rand(4)
            print(f"random vector {quat}")
            q1 = Quaternion(quat[0],quat[1],quat[2],quat[3])
            scipy_q1 = R.from_quat([quat[0],quat[1],quat[2],quat[3]])
            q2 = q1.inverse()
            scipy_q2 = scipy_q1.inv()
            print(q2)
            print(scipy_q2.as_quat())
            self.assertTrue(np.allclose(q2.as_matrix(),scipy_q2.as_matrix()))


if __name__ == "__main__":
    unittest.main()