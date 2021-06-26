import unittest

import numpy as np
from sophuspy import *
from scipy.spatial.transform import Rotation as R

class TestSO3(unittest.TestCase):
    def test_creat(self):
        q = SO3()
        print(q.unit_quaternion)
        self.assertTrue(np.array_equal(q.unit_quaternion.data, [0,0,0,1]))

    # def test_print(self):
    #     q = SO3()
    #     self.assertTrue(q.__str__() == "[0. 0. 0. 1.]", f"output is {q.__str__()}")
    
    def test_exp(self):
        for i in range(100):
            quat = np.random.rand(3)
            sohpus_so3 = SO3.exp([quat[0], quat[1], quat[2]])
            scipy_r = R.from_rotvec([quat[0],quat[1],quat[2]])
            print(sohpus_so3.matrix())
            print(scipy_r.as_matrix())
            self.assertTrue(np.allclose(sohpus_so3.matrix(), scipy_r.as_matrix()))

    def test_log(self):
        for i in range(100):
            quat = np.random.rand(4)
            sohpus_so3 = SO3.from_quat([quat[0],quat[1],quat[2],quat[3]])
            scipy_r = R.from_quat([quat[0],quat[1],quat[2],quat[3]])
            print(sohpus_so3.log())
            print(scipy_r.as_rotvec())
            self.assertTrue(np.allclose(sohpus_so3.log(), scipy_r.as_rotvec()))

    def test_dervative(self):
        v = np.array([0.5,0.4,-.5])
        dv_t = 0.0001
        dv = np.array([dv_t , 0.000, 0.000])
        der = (SO3.exp(v+dv).unit_quaternion.data - SO3.exp(v).unit_quaternion.data) / dv_t
        der2 = SO3.Dx_exp_x(v) @ dv / dv_t
        print(der, der2, np.linalg.norm(der-der2))
        self.assertTrue(np.linalg.norm(der-der2) < 2e-5)

    def test_dervative_so3(self):
        dv_t = 0.0001
        v = np.array([0,0,0])
        dv_array = [np.array([1.0 , 0.0, 0.0]), np.array([0.0 , 1.0, 0.0]), np.array([0.0 , 0.0, 1.0])]
        for i in range(100):
            if i != 0:
                v = np.random.rand(3)
            for dv in dv_array:
                dv = dv * dv_t

                der = ((SO3.exp(v).inverse() * SO3.exp(v + dv)).log()) / dv_t
                print("sim derivative: ", der)
                norm_v = np.linalg.norm(v)

                Jr = SO3.Jr(v)
                der2 = Jr @ dv / dv_t
                print("SO3 calc derivative: ", der2)
                print("diff: ", der2 - der, np.abs((der2 - der)).max())
                self.assertTrue(np.abs((der2 - der)).max() < 2e-5)

                der = ((SO3.exp(v) * SO3.exp(dv)).log() - SO3.exp(v).log()) / dv_t
                print("sim derivative: ", der)
                norm_v = np.linalg.norm(v)

                Jr_inv = SO3.Jr_inv(v)
                der2 = Jr_inv @ dv / dv_t
                print("SO3 calc derivative: ", der2)
                print("diff: ", der2 - der, np.abs((der2 - der)).max())
                self.assertTrue(np.abs((der2 - der)).max() < 2e-5)


                der = ((SO3.exp(v + dv) * SO3.exp(v).inverse()).log()) / dv_t
                print("sim derivative: ", der)
                norm_v = np.linalg.norm(v)

                Jl = SO3.Jl(v)
                der2 = Jl @ dv / dv_t
                print("SO3 calc derivative: ", der2)
                print("diff: ", der2 - der, np.abs((der2 - der)).max())
                self.assertTrue(np.abs((der2 - der)).max() < 2e-5)

                der = ((SO3.exp(dv) * SO3.exp(v)).log() - SO3.exp(v).log()) / dv_t
                print("sim derivative: ", der)
                norm_v = np.linalg.norm(v)

                Jl_inv = SO3.Jl_inv(v)
                der2 = Jl_inv @ dv / dv_t
                print("SO3 calc derivative: ", der2)
                print("diff: ", der2 - der, np.abs((der2 - der)).max())
                self.assertTrue(np.abs((der2 - der)).max() < 2e-5)