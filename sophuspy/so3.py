import numpy as np

from .quaternion import Quaternion

class SO3:
    def __init__(self, quat = Quaternion()):
        self.unit_quaternion = quat
        
    @staticmethod
    def from_quat(quat, order = "xyzw"):
        np_quat = np.asarray(quat)
        if np_quat.size != 4:
            raise ValueError()
        if order == "xyzw":
            q = Quaternion(np_quat[0], np_quat[1], np_quat[2], np_quat[3])
        elif order == "wxyz":
            q = Quaternion(np_quat[1], np_quat[2], np_quat[3], np_quat[0])
        else:
            raise ValueError
        return SO3(q)

    @staticmethod
    def exp(vec):
        np_vec = np.asarray(vec)
        if np_vec.size != 3 or (np_vec.shape[0] != 3 and np_vec.shape[1] != 3):
            raise ValueError()

    @staticmethod
    def from_rotv(vec):
        return SO3.exp(vec)
    
    @staticmethod
    def from_mat(mat):
        np_vec = np.asarray(mat)
        if np_vec.shape != (3,3):
            raise ValueError()

    def matrix(self):
        return self.quat.as_matrix()

    def log(self):
        pass

    def inverse(self):
        return SO3(self.unit_quaternion.inverse())

    def __mul__(self, other):
        return SO3(self.unit_quaternion * other)

    def __matmul__(self, v):
        return self.unit_quaternion @ v
    
