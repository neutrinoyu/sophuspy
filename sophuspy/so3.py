import numpy as np

from .quaternion import Quaternion

class SO3:
    def __init__(self):
        self.unit_quaternion = Quaternion()
        
    @staticmethod
    def from_quat(vec):
        np_vec = np.asarray(vec)
        if np_vec.size != 3:
            raise ValueError()

    @staticmethod
    def exp(vec):
        np_vec = np.asarray(vec)
        if np_vec.size != 3:
            raise ValueError()

    @staticmethod
    def from_rotv(vec):
        return exp(vec)
    
    @staticmethod
    def from_mat(mat):
        np_vec = np.asarray(vec)
        if np_vec.size != 3:
            raise ValueError()

    def matrix(self):
        return self.quat.as_matrix()

    def log(self):
        pass

    def inverse(self):
        pass

    @staticmethod
    def mul_quat_qual(a, b):
        x = a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z
        y = a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y
        z = a.w * b.y + a.y * b.w + a.z * b.x - a.x * b.z
        w = a.w * b.z + a.z * b.w + a.x * b.y - a.y * b.x

        return SO3.from_quat([x,y,z,w])

    @staticmethod
    def mul_quat_vec(q, v):
        uv = np.cross(q[:3], v)
        uv += uv

        return v + q[3] * uv + np.cross(q[:3], uv)
    
