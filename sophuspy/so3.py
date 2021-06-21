import numpy as np
from numpy.lib.arraysetops import isin

from .quaternion import Quaternion

class SO3:
    def __init__(self, quat = Quaternion()):
        if not isinstance(quat, Quaternion):
            raise ValueError
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
    def exp_theta(vec):
        np_vec = np.asarray(vec)
        if np_vec.size != 3 or (np_vec.shape[0] != 3 and np_vec.shape[1] != 3):
            raise ValueError()
        tht_p2 = (np_vec**2).sum()
        tht = np.sqrt(tht_p2)

        if tht_p2 < 1e-10:
            tht_p4 = tht_p2 * tht_p2
            imag = 0.5 - 1/48 * tht_p2 + 1/3840 * tht_p4
            real = 1 - 1/8 * tht_p2 + 1/384 * tht_p4
        else:
            imag = np.sin(tht / 2) / tht
            real = np.cos(tht / 2)
        quat = Quaternion(imag * np_vec[0], imag * np_vec[1], imag * np_vec[2], real)
        
        return SO3(quat), tht

    @staticmethod
    def exp(vec):
        return SO3.exp_theta(vec)[0]

    @staticmethod
    def from_rotv(vec):
        return SO3.exp(vec)
    
    @staticmethod
    def from_mat(mat):
        np_vec = np.asarray(mat)
        if np_vec.shape != (3,3):
            raise ValueError()
        raise

    @staticmethod
    def identity():
        return SO3(Quaternion(0,0,0,1))
        

    def matrix(self):
        return self.unit_quaternion.as_matrix()

    def log(self):
        sq_norm = (self.unit_quaternion.vec**2).sum()
        w = self.unit_quaternion.data[3]

        t = 0
        if sq_norm < 1e-10:
            t = 2 / w - (1 - sq_norm / w**2 / 3)
        else:
            n = np.sqrt(sq_norm)
            if abs(w) < 1e-6:
                if w > 0:
                    t = np.pi / n
                else:
                    t = - np.pi / n
            else:
                t = 2 * np.arctan(n/w) / n
        return t * self.unit_quaternion.vec

    @staticmethod
    def hat(omega):
        matrix = np.zeros((3,3))
        matrix[1,2] = -omega[0]
        matrix[2,1] = omega[0]
        matrix[0,2] = omega[1]
        matrix[2,0] = -omega[1]
        matrix[0,1] = -omega[2]
        matrix[1,0] = omega[2]
        return matrix

    @staticmethod
    def vee(Omega):
        return np.asarray(Omega[2,1], Omega[0,2], Omega[1,0])

    def inverse(self):
        return SO3(self.unit_quaternion.inverse())

    def __mul__(self, other):
        return SO3(self.unit_quaternion * other.unit_quaternion)

    def __matmul__(self, v):
        return self.unit_quaternion @ v
    
    def __repr__(self):
        log_ = self.log()
        return "SO3(%.4f, %.4f, %.4f)" % (log_[0], log_[1], log_[2])

    def __str__(self):
        log_ = self.log()
        return str(log_)