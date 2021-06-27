import numpy as np
from numpy.lib.arraysetops import isin
from numpy.linalg.linalg import norm

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
        np_vec = np_vec.reshape((3,))
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
            t = 2 / w -  2/3 * sq_norm / w**3
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

    @staticmethod
    def Dx_exp_x_at_0():
        J = np.array([[0.5, 0, 0],
                      [0, 0.5, 0],
                      [0, 0, 0.5],
                      [0, 0, 0]])
        return J

    @staticmethod
    def Dx_exp_x(omega):
        c0 = omega[0] * omega[0]
        c1 = omega[1] * omega[1]
        c2 = omega[2] * omega[2]
        c3 = c0 + c1 + c2

        if c3 < 1e-5:
            return SO3.Dx_exp_x_at_0()
        
        c4 = np.sqrt(c3)
        c5 = 1.0 / c4
        c6 = 0.5 * c4
        c7 = np.sin(c6)
        c8 = c5 * c7
        c9 = np.power(c3, -3.0 / 2.0)
        c10 = c7 * c9
        c11 = 1.0 / c3
        c12 = np.cos(c6)
        c13 = 0.5 * c11 * c12
        c14 = c7 * c9 * omega[0]
        c15 = 0.5 * c11 * c12 * omega[0]
        c16 = -c14 * omega[1] + c15 * omega[1]
        c17 = -c14 * omega[2] + c15 * omega[2]
        c18 = omega[1] * omega[2]
        c19 = -c10 * c18 + c13 * c18
        c20 = 0.5 * c5 * c7
        
        J = np.empty((4,3))
        J[0, 0] = -c0 * c10 + c0 * c13 + c8
        J[0, 1] = c16
        J[0, 2] = c17
        J[1, 0] = c16
        J[1, 1] = -c1 * c10 + c1 * c13 + c8
        J[1, 2] = c19
        J[2, 0] = c17
        J[2, 1] = c19
        J[2, 2] = -c10 * c2 + c13 * c2 + c8
        J[3, 0] = -c20 * omega[0]
        J[3, 1] = -c20 * omega[1]
        J[3, 2] = -c20 * omega[2]

        return J

    @staticmethod
    def J_at_0():
        J = np.array([[1.0, 0, 0],
                      [0, 1.0, 0],
                      [0, 0, 1.0]])
        return J

    @staticmethod
    def Jl(omega):
        np_omega = np.asarray(omega).reshape((3,1))
        
        norm_omega = np.linalg.norm(np_omega)

        if norm_omega < 1e-5:
            return SO3.J_at_0()

        unit_omega = np_omega / norm_omega
        t1 = np.sin(norm_omega) / norm_omega

        return t1 * np.eye(3) + (1-t1) * unit_omega * unit_omega.T + (1 - np.cos(norm_omega)) / norm_omega * SO3.hat(unit_omega)

    @staticmethod
    def Jl_inv(omega):
        np_omega = np.asarray(omega).reshape((3,1))
        
        norm_omega = np.linalg.norm(np_omega)

        if norm_omega < 1e-5:
            return SO3.J_at_0()

        unit_omega = np_omega / norm_omega
        t1 = norm_omega / 2 / np.tan(norm_omega / 2)

        return t1 * np.eye(3) + (1-t1) * unit_omega * unit_omega.T - norm_omega / 2 * SO3.hat(unit_omega)

    @staticmethod
    def Jr(omega):
        return SO3.Jl(np.array([-omega[0], -omega[1], -omega[2]]))

    @staticmethod
    def Jr_inv(omega):
        return SO3.Jl_inv(np.array([-omega[0], -omega[1], -omega[2]]))

    def __mul__(self, other):
        if not isinstance(other, SO3):
            raise ValueError
        return SO3(self.unit_quaternion * other.unit_quaternion)

    def __matmul__(self, v):
        return self.unit_quaternion @ v
    
    def __repr__(self):
        log_ = self.log()
        return "SO3(%.4f, %.4f, %.4f)" % (log_[0], log_[1], log_[2])

    def __str__(self):
        log_ = self.log()
        return str(log_)