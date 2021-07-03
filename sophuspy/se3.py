import numpy as np


from .quaternion import Quaternion
from .so3 import SO3


class SE3:
    def __init__(self, rotation = SO3(), translation = np.zeros((3,))):
        self.rot = rotation
        self.trans = translation

    @staticmethod
    def exp(vec):
        np_vec = np.asarray(vec)
        if np_vec.size != 6 or (np_vec.shape[0] != 6 and np_vec.shape[1] != 6):
            raise ValueError()

        so3, tht = SO3.exp_theta(np_vec[3:6])
        omega = SO3.hat(np_vec[3:6])
        omega_sq = omega @ omega
        if tht < 1e-5:
            V = so3.matrix()
        else:
            tht_sq = tht * tht
            V = np.eye(3) + (1-np.cos(tht)) / tht_sq * omega + (tht - np.sin(tht))/ (tht_sq * tht) * omega_sq
        return SE3(so3, V@np_vec[:3])

    def log(self):
        rot_log, theta = self.so3().log_theta()
        
        Omega = SO3.hat(rot_log)
        if abs(theta) < 1e-5:
            V_inv = np.eye(3) - 0.5 * Omega + 1/12 * Omega @ Omega
            trans_log = V_inv @ self.translation()
        else:
            V_inv = np.eye(3) - 0.5 * Omega + (1 - theta / (2 * np.tan(theta / 2))) / (theta * theta) * Omega @ Omega
            trans_log = V_inv @ self.translation()

        return np.hstack([trans_log, rot_log])

    def so3(self):
        return self.rot

    def rotation_matrix(self):
        return self.rot.matrix()
    
    def translation(self):
        return self.trans

    def __repr__(self):
        log_ = self.log()
        return "SE3(%.4f, %.4f, %.4f, %.4f, %.4f, %.4f)" % (log_[0], log_[1], log_[2], log_[3], log_[4], log_[5])

    def __str__(self):
        log_ = self.log()
        return str(log_)