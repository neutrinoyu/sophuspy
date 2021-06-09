import numpy as np

class Quaternion:
    """
    follow the definition in Eigen. Use Hamilton's Quaternion (e.g. ijk=−1). But memory order is xyzw.
    """
    def __init__(self, w=1, x=0, y=0, z=0):
        self.data = np.array([x,y,z,w])
    
    @property
    def x(self):
        return self.data[0]

    @property
    def y(self):
        return self.data[1]

    @property
    def z(self):
        return self.data[2]

    @property
    def w(self):
        return self.data[3]

    def copy(self):
        return Quaternion(self.w, self.x, self.y, self.z)

    def set_indentity(self):
        self.data[0:3] = 0
        self.data[3] = 1

    def norm2(self):
        return np.linalg.norm(self.data)

    def normalize(self):
        n2 = self.norm2()
        if np.isclose(n2, 0):
            return
        self.data /= n2
        
    def inverse(self):
        return Quaternion(self.w, -self.x, -self.y, -self.z)
        
    def as_matrix(self):
        mat = np.empty((3,3))
        
        tx  = 2  * self.x
        ty  = 2  * self.y
        tz  = 2  * self.z
        twx = tx * self.w
        twy = ty * self.w
        twz = tz * self.w
        txx = tx * self.x
        txy = ty * self.x
        txz = tz * self.x
        tyy = ty * self.y
        tyz = tz * self.y
        tzz = tz * self.z

        mat[0,0] = 1 - (tyy + tzz)
        mat[0,1] = txy - twz
        mat[0,2] = txz + twy
        mat[1,0] = txy + twz
        mat[1,1] = 1 - (txx + tzz)
        mat[1,2] = tyz - twx
        mat[2,0] = txz - twy
        mat[2,1] = tyz + twx
        mat[2,2] = 1 - (txx + tyy)

        return mat

    def from_two_vectors(self, v1, v2):
        npv1 = np.asarray(v1).reshape((3,))
        npv2 = np.asarray(v2).reshape((3,))

        npv1 /= np.linalg.norm(npv1)
        npv2 /= np.linalg.norm(npv2)

        cos_tht = np.dot(npv1, npv2)

        if (np.isclose(tht, -1)) {
            raise ValueError
        }
        axis = np.cross(npv1, npv2)
        s = np.sqrt((1+cos_tht) * 2)
        self[:3] = axis / s
        self.data[3] = s / 2
        self.normalize()
        
    def slerp(ratio, other):
        if not isinstance(quat, Quaternion):
            raise ValueError
        
        d = np.dot(self.data, other.data)
        if abs(d) > 1:
            s0 = 1-ratio
            s1 = ratio
        else:
            tht = acos(abs(d))
            sin_tht = sin(tht)

            s0 = sin( (1-t) * tht) / sin_tht
            s1 = sin( t * tht ) / sin_tht
        
        if d < 0:
            s1 = -s1
        
        q = self.copy()
        q.data = q.data * s0 + other.data * s1
        q.normalize()
        return q

    