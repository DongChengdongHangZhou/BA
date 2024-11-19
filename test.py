import numpy as np

data = np.load('minutiae_numpy/1_1.npy')
print(data)

from scipy.spatial.transform import Rotation as Rfunc

# np.random.seed(0)

def rotation_matrix2euler(rotation_matrix):
    r = Rfunc.from_matrix(rotation_matrix)
    euler = r.as_euler('xyz', degrees=False)
    return euler

w = 0.97107551155288474
x = 0.23520406094902616
y = -0.024380051422175673
z = -0.033121196657761538
r = Rfunc.from_quat(np.array([x,y,z,w]))
euler = r.as_euler('xyz', degrees=False)
rot_matrix = r.as_matrix()

r2 = Rfunc.from_euler('xyz',np.stack((euler,euler),axis=0),degrees=False)
rot_matrix2 = r2.as_matrix()
print(rot_matrix2 - rot_matrix)

