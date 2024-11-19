import numpy as np
import random
from scipy.optimize import fsolve
from scipy.spatial.transform import Rotation as Rfunc

def calculate(cam_params,ori2d):
    num_ori = ori2d.shape[0]
    res = []
    for i in range(num_ori):
        for j in range(i+1,num_ori):
            f = cam_params[0][6]
            cx = cam_params[0][7]
            cy = cam_params[0][8]
            K = np.array([[f,0,cx,0],[0,f,cy,0],[0,0,1,0]])
            euler_i = cam_params[i][0:3]
            tx_i, ty_i, tz_i = cam_params[i][3], cam_params[i][4], cam_params[i][5]
            R_i = Rfunc.from_euler('xyz',euler_i,degrees=False).as_matrix()
            euler_j = cam_params[j][0:3]
            tx_j, ty_j, tz_j = cam_params[j][3], cam_params[j][4], cam_params[j][5]
            R_j = Rfunc.from_euler('xyz',euler_j,degrees=False).as_matrix()
            Rt_i = np.zeros((4,4))
            Rt_i[0:3,0:3] = R_i
            Rt_i[0,3] = tx_i
            Rt_i[1,3] = ty_i
            Rt_i[2,3] = tz_i
            Rt_i[3,3] = 1
            Rt_j = np.zeros((4,4))
            Rt_j[0:3,0:3] = R_j
            Rt_j[0,3] = tx_j
            Rt_j[1,3] = ty_j
            Rt_j[2,3] = tz_j
            Rt_j[3,3] = 1

            theta_i = ori2d[i]
            theta_j = ori2d[j]

            A11 = np.dot(K[0],Rt_i[:,0])
            A21 = np.dot(K[0],Rt_i[:,1])
            A31 = np.dot(K[0],Rt_i[:,2])
            A51 = np.dot(K[1],Rt_i[:,0])
            A61 = np.dot(K[1],Rt_i[:,1])
            A71 = np.dot(K[1],Rt_i[:,2])

            A12 = np.dot(K[0],Rt_j[:,0])
            A22 = np.dot(K[0],Rt_j[:,1])
            A32 = np.dot(K[0],Rt_j[:,2])
            A52 = np.dot(K[1],Rt_j[:,0])
            A62 = np.dot(K[1],Rt_j[:,1])
            A72 = np.dot(K[1],Rt_j[:,2])

            def func(variables):
                ox, oy, oz = variables[0], variables[1], variables[2]
                return [
                np.arctan2((A51*ox+A61*oy+A71*oz),(A11*ox+A21*oy+A31*oz))-theta_i,
                np.arctan2((A52*ox+A62*oy+A72*oz),(A12*ox+A22*oy+A32*oz))-theta_j,
                ox**2+oy**2+oz**2 - 1
                ]
            
            res.append(fsolve(func,[1,0,0]))

    return np.array(res).mean(axis=0)


def estimate_ori3d(camera_params,camera_indices,point3d_indices,num_point3d,ori_2d):
    ori_3d = []
    for i in range(num_point3d):
        idx = np.where(point3d_indices==i)[0]
        cam_params_i = camera_params[camera_indices[idx]]
        ori2d_i = ori_2d[idx]
        ori3d_i = calculate(cam_params_i,ori2d_i)
        ori_3d.append(ori3d_i)
    ori_3d = np.array(ori_3d)
    norm = np.linalg.norm(ori_3d,axis=1)
    return ori_3d/np.stack((norm,norm,norm),axis=1)