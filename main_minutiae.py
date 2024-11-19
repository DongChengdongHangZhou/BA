import numpy as np
from scipy.spatial.transform import Rotation as Rfunc
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares
import open3d as o3d
from compute_angle3d_ransac import estimate_ori3d

def convert_oxoyoz_to_azi_ele(oxoyoz):
    '''
    using azimuthal angle and elevational angle to represent o_x, o_y, o_z
    '''
    ori_3d_azi_ele = np.zeros((oxoyoz.shape[0],2)) 
    ori_3d_azi_ele[:,0] = np.arctan2(oxoyoz[:,0],oxoyoz[:,1])
    ori_3d_azi_ele[:,1] = np.arcsin(oxoyoz[:,2])
    return ori_3d_azi_ele

def convert_azi_ele_to_oxoyoz(azi_ele):
    '''
    using o_x, o_y, o_z to represent azimuthal angle and elevational angle
    '''
    oxoyoz = np.zeros((azi_ele.shape[0],3))
    oxoyoz[:,2] = np.sin(azi_ele[:,1])
    oxoyoz[:,0] = np.sqrt(np.ones_like(oxoyoz[:,2]) - oxoyoz[:,2]**2) * np.sin(azi_ele[:,0])
    oxoyoz[:,1] = np.sqrt(np.ones_like(oxoyoz[:,2]) - oxoyoz[:,2]**2) * np.cos(azi_ele[:,0])
    return oxoyoz

def load_params(camera_path,images_path,points3d_path,minutiae_root_path):
    file_points3d = open(points3d_path, 'r')
    num_point3d = 0
    points_3d = []
    with open(points3d_path, 'r') as file:
        for line_num, line in enumerate(file_points3d, start=1):
            if '#' in line:
                continue
            num_point3d = num_point3d + 1
            data = line.replace("\n","")
            data = data.split(' ')
            points_3d.append([int(data[0]),float(data[1]),float(data[2]),float(data[3])])
    points_3d = np.array(points_3d)

    file_camera = open(camera_path, 'r')
    camera_params = []
    with open(camera_path, 'r') as file:
        for line_num, line in enumerate(file_camera, start=1):
            if 'SIMPLE_PINHOLE' in line:
                break
    line = line.replace("\n","")
    line = line.split(' ')
    f = float(line[4])
    cx = int(line[5])
    cy = int(line[6])

    num_camera = 0
    points_2d = []
    ori_2d = []
    camera_indices = []
    point3d_indices = []
    with open(images_path, 'r') as file:
        for line_num, line in enumerate(file, start=1):
            if ('#' in line)==False:
                data = line.replace("\n","")
                data = data.split(' ')
                if ('jpg' in line) or ('jpeg' in line) or ('png' in line) or ('bmp' in line) or ('tiff' in line):
                    img_id, qw, qx, qy, qz, tx, ty, tz = int(data[0]), float(data[1]), float(data[2]), float(data[3]), float(data[4]), float(data[5]), float(data[6]), float(data[7])
                    euler = Rfunc.from_quat(np.array([qx,qy,qz,qw])).as_euler('xyz', degrees=False)
                    param = np.hstack((euler,np.array([tx,ty,tz]),np.array([f,cx,cy])))
                    camera_params.append(param)
                    num_camera = num_camera + 1
                else:
                    num_point2d = int(len(data)/3)
                    ori2d_temp = np.load(minutiae_root_path+str(img_id)+'.npy')[:,2:4]
                    for i in range(num_point2d):
                        if int(data[i*3+2])!= -1:
                            points_2d.append([int(float(data[i*3])-0.5),int(float(data[i*3+1])-0.5)])
                            ori_2d.append(ori2d_temp[i])
                            camera_indices.append(img_id)
                            point3d_indices.append(np.where(points_3d[:,0]==int(data[i*3+2]))[0][0])
    
    points_3d = points_3d[:,1:4]
    camera_params = np.array(camera_params)
   
    points_2d = np.array(points_2d)
    ori_2d = np.array(ori_2d)
    ori_2d[:, [1, 0]] = ori_2d[:, [0, 1]] # swap the two columns since the $\theta_x$, $\theta_y$ stored in numpy is reversed
    ori_2d = np.arctan2(ori_2d[:,0],ori_2d[:,1])
    
    point3d_indices = np.array(point3d_indices)
    camera_indices = np.array(camera_indices) - 1
    ori_3d = estimate_ori3d(camera_params,camera_indices,point3d_indices,num_point3d,ori_2d)

    ori_3d_azi_ele = convert_oxoyoz_to_azi_ele(ori_3d)
    points_3d = np.concatenate((points_3d,ori_3d_azi_ele),axis=1)
    x0 = np.hstack((camera_params.ravel(),points_3d.ravel()))
    points_2d = np.concatenate((points_2d,ori_2d[:,np.newaxis]),axis=1)
    return x0, camera_params, camera_indices, points_3d, points_2d, camera_indices, point3d_indices, num_point3d, num_point2d, ori_2d

def project(points, cam_params_pointwise):
    assert points.shape[1] == 3
    assert cam_params_pointwise.shape[1] == 9
    assert cam_params_pointwise.shape[0] == points.shape[0]
    euler = cam_params_pointwise[:,0:3]
    tx, ty, tz = cam_params_pointwise[:,3], cam_params_pointwise[:,4], cam_params_pointwise[:,5]
    f, cx, cy = cam_params_pointwise[:,6], cam_params_pointwise[:,7], cam_params_pointwise[:,8]
    K = np.zeros((points.shape[0],3,4))
    K[:,0,0] = f
    K[:,0,2] = cx
    K[:,1,1] = f
    K[:,1,2] = cy
    K[:,2,2] = 1
    R = Rfunc.from_euler('xyz',euler,degrees=False).as_matrix()
    Rt = np.zeros((points.shape[0],4,4))
    Rt[:,0:3,0:3] = R
    Rt[:,0,3] = tx
    Rt[:,1,3] = ty
    Rt[:,2,3] = tz
    Rt[:,3,3] = 1
    KRt = np.matmul(K,Rt)
    points = np.expand_dims((np.concatenate((points,np.ones((points.shape[0],1))),axis=1)),axis=2)
    points_proj = np.matmul(KRt,points)
    points_proj = points_proj[:,:,0]
    points_proj[:,0] = points_proj[:,0]/points_proj[:,2]
    points_proj[:,1] = points_proj[:,1]/points_proj[:,2]
    points_proj = points_proj[:,0:2]
    return points_proj

def project_ori(directions,cam_params_pointwise):
    assert directions.shape[1] == 3
    assert cam_params_pointwise.shape[1] == 9
    assert cam_params_pointwise.shape[0] == directions.shape[0]
    euler = cam_params_pointwise[:,0:3]
    tx, ty, tz = cam_params_pointwise[:,3], cam_params_pointwise[:,4], cam_params_pointwise[:,5]
    f, cx, cy = cam_params_pointwise[:,6], cam_params_pointwise[:,7], cam_params_pointwise[:,8]
    K = np.zeros((directions.shape[0],3,4))
    K[:,0,0] = f
    K[:,0,2] = cx
    K[:,1,1] = f
    K[:,1,2] = cy
    K[:,2,2] = 1
    R = Rfunc.from_euler('xyz',euler,degrees=False).as_matrix()
    Rt = np.zeros((directions.shape[0],4,4))
    Rt[:,0:3,0:3] = R
    Rt[:,0,3] = tx
    Rt[:,1,3] = ty
    Rt[:,2,3] = tz
    Rt[:,3,3] = 1
    A1 = np.matmul(K[:,0,:][:,np.newaxis,:],Rt[:,:,0][:,:,np.newaxis])[:,0,0]
    A2 = np.matmul(K[:,0,:][:,np.newaxis,:],Rt[:,:,1][:,:,np.newaxis])[:,0,0]
    A3 = np.matmul(K[:,0,:][:,np.newaxis,:],Rt[:,:,2][:,:,np.newaxis])[:,0,0]
    A5 = np.matmul(K[:,1,:][:,np.newaxis,:],Rt[:,:,0][:,:,np.newaxis])[:,0,0]
    A6 = np.matmul(K[:,1,:][:,np.newaxis,:],Rt[:,:,1][:,:,np.newaxis])[:,0,0]
    A7 = np.matmul(K[:,1,:][:,np.newaxis,:],Rt[:,:,2][:,:,np.newaxis])[:,0,0]
    theta_cos = (A1*directions[:,0]+A2*directions[:,1]+A3*directions[:,2])/np.sqrt((A1*directions[:,0]+A2*directions[:,1]+A3*directions[:,2])**2+(A5*directions[:,0]+A6*directions[:,1]+A7*directions[:,2])**2)
    theta_sin = (A5*directions[:,0]+A6*directions[:,1]+A7*directions[:,2])/np.sqrt((A1*directions[:,0]+A2*directions[:,1]+A3*directions[:,2])**2+(A5*directions[:,0]+A6*directions[:,1]+A7*directions[:,2])**2)
    return np.stack((theta_cos,theta_sin),axis=1)


def fun(params_obj, n_cameras, n_points, camera_indices, point3d_indices, points_2d):
    lambda_ori = 10
    camera_params = params_obj[:n_cameras * 9].reshape((n_cameras, 9))
    points_3d = params_obj[n_cameras * 9:].reshape((n_points, 5))
    position_3d = points_3d[:,0:3]
    position_2d = points_2d[:,0:2]
    ori_3d_azi_ele = points_3d[:,3:5]

    orientation_3d = convert_azi_ele_to_oxoyoz(ori_3d_azi_ele)
    
    orientation_2d = points_2d[:,2:3]
    orientation_2d_cos, orientation_2d_sin = np.cos(orientation_2d), np.sin(orientation_2d)
    ori2d_gt = np.concatenate((orientation_2d_cos, orientation_2d_sin),axis=1)
    points_proj = project(position_3d[point3d_indices], camera_params[camera_indices])
    residual_position = points_proj - position_2d
    ori_proj = project_ori(orientation_3d[point3d_indices], camera_params[camera_indices])

    residual_ori = lambda_ori*(1 - np.matmul(ori_proj[:,np.newaxis,:],ori2d_gt[:,:,np.newaxis]))[:,:,0]
    residual = np.concatenate((residual_position,residual_ori),axis=1)
    return residual.ravel()

def bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices):
    m = camera_indices.size * 3 # number of elements of residual (the raveled dimension of the returned data of fun())
    n = n_cameras * 9 + n_points * 5
    A = lil_matrix((m, n), dtype=int)

    i = np.arange(camera_indices.size)
    for s in range(9):
        A[3 * i, camera_indices * 9 + s] = 1
        A[3 * i + 1, camera_indices * 9 + s] = 1
        A[3 * i + 2, camera_indices * 9 + s] = 1

    for s in range(5):
        A[3 * i, n_cameras * 9 + point_indices * 5 + s] = 1
        A[3 * i + 1, n_cameras * 9 + point_indices * 5 + s] = 1
        A[3 * i + 2, n_cameras * 9 + point_indices * 5 + s] = 1

    return A

def visualize(mat,normals):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(mat)
    point_cloud.normals = o3d.utility.Vector3dVector(normals)

    o3d.visualization.draw_geometries([point_cloud])

if __name__ == '__main__':
    minutiae_root_path = './minutiae_numpy/1_'
    camera_path,images_path,points3d_path = 'data_corr/cameras.txt','data_corr/images.txt','data_corr/points3D.txt'
    x0, camera_params, camera_indices, points_3d, points_2d, camera_indices, point3d_indices, num_point3d, num_point2d, ori_2d = load_params(camera_path,images_path,points3d_path,minutiae_root_path)
    
    position_3d_initial = points_3d[:,0:3]
    orientation_3d_azi_ele_initial = points_3d[:,3:5]

    orientation_3d_initial = convert_azi_ele_to_oxoyoz(orientation_3d_azi_ele_initial)

    visualize(position_3d_initial,orientation_3d_initial)
    
    num_cameras = camera_params.shape[0]
    num_paramters = 9 * num_cameras + 5 * num_point3d
    num_residuals = 3 * num_point2d
    f0 = fun(x0, num_cameras, num_point3d, camera_indices, point3d_indices, points_2d)
    plt.plot(f0)
    A = bundle_adjustment_sparsity(num_cameras, num_point3d, camera_indices, point3d_indices)

    res = least_squares(fun, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-4, method='trf',
                        args=(num_cameras, num_point3d, camera_indices, point3d_indices, points_2d))

    optimized_result = res.x                    
    points_3d_optimized = optimized_result[num_cameras * 9:].reshape((num_point3d, 5))
    position_3d_optimized = points_3d_optimized[:,0:3]
    orientation_3d_azi_ele_optimized = points_3d_optimized[:,3:5]

    orientation_3d_optimized = convert_azi_ele_to_oxoyoz(orientation_3d_azi_ele_optimized)

    visualize(position_3d_optimized,orientation_3d_optimized)

    plt.plot(res.fun)
    plt.show()

