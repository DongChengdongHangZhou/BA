import urllib
import bz2
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
import time
from scipy.optimize import least_squares
import open3d as o3d

# import pcl
# import pcl.pcl_visualization
# import random
# https://blog.csdn.net/baidu_40840693/article/details/115554682
# def vis_pair(cloud1, cloud2, rdm=False):
#     color1 = [255, 0, 0]
#     color2 = [0, 255, 0]
#     if rdm:
#         color1 = [255, 0, 0]
#         color2 = [random.randint(0, 255) for _ in range(3)]
#     visualcolor1 = pcl.pcl_visualization.PointCloudColorHandleringCustom(cloud1, color1[0], color1[1], color1[2])
#     visualcolor2 = pcl.pcl_visualization.PointCloudColorHandleringCustom(cloud2, color2[0], color2[1], color2[2])
#     vs = pcl.pcl_visualization.PCLVisualizering
#     vss1 = pcl.pcl_visualization.PCLVisualizering()  # 初始化一个对象，这里是很重要的一步
#     vs.AddPointCloud_ColorHandler(vss1, cloud1, visualcolor1, id=b'cloud', viewport=0)
#     vs.AddPointCloud_ColorHandler(vss1, cloud2, visualcolor2, id=b'cloud1', viewport=0)
#     vs.SetBackgroundColor(vss1, 0, 0, 0)
#     #vs.InitCameraParameters(vss1)
#     #vs.SetFullScreen(vss1, True)
#     # v = True
#     while not vs.WasStopped(vss1):
#         vs.Spin(vss1)


def read_bal_data(file_name):
    with bz2.open(file_name, "rt") as file:
        n_cameras, n_points, n_observations = map(
            int, file.readline().split())

        camera_indices = np.empty(n_observations, dtype=int)
        point_indices = np.empty(n_observations, dtype=int)
        points_2d = np.empty((n_observations, 2))

        for i in range(n_observations):
            camera_index, point_index, x, y = file.readline().split()
            camera_indices[i] = int(camera_index)
            point_indices[i] = int(point_index)
            points_2d[i] = [float(x), float(y)]

        camera_params = np.empty(n_cameras * 9)
        for i in range(n_cameras * 9):
            camera_params[i] = float(file.readline())
        camera_params = camera_params.reshape((n_cameras, -1))

        points_3d = np.empty(n_points * 3)
        for i in range(n_points * 3):
            points_3d[i] = float(file.readline())
        points_3d = points_3d.reshape((n_points, -1))

    return camera_params, points_3d, camera_indices, point_indices, points_2d

def rotate(points, rot_vecs):
    """Rotate points by given rotation vectors.
    
    Rodrigues' rotation formula is used.
    """
    theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
    with np.errstate(invalid='ignore'):
        v = rot_vecs / theta
        v = np.nan_to_num(v)
    dot = np.sum(points * v, axis=1)[:, np.newaxis]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    return cos_theta * points + sin_theta * np.cross(v, points) + dot * (1 - cos_theta) * v

def project(points, camera_params):
    """Convert 3-D points to 2-D by projecting onto images."""
    points_proj = rotate(points, camera_params[:, :3])
    points_proj += camera_params[:, 3:6]
    points_proj = -points_proj[:, :2] / points_proj[:, 2, np.newaxis]
    f = camera_params[:, 6]
    k1 = camera_params[:, 7]
    k2 = camera_params[:, 8]
    n = np.sum(points_proj**2, axis=1)
    r = 1 + k1 * n + k2 * n**2
    points_proj *= (r * f)[:, np.newaxis]
    return points_proj

def fun(params, n_cameras, n_points, camera_indices, point_indices, points_2d):
    """Compute residuals.
    
    `params` contains camera parameters and 3-D coordinates.
    """
    camera_params = params[:n_cameras * 9].reshape((n_cameras, 9))
    points_3d = params[n_cameras * 9:].reshape((n_points, 3))
    points_proj = project(points_3d[point_indices], camera_params[camera_indices])
    return (points_proj - points_2d).ravel()

def bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices):
    m = camera_indices.size * 2 # number of elements of residual (the raveled dimension of the returned data of fun())
    n = n_cameras * 9 + n_points * 3
    A = lil_matrix((m, n), dtype=int)

    i = np.arange(camera_indices.size)
    for s in range(9):
        A[2 * i, camera_indices * 9 + s] = 1
        A[2 * i + 1, camera_indices * 9 + s] = 1

    for s in range(3):
        A[2 * i, n_cameras * 9 + point_indices * 3 + s] = 1
        A[2 * i + 1, n_cameras * 9 + point_indices * 3 + s] = 1

    return A

def visualize(mat):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(mat)
    o3d.visualization.draw_geometries([point_cloud])

if __name__ == '__main__':
    FILE_NAME = "problem-49-7776-pre.txt.bz2"
    camera_params, points_3d, camera_indices, point_indices, points_2d = read_bal_data(FILE_NAME)

    n_cameras = camera_params.shape[0]
    n_points = points_3d.shape[0]

    n = 9 * n_cameras + 3 * n_points
    m = 2 * points_2d.shape[0]

    print("n_cameras: {}".format(n_cameras))
    print("n_points: {}".format(n_points))
    print("Total number of parameters: {}".format(n))
    print("Total number of residuals: {}".format(m))

    visualize(points_3d)
    x0 = np.hstack((camera_params.ravel(), points_3d.ravel()))
    f0 = fun(x0, n_cameras, n_points, camera_indices, point_indices, points_2d)
    plt.plot(f0)
    A = bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices)

    res = least_squares(fun, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-4, method='trf',
                        args=(n_cameras, n_points, camera_indices, point_indices, points_2d))

    plt.plot(res.fun)
    plt.show()