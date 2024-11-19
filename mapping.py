import numpy as np
from scipy.spatial.transform import Rotation as Rfunc

'''
[u,v,1].T = K*Rt*Pw
K is the intrinsic parameter of camera;
Rt is the 4*4 matrix including rotation matrix R and translation vector t;
Pw is the coordinates of the 3D points;
After calculate K*Rt*Pw, we get [u',v',z'].T
u = u'/z', v = v'/z'
'''

def construct_R(w,x,y,z,tx,ty,tz):
    R = Rfunc.from_quat(np.array([x,y,z,w])).as_matrix() # R = np.array([[1-2*y*y-2*z*z,2*x*y-2*z*w,2*x*z+2*y*w],[2*x*y+2*z*w,1-2*x*x-2*z*z,2*y*z-2*x*w],[2*x*z-2*y*w,2*y*z+2*x*w,1-2*x*x-2*y*y]])
    Rt = np.zeros((4,4))
    Rt[0:3,0:3] = R
    Rt[3][3] = 1
    Rt[0:3,3] = np.array([tx,ty,tz]).T
    return Rt

if __name__ == '__main__':
    fx = 4024.6173915838999 # from camera.txt
    fy = fx # for simple pinhole model, fx=fy
    cx = 240 # from camera.txt
    cy = 240 # from camera.txt

    '''
    w, x, y, z, tx, ty, tz are from images.txt
    '''
    w = 0.97107551155288474
    x = 0.23520406094902616
    y = -0.024380051422175673
    z = -0.033121196657761538
    tx = 0.68748114736826682
    ty = 3.3122981562976586
    tz = 1.2347818382081062
    '''
    xw, yw, zw from point3D.txt
    '''
    xw = -0.046431025423834371
    yw = 0.97079295444953084
    zw = 8.8639891731946072

    K = np.array([[fx,0,cx,0],[0,fy,cy,0],[0,0,1,0]])
    Rt = construct_R(w,x,y,z,tx,ty,tz)
    Pw = np.array([xw,yw,zw,1]).T

    res = np.dot(np.dot(K,Rt),Pw)
    print(res[0]/res[2])
    print(res[1]/res[2])