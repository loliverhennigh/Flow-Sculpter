
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
from mpl_toolkits.mplot3d import Axes3D

import math

def voxel_plot(voxel):
  X = []
  Y = []
  Z = []
  for i in xrange(voxel.shape[0]-2):
    for j in xrange(voxel.shape[1]-2):
      for k in xrange(voxel.shape[2]-2):
        if voxel[i+1,j+1,k+1] > 0:
          if not(voxel[i,j+1,k+1] > 0 and voxel[i+1,j,k+1] > 0 and voxel[i+1,j+1,k] > 0 and voxel[i+2,j+1,k+1] > 0 and voxel[i+1,j+2,k+1] > 0 and voxel[i+1,j+1,k+2] > 0):
            X.append(i)
            Y.append(j)
            Z.append(k)
  X = np.array(X)
  Y = np.array(Y)
  Z = np.array(Z)
  #X_q, Y_q, Z_q = np.meshgrid(np.arange(0, voxel.shape[0], 1), np.arange(0, voxel.shape[0], 1), np.arange(nz/2, nz/2+4, 1))
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.scatter(X, Y, Z)
  ax.set_xlim3d((0,voxel.shape[0]))
  ax.set_ylim3d((0,voxel.shape[1]))
  ax.set_zlim3d((0,voxel.shape[2]))
  plt.show()

b = np.load("test_flow_boundary.npy")
voxel_plot(b)
x = np.load("test_flow_steady_flow.npz")
#x = np.load("test_flow.0.070000.npz")
plt.imshow(x.f.v[2,:,x.f.v.shape[2]/2+5,:])
#plt.imshow(x.f.v[0,:,:])
plt.show()
plt.imshow(np.nan_to_num(x.f.v[0,:,x.f.v.shape[2]/2+5,:]))
#plt.imshow(np.nan_to_num(x.f.v[0,:,:]))
plt.show()
plt.imshow(x.f.v[1,:,x.f.v.shape[2]/2+5,:])
#plt.imshow(x.f.v[1,:,:])
plt.show()
plt.imshow(b[:,x.f.v.shape[2]/2+5,:])
#plt.imshow(b[:,:])
#plt.show()
plt.imshow(x.f.rho[:,x.f.v.shape[2]/2+5,:])
#plt.imshow(x.f.rho[:,:])
plt.show()

