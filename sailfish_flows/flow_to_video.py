
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
from mpl_toolkits.mplot3d import Axes3D
import glob
from tqdm import *

import math


# video init
fourcc = cv2.cv.CV_FOURCC('m', 'p', '4', 'v') 
video = cv2.VideoWriter()

# get list of frames
files = glob.glob("test_flow.0.*")
files.sort()

b = np.load(files[0])
shape = b.f.rho.shape

# make video
success = video.open('flow_video.mov', fourcc, 8, ((shape[1]-2), (shape[0]-2)), True)

# get list of frames
files = glob.glob("test_flow.0.*")
files.sort()

# generate video
for f in files:
  x = np.load(f)
  if len(x.f.v.shape) == 4:
    frame = np.sqrt(np.square(x.f.v[0,1:-1,shape[2]/2,1:-1]) + np.square(x.f.v[1,1:-1,shape[2]/2,1:-1]) + np.square(x.f.v[2,1:-1,shape[2]/2,1:-1]))
    #frame = x.f.rho[:,shape[2]/2,:]
    frame[np.where(np.isnan(frame))] = 1.0
    frame = frame - .5
  elif len(x.f.v.shape) == 3:
    frame = np.sqrt(np.square(x.f.v[0,:,:]) + np.square(x.f.v[1,:,:]))

  frame = np.nan_to_num(frame)
  print(np.max(frame))
  frame = np.uint8(255 * (frame - np.min(frame))/(np.max(frame) - np.min(frame)))
  frame = cv2.applyColorMap(frame, 2)
  #frame = cv2.reshape(frame, (frame.shape[1]/2, frame.shape[0]/2))

  video.write(frame)
 
# release video
video.release()
cv2.destroyAllWindows()
 
