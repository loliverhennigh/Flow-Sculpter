
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

def binomial(top, bottom):
  coef = (math.factorial(top))/(math.factorial(bottom)*math.factorial(top-bottom))
  return coef
 
def rotateImage(image, angle):
  center=tuple(np.array(image.shape[0:2])/2)
  angle = np.degrees(angle)
  rot_mat = cv2.getRotationMatrix2D(center,angle,1.0)
  return cv2.warpAffine(image, rot_mat, image.shape[0:2],flags=cv2.INTER_NEAREST)

def get_params_range(nr_params, dims):
  if dims == 2:
    params_range_lower = np.array([-1.0, 0.0, 0.0] + (nr_params-4)*[0.0] + [-0.1] )
    params_range_upper = np.array([ 1.0, 1.0, 2.0] + (nr_params-4)*[0.3] + [ 0.1] )
  elif dims == 3:
    params_range_lower = np.array([-1.0, -1.0, 0.0, 0.0, 0.0, 0.0] 
                                + (nr_params-7)*[0.0] + [-0.1] )
    params_range_upper = np.array([ 1.0,  1.0, 1.0, 2.0, 1.0, 2.0]
                                + (nr_params-7)*[0.3] + [ 0.1] )
  return params_range_lower, params_range_upper
 
def get_random_params(nr_params, dims):
  params = np.random.rand((nr_params))
  params_range_lower, params_range_upper = get_params_range(nr_params, dims)
  params_range_upper = params_range_upper - params_range_lower
  params = (params * params_range_upper) + params_range_lower
  return params

def wing_boundary_2d(angle, N_1, N_2, A_1, A_2, d_t, shape, boundary=None):

  # make lines for upper and lower wing profile
  if boundary is None:
    boundary = np.zeros(shape)
  c = 1.0 
  x_1 = np.arange(0.0, 1.00, 1.0/(shape[0]))
  x_2 = np.arange(0.0, 1.00, 1.0/(shape[0]))
  phi_1 = x_1/c
  phi_2 = x_2/c
  y_1 = np.power(phi_1, 0.5)*np.power(1.0-phi_1, 1.0)
  y_2 = np.power(phi_2, 0.5)*np.power(1.0-phi_2, 1.0)
  y_1_store = 0.0
  y_2_store = 0.0
  for i in xrange(len(A_1)):
    y_1_store += A_1[i]*binomial(len(A_1), i)*(phi_1**i)*((1.0-phi_1)**(len(A_1)-i))
    y_2_store += A_2[i]*binomial(len(A_2), i)*(phi_2**i)*((1.0-phi_2)**(len(A_2)-i))
  y_1 = y_1*y_1_store
  y_2 = y_2*y_2_store
  y_2 = - y_2

  for i in xrange(len(x_1)):
    y_upper = int(np.max(y_1[i]) * shape[1] + shape[1]/2)
    y_lower = int(np.min(y_2[i]) * shape[1] + shape[1]/2)
    x_pos = int(x_1[i] * shape[0])
    if x_pos >= shape[0]:
      continue
    boundary[y_lower:y_upper, x_pos] = 1.0

  boundary = rotateImage(boundary, angle)
  boundary = boundary.reshape(shape + [1])

  return boundary

def wing_boundary_batch(nr_params, batch_size, shape, dims):

  boundary_batch = []
  input_batch = []
  for i in xrange(batch_size): 
    params = get_random_params(nr_params, dims)
    if dims == 2:
      boundary_batch.append(wing_boundary_2d(params[0], params[1], params[2],
                                             params[3:(nr_params-4)/2],
                                             params[(nr_params-4)/2:-1],
                                             params[-1], shape))
    #elif dims == 3:
      #boundary_batch.append(wing_boundary_3d(params[0], params[1], params[2],
      #                                       params[3], params[4], params[5],
      #                                       params[6:(nr_params-7)/2],
      #                                       params[(nr_params-4)/2:-1],
      #                                       params[-1], shape))
    input_batch.append(params)
  boundary_batch = np.stack(boundary_batch, axis=0)
  input_batch = np.stack(input_batch, axis=0)
  return input_batch, boundary_batch

"""
_, boundary_batch = wing_boundary_batch_2d(82, 32, [128,128])
for i in xrange(32):
  plt.imshow(boundary_batch[i,:,:,0])
  plt.show()
"""
