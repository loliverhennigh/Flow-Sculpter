
import numpy as np
import cv2
import matplotlib.pyplot as plt

def rotateImage(image, angle):
    center=tuple(np.array(image.shape[0:2])/2)
    angle = np.degrees(angle)
    rot_mat = cv2.getRotationMatrix2D(center,angle,1.0)
    return cv2.warpAffine(image, rot_mat, image.shape[0:2],flags=cv2.INTER_NEAREST)

def wing_boundary_2d(angle, N_1, N_2, A_1, A_2, d_t, shape):

  # make lines for upper and lower wing profile
  boundary = np.zeros(shape)
  c = 1.0 
  x_1 = np.arange(0.0, 1.00, 1.0/(shape[0]))
  x_2 = np.arange(0.0, 1.00, 1.0/(shape[0]))
  phi_1 = x_1/c
  phi_2 = x_2/c
  y_1 = np.power(phi_1, N_1)*np.power(1.-phi_1, N_2)
  y_2 = np.power(phi_2, N_1)*np.power(1.-phi_2, N_2)
  y_1_store = 0.0
  y_2_store = 0.0
  for i in xrange(len(A_1)):
    y_1_store += A_1[i]*phi_1**i
    y_2_store += A_2[i]*phi_2**i
  y_1 = y_1*y_1_store
  y_2 = y_2*y_2_store
  y_1 = y_1 + phi_1 * d_t
  y_2 = y_2 - phi_2 * d_t
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

def wing_boundary_batch_2d(length_input, batch_size, shape):

  boundary_batch = []
  input_batch = []
  for i in xrange(batch_size): 
    angle = np.random.rand(1)  - .5
    n_1 = np.random.rand(1)
    n_2 = np.random.rand(1) * 2.0
    a_1 = np.random.rand((length_input-4)/2) / 5.0
    a_2 = np.random.rand((length_input-4)/2) / 5.0
    d_t = (np.random.rand(1) -.5 ) / 5.0
    boundary_batch.append(wing_boundary_2d(angle[0], n_1[0], n_2[0], a_1, a_2, d_t[0], shape))
    input_batch.append(np.concatenate([angle, n_1, n_2, a_1, a_2, d_t], axis=0))
  boundary_batch = np.stack(boundary_batch, axis=0)
  input_batch = np.stack(input_batch, axis=0)
  
  return input_batch, boundary_batch

"""
input_batch, boundary_batch = wing_boundary_batch_2d(11, 32, [256,256])
for i in xrange(32):
  plt.imshow(boundary_batch[i,:,:,0])
  plt.show()
"""


