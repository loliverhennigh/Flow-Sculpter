
# construct xml dataset of objects

import os
import errno
import numpy as np
import lxml.etree as etree
from tqdm import *
import glob
import subprocess

import sys
sys.path.append('../')
import Flow_Sculpter.utils.boundary_utils as boundary_utils

import matplotlib.pyplot as plt

from Queue import Queue
import threading

# seed numpy for same rotation every run
np.random.seed(0)

# params to run
num_wingfoil_designs = 50000
base_path = os.path.abspath("../data/") + "/"
num_wingfoil_params_2d = [16, 26, 36, 46]
num_wingfoil_params_3d = [46]
#num_wingfoil_params_3d = []
#sizes_2d = [64, 128, 256]
#sizes_2d = [64]
sizes_2d = []
sizes_3d = [32]
#sizes_3d = [32, 64, 96]
nr_threads = 30

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def wing_save_worker():
  while True:
    ids, dim, size, num_wingfoil_params = queue.get()
    # xml filename
    wing_filename = (base_path + "wingfoils_boundary_learn/" + str(dim) + "/"
                 + str(size).zfill(4) + "/" 
                 + str(num_wingfoil_params).zfill(4) + "/")
    mkdir_p(wing_filename)
    wing_filename += str(ids).zfill(6)

    # make numpy boundary file
    if not os.path.isfile(wing_filename + ".npz"):
      params = boundary_utils.get_random_params(num_wingfoil_params, dim)
      if dim == 2: 
        wing = boundary_utils.wing_boundary_2d(params[0], params[1], params[2],
                                params[3:(num_wingfoil_params-4)/2],
                                params[(num_wingfoil_params-4)/2:-1],
                                params[-1], dim*[size])
      elif dim == 3: 
        wing = boundary_utils.wing_boundary_3d(params[0], params[1], params[2],
                                               params[3], params[4], params[5],
                                               params[6:(num_wingfoil_params-7)/3+6],
                                               params[(num_wingfoil_params-7)/3+6:2*(num_wingfoil_params-7)/3+6],
                                               params[2*(num_wingfoil_params-7)/3+6:-1],
                                               params[-1], dim*[size])
      wing = np.greater(wing, 0.5)
      np.savez(wing_filename, params=params, wing=wing)

    # end worker
    queue.task_done()
    
# Start Que 
queue = Queue(400)

# Start threads
for i in xrange(nr_threads):
  get_thread = threading.Thread(target=wing_save_worker)
  get_thread.daemon = True
  get_thread.start()

ids = 0
for i in tqdm(xrange(num_wingfoil_designs)):
  ids += 1
  for dim in [2,3]:
    if dim == 2:
      sizes = sizes_2d
      num_wingfoil_params = num_wingfoil_params_2d  
    elif dim == 3:
      sizes = sizes_3d
      num_wingfoil_params = num_wingfoil_params_3d
    for size in sizes:
      for num_params in num_wingfoil_params:
        # put on worker thread
        queue.put((ids, dim, size, num_params))
      
