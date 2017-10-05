
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from lxml import etree
import glob
from tqdm import *
import sys
import os.path
import gc
import skfmm
import time

from Queue import Queue
import threading

class Boundary_data:
  def __init__(self, base_dir, size, dim, num_params, train_test_split=.8, max_queue=200, nr_threads=2):

    # base dir where all the xml files are
    self.base_dir = base_dir
    self.size = size
    self.dim = dim
    self.num_params = num_params

    # lists to store the datasets
    self.geometries    = []

    # train vs test split (numbers under this value are in train, over in test)
    self.train_test_split = train_test_split
    self.split_line = 0

    # place in test set
    self.test_set_pos = 0

    # make queue
    self.max_queue = max_queue
    self.queue = Queue() # to stop halting when putting on the queue
    self.queue_batches = []

    # Start threads
    for i in xrange(nr_threads):
      get_thread = threading.Thread(target=self.data_worker)
      get_thread.daemon = True
      get_thread.start()

  def data_worker(self):
    while True:
      data_file = self.queue.get()

      # load data file
      try: 
        data_file = np.load(data_file)
        param_array = data_file.f.params
        geometry_array = data_file.f.wing

        # add to que
        self.queue_batches.append((param_array, geometry_array))
        self.queue.task_done()
      except:
        self.queue.task_done()
        
  
  def parse_data(self): 
    # reads in all xml data into lists

    # get list of all xml file in dataset
    files = glob.glob((self.base_dir + "wingfoils_boundary_learn/" + str(self.dim) + "/"
                     + str(self.size).zfill(4) + "/" 
                     + str(self.num_params).zfill(4) + "/*"))

    print("loading dataset")
    for f in tqdm(files):
      # store name
      self.geometries.append(f)

    self.split_line = int(self.train_test_split * len(self.geometries))
    print("finished parsing " + str(len(self.geometries)) + " data points")
    self.test_set_pos = self.split_line

  def minibatch(self, train=True, batch_size=32, signed_distance_function=False):

    for i in xrange(self.max_queue - len(self.queue_batches) - self.queue.qsize()):
      if train:
        sample = np.random.randint(0, self.split_line)
      else:
        sample = self.test_set_pos 
        self.test_set_pos += 1
      if (0 <= sample) & (len(self.geometries) > sample):
        self.queue.put(self.geometries[sample])
   
    while len(self.queue_batches) < batch_size:
      print(self.queue.qsize())
      print("spending time waiting for queue, consider makein more threads") 
      time.sleep(1.0)

    batch_param = []
    batch_boundary = []
    for i in xrange(batch_size): 
      batch_param.append(self.queue_batches[0][0].astype(np.float32))
      batch_boundary.append(self.queue_batches[0][1].astype(np.float32))
      self.queue_batches.pop(0)
    batch_param = np.stack(batch_param, axis=0)
    batch_boundary = np.stack(batch_boundary, axis=0)

    return batch_param, batch_boundary

"""
#dataset = Sailfish_data("../../data/", size=32, dim=3)
dataset = Boundary_data("../../data/", size=256, dim=2, num_params=46)
dataset.parse_data()
for i in xrange(100):
  batch_params, batch_boundary = dataset.minibatch(batch_size=100)
  print(batch_boundary.shape)
  plt.imshow(batch_boundary[0,:,:,0])
  plt.show()
  #time.sleep(.4)
"""
