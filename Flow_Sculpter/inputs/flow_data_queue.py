
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

class Sailfish_data:
  def __init__(self, base_dir, size, dim, train_test_split=.8, max_queue=20, nr_threads=2):

    # base dir where all the xml files are
    self.base_dir = base_dir
    self.size = size
    self.dim = dim

    # lists to store the datasets
    self.geometries    = []
    self.steady_flows = []

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
      geometry_file, steady_flow_file = self.queue.get()

      try:
        # load geometry file
        geometry_array = np.load(geometry_file)

        # load flow file
        steady_flow_array = np.load(steady_flow_file)
  
        # add to que
        self.queue_batches.append((geometry_array, steady_flow_array))
        self.queue.task_done()

      except:
        self.queue.task_done()
  
  def parse_data(self): 
    # reads in all xml data into lists

    # get list of all xml file in dataset
    tree = etree.parse(self.base_dir + "experiment_runs_master.xml")
    root = tree.getroot()
    run_roots = root.findall("run")

    print("loading dataset")
    for run_root in tqdm(run_roots):
      # check if right size
      xml_size = int(run_root.find("size").text)
      if xml_size != self.size:
        continue

      # check if right dim
      xml_dim = int(run_root.find("dim").text)
      if xml_dim != self.dim:
        continue

      # parse xml file
      xml_file = run_root.find("xml_filename").text
      tree = etree.parse(xml_file)
      root = tree.getroot()

      # check if flow data is availible
      is_availible = root.find("flow_data").find("availible").text
      if is_availible == "False":
        continue
      
      # get needed filenames
      geometry_file    = root.find("flow_data").find("geometry_file").text
      steady_flow_file = root.find("flow_data").find("flow_file").text

      # check file for geometry
      if not os.path.isfile(geometry_file):
        continue

      # check file for steady state flow
      if not os.path.isfile(steady_flow_file):
        continue

      # store name
      self.geometries.append(geometry_file)
      self.steady_flows.append(steady_flow_file)

    gc.collect()
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
        self.queue.put((self.geometries[sample], self.steady_flows[sample]))
   
    while len(self.queue_batches) < batch_size:
      print("spending time waiting for queue, consider makein more threads") 
      time.sleep(1.0)

    batch_boundary = []
    batch_data = []
    for i in xrange(batch_size): 
      if signed_distance_function:
        geometry_array = self.queue_batches[0][0].astype(np.float32)
        geometry_array = (-2.0*geometry_array) + 1.0
        geometry_array = skfmm.distance(geometry_array, dx=1.0)
        batch_boundary.append(geometry_array)
      else:
        batch_boundary.append(self.queue_batches[0][0].astype(np.float32))
      batch_data.append(self.queue_batches[0][1])
      self.queue_batches.pop(0)
    batch_boundary = np.stack(batch_boundary, axis=0)
    batch_data = np.stack(batch_data, axis=0)
   
    # maybe flip batch 
    """
    flip = np.random.randint(0,2)
    if flip == 1:
      batch_data = np.flip(batch_data, axis=2)
      batch_data[...,1] = -batch_data[...,1]
      batch_boundary = np.flip(batch_boundary, axis=2)
    """
   
    return batch_boundary, batch_data

"""
#dataset = Sailfish_data("../../data/", size=32, dim=3)
dataset = Sailfish_data("../../data/", size=256, dim=2)
dataset.parse_data()
for i in xrange(100):
  batch_boundary, batch_data = dataset.minibatch(batch_size=100)
  print(batch_data.shape)
  print(batch_boundary.shape)
  plt.imshow(batch_data[0,:,:,2])
  plt.show()
  plt.imshow(batch_data[0,:,:,1])
  plt.show()
  plt.imshow(batch_data[0,:,:,0])
  plt.show()
  plt.imshow(batch_boundary[0,:,:,0])
  plt.show()
  #time.sleep(.4)
"""
