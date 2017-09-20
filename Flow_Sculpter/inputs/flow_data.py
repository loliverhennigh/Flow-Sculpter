
import numpy as np
import vtk
from vtk.util.numpy_support import *
import matplotlib.pyplot as plt
from lxml import etree
import glob
from tqdm import *
import sys
import os.path

class Sailfish_data:
  def __init__(self, base_dir, train_test_split=.8):

    # base dir where all the xml files are
    self.base_dir = base_dir

    # lists to store the datasets
    self.geometries    = []
    self.steady_flows = []
    self.drag_vectors = []

    # train vs test split (numbers under this value are in train, over in test)
    self.train_test_split = train_test_split
    self.split_line = 0

  def load_data(self, dim, size): 
    # reads in all xml data into lists

    # get list of all xml file in dataset
    tree = etree.parse(self.base_dir + "experiment_runs_master.xml")
    root = tree.getroot()
    run_roots = root.findall("run")

    print("loading dataset")
    reader = vtk.vtkXMLMultiBlockDataReader()
    stopper = 0
    for run_root in tqdm(run_roots):
      stopper += 1
      if stopper > 20000:
        break

      # check if right size
      xml_size = int(run_root.find("size").text)
      if xml_size != size:
        continue

      # check if right dim
      xml_dim = int(run_root.find("dim").text)
      if xml_dim != dim:
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

      # read file for geometry
      if not os.path.isfile(geometry_file):
        continue
      geometry_array = np.load(geometry_file)
      geometry_array = geometry_array.astype(np.int)
      geometry_array = np.swapaxes(geometry_array, 0, -1)
      geometry_array = geometry_array[size/2+1:5*size/2+1,1:-1]
      geometry_array = np.expand_dims(geometry_array, axis=-1)

      # read file for steady state flow
      if not os.path.isfile(steady_flow_file):
        continue
      steady_flow_array = np.load(steady_flow_file)
      velocity_array = steady_flow_array.f.v
      pressure_array = np.expand_dims(steady_flow_array.f.rho, axis=0)
      velocity_array[np.where(np.isnan(velocity_array))] = 0.0
      pressure_array[np.where(np.isnan(pressure_array))] = 1.0
      pressure_array = pressure_array - 1.0
      steady_flow_array = np.concatenate([velocity_array, pressure_array], axis=0)
      steady_flow_array = np.swapaxes(steady_flow_array, 0, -1)
      steady_flow_array = steady_flow_array[size/2:5*size/2]
      np.nan_to_num(steady_flow_array, False)

      # store
      self.geometries.append(geometry_array)
      self.steady_flows.append(steady_flow_array)

    self.split_line = int(self.train_test_split * len(self.geometries))

  def minibatch(self, train=True, batch_size=32, batch_type="flow"):
    batch_boundary = []
    batch_data = []
    for i in xrange(batch_size): 
      if train:
        sample = np.random.randint(0, self.split_line)
      else:
        sample = np.random.randint(self.split_line, len(self.geometries))
      batch_boundary.append(self.geometries[sample])
      if batch_type == "flow":
        batch_data.append(self.steady_flows[sample])
    batch_boundary = np.stack(batch_boundary, axis=0)
    if batch_type == "flow":
      batch_data = np.stack(batch_data, axis=0)
    """
    flip = np.random.randint(0,2)
    if flip == 1:
      batch_data = np.flip(batch_data, axis=1)
      batch_boundary = np.flip(batch_boundary, axis=1)
    """
    return batch_boundary, batch_data

"""
dataset = Sailfish_data("../../data/")
dataset.load_data(dim=2, size=64)
batch_boundary, batch_data = dataset.minibatch(batch_type="flow")
print(batch_boundary.shape)
print(batch_data.shape)
for i in xrange(32):
  plt.imshow(batch_data[i,:,:,2])
  plt.show()
  plt.imshow(batch_boundary[i,:,:,0])
  plt.show()
"""

