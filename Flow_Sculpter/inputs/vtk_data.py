
import numpy as np
import vtk
from vtk.util.numpy_support import *
import matplotlib.pyplot as plt
from lxml import etree
import glob
from tqdm import *
import sys
sys.setrecursionlimit(100000)

class VTK_data:
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
    #stopper = 0
    for run_root in tqdm(run_roots):
      #stopper += 1
      #if stopper > 10000:
      #  break

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
      drag_vector_file = root.find("flow_data").find("drag_file").text

      # read file for geometry
      reader.SetFileName(geometry_file)
      reader.Update()
      data = reader.GetOutput()
      data_iterator = data.NewIterator()
      img_data = data_iterator.GetCurrentDataObject() 
      img_data.Update()
      point_data = img_data.GetPointData()
      array_data = point_data.GetArray(0)
      np_array = vtk_to_numpy(array_data)
      img_shape = img_data.GetWholeExtent()
      if img_shape[-1] == 0:
        np_shape = [img_shape[3] - img_shape[2] + 1, img_shape[1] - img_shape[0] + 1, 1]
      else:
        np_shape = [img_shape[5] - img_shape[4] + 1, img_shape[3] - img_shape[2] + 1, img_shape[1] - img_shape[0] + 1, 1]
      geometry_array = np_array.reshape(np_shape)
      geometry_array = geometry_array[...,0:np_shape[0],:]
      geometry_array = np.abs(geometry_array - 1.0)
      geometry_array = np.minimum(geometry_array, 1.0)
      geometry_array = np.abs(np.abs(geometry_array - 1.0) - 1.0)
      if img_shape[-1] == 0:
        geometry_array = fill_gaps(geometry_array, [size/2,size/2], size)
      geometry_array = 2.0*geometry_array - 1.0
      if np.isnan(geometry_array).any():
        continue

      # read file for steady state flow
      #reader = vtk.vtkXMLMultiBlockDataReader()
      reader.SetFileName(steady_flow_file)
      reader.Update()
      data = reader.GetOutput()
      data_iterator = data.NewIterator()
      img_data = data_iterator.GetCurrentDataObject() 
      img_data.Update()
      point_data = img_data.GetPointData()
      velocity_array_data = point_data.GetArray(0)
      pressure_array_data = point_data.GetArray(1)
      velocity_np_array = vtk_to_numpy(velocity_array_data)
      pressure_np_array = vtk_to_numpy(pressure_array_data)
      img_shape = img_data.GetWholeExtent()
      if img_shape[-1] == 0:
        np_shape = [img_shape[3] - img_shape[2] + 1, img_shape[1] - img_shape[0] + 1]
        velocity_np_shape = np_shape + [2]
        pressure_np_shape = np_shape + [1]
      else:
        np_shape = [img_shape[5] - img_shape[4] + 1, img_shape[3] - img_shape[2] + 1, img_shape[1] - img_shape[0] + 1]
        velocity_np_shape = np_shape + [3]
        pressure_np_shape = np_shape + [1]
      velocity_np_array = velocity_np_array.reshape(velocity_np_shape)
      pressure_np_array = pressure_np_array.reshape(pressure_np_shape)
      steady_flow_array = np.concatenate([velocity_np_array, pressure_np_array], axis=2)
      steady_flow_array = steady_flow_array[...,0:np_shape[0],:]
      if np.isnan(steady_flow_array).any():
        continue

      # read file for drag vector
      csv_reader = open(drag_vector_file, "r")
      drag_values = csv_reader.readlines()
      drag_array = np.zeros((len(drag_values)))
      for i in xrange(len(drag_values)):
        values = drag_values[i].split(' ')
        drag_array[i] = float(values[1])
      if np.isnan(drag_array).any():
        continue
      csv_reader.close()

      # if no nans then store
      self.geometries.append(geometry_array)
      self.steady_flows.append(steady_flow_array)
      self.drag_vectors.append(drag_array)

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
      elif batch_type == "drag":
        batch_data.append(self.drag_vectors[sample])
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

def fill_gaps(matrix, pos, radius):
  sub_matrix = matrix[pos[0]:radius+pos[0]+2,pos[1]:radius+pos[1]+2,0]
  floodfill(sub_matrix, 0, 0)
  sub_matrix = np.minimum(sub_matrix+1.0, 1.0)

  matrix[pos[0]:radius+pos[0]+2,pos[1]:radius+pos[1]+2,0] = sub_matrix
  #plt.imshow(matrix[:,:,0])
  #plt.show()
  return matrix

def floodfill(matrix, x, y):
  if matrix[x][y] == 0.0:
    matrix[x][y] = -1.0
    #recursively invoke flood fill on all surrounding cells:
    if x > 0:
      floodfill(matrix,x-1,y)
    if x < len(matrix[y]) - 1:
      floodfill(matrix,x+1,y)
    if y > 0:
      floodfill(matrix,x,y-1)
    if y < len(matrix) - 1:
      floodfill(matrix,x,y+1)

"""
dataset = VTK_data("./xml_runs")
dataset.load_data()
batch_boundary, batch_data = dataset.minibatch(batch_type="flow")
for i in xrange(32):
  plt.imshow(batch_boundary[i][:,:,0])
  plt.show()
  plt.imshow(batch_data[i][:,:,0])
  #plt.plot(batch_data[i])
  plt.show()
""" 
