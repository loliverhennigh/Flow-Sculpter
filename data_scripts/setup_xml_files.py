
# construct xml dataset of objects

import os
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
#base_path = os.path.abspath("../data/") + "/"
base_path = os.path.abspath("../data/") + "/"
num_wingfoil_sim = 5000
num_heat_sink_sim = 5000
num_wingfoil_params = 46
num_heat_sink_params = 15
heat_sink_sizes_2d = [128]
wing_sizes_2d = [64, 128, 256]
wing_sizes_3d = []
nr_threads = 10

# helper for saving xml
def wing_save_xml(filename, ids, vox_filename, dim, size):
  single_root = etree.Element("Param")
  # save info
  #etree.SubElement(single_root, "class_id").text = ids
  etree.SubElement(single_root, "obj_ids").text = ids
  etree.SubElement(single_root, "binvox_name").text = vox_filename
  save_path = (base_path + "simulation_data_" + str(dim) + "D/" 
              + vox_filename.split('/')[-2] + "_" 
              + vox_filename.split('/')[-1][:-7] + "/")
  etree.SubElement(single_root, "save_path").text = save_path
  etree.SubElement(single_root, "dim").text = str(dim)
  etree.SubElement(single_root, "cmd").text = ("python ../sailfish_flows/steady_state_flow_" + str(dim) + "d.py " 
                                              + "--vox_filename=" + vox_filename + " "
                                              + "--vox_size="     + str(size) + " "
                                              + "--output="       + str(save_path) + "steady_state_flow")
  etree.SubElement(single_root, "size").text = str(size)
  # save simulation info (will run simulation later)
  flow_data = etree.SubElement(single_root, "flow_data")
  etree.SubElement(flow_data, "availible").text = str(False)
  tree = etree.ElementTree(single_root)
  tree.write(filename, pretty_print=True)

# helper for saving xml
def heat_sink_save_xml(filename, ids, geometry_filename, size):
  single_root = etree.Element("Param")
  # save info
  #etree.SubElement(single_root, "class_id").text = ids
  etree.SubElement(single_root, "obj_ids").text = ids
  etree.SubElement(single_root, "geometry_name").text = geometry_filename
  save_path = (base_path + "heat_sink_data/" 
              + str(ids).zfill(4) + "_" 
              + str(size).zfill(4) + "/")
  etree.SubElement(single_root, "save_path").text = save_path
  etree.SubElement(single_root, "cmd").text = ("python ../diffusion/heat_sink.py " 
                                              + geometry_filename + " "
                                              + str(save_path))
  etree.SubElement(single_root, "size").text = str(size)
  # save simulation info (will run simulation later)
  heat_sink_data = etree.SubElement(single_root, "heat_sink_data")
  etree.SubElement(heat_sink_data, "availible").text = str(False)
  tree = etree.ElementTree(single_root)
  tree.write(filename, pretty_print=True)


def wing_save_worker():
  while True:
    ids, dim, k, sizes = queue.get()
    # xml filename
    xml_filename = (base_path + "xml_files/wing_" + str(ids).zfill(4) + "_"
                 + str(dim) + "_" + str(sizes[k]).zfill(4) 
                 + ".xml")
    wing_filename = (base_path + "wingfoils/wing_" + str(ids).zfill(4) + "_"
                 + str(dim) + "_" + str(sizes[k]).zfill(4) 
                 + ".npy")

    # make numpy boundary file
    if not os.path.isfile(wing_filename):
      params = boundary_utils.get_random_params(num_wingfoil_params, dim)
      if dim == 2: 
        wing = boundary_utils.wing_boundary_2d(params[0], params[1], params[2],
                                params[3:(num_wingfoil_params-4)/2],
                                params[(num_wingfoil_params-4)/2:-1],
                                params[-1], dim*[sizes[k]])
      elif dim == 3: 
        wing = boundary_utils.wing_boundary_3d(params[0], params[1], params[2],
                                               params[3], params[4], params[5],
                                               params[6:(num_wingfoil_params-7)/3+6],
                                               params[(num_wingfoil_params-7)/3+6:2*(num_wingfoil_params-7)/3+6],
                                               params[2*(num_wingfoil_params-7)/3+6:-1],
                                               params[-1], dim*[sizes[k]])
      wing = np.greater(wing, 0.5)
      np.save(wing_filename[:-4], wing)

    if not os.path.isfile(xml_filename):
      wing_save_xml(xml_filename, str(ids), wing_filename, dim, sizes[k])

    # make xml element for main
    main_run = etree.SubElement(main_root, "run")
    etree.SubElement(main_run, "dim").text = str(dim)
    etree.SubElement(main_run, "size").text = str(sizes[k])
    etree.SubElement(main_run, "xml_filename").text = xml_filename

    # end worker
    queue.task_done()
 
def heat_sink_save_worker(ids, k, sizes):

  # xml filename
  xml_filename = (base_path + "xml_files/heat_sink_" + str(ids).zfill(4) + "_"
               + str(sizes[k]).zfill(4) 
               + ".xml")
  heat_sink_filename = (base_path + "heat_sinks/heat_sink_" + str(ids).zfill(4) + "_"
               + str(sizes[k]).zfill(4) 
               + ".npy")

  # make numpy boundary file
  if not os.path.isfile(heat_sink_filename):
    params = boundary_utils.get_random_params_heat_sink(num_heat_sink_params)
    heat_sink = boundary_utils.heat_sink_boundary_2d(params, 2*[sizes[k]])
    np.save(heat_sink_filename[:-4], heat_sink)

  if not os.path.isfile(xml_filename):
    heat_sink_save_xml(xml_filename, str(ids), heat_sink_filename, sizes[k])

  # make xml element for main
  main_run = etree.SubElement(main_root, "run")
  etree.SubElement(main_run, "size").text = str(sizes[k])
  etree.SubElement(main_run, "xml_filename").text = xml_filename

# Start Que 
queue = Queue(100)

# Start threads
for i in xrange(nr_threads):
  get_thread = threading.Thread(target=wing_save_worker)
  get_thread.daemon = True
  get_thread.start()

# root for main xml file
main_root = etree.Element("experiments")

# make dir to save xml files
try:
  os.mkdir(base_path + "xml_files")
except:
  pass

ids = 0
for i in tqdm(xrange(num_wingfoil_sim)):
  ids += 1
  for dim in [2,3]:
    if dim == 2:
      sizes = wing_sizes_2d
    elif dim == 3:
      sizes = wing_sizes_3d
    for k in xrange(len(sizes)):
      # put on worker thread
      queue.put((ids, dim, k, sizes))
      
# save main xml file
tree = etree.ElementTree(main_root)
tree.write(base_path + "experiment_runs_master.xml", pretty_print=True)

# root for main xml file
main_root = etree.Element("experiments")

ids = 0
for i in tqdm(xrange(num_heat_sink_sim)):
  ids += 1
  sizes = heat_sink_sizes_2d
  for k in xrange(len(sizes)):
    # put on worker thread
    heat_sink_save_worker(ids, k, sizes)

queue.join()
      
# save main xml file
tree = etree.ElementTree(main_root)
tree.write(base_path + "experiment_runs_master_heat_sink.xml", pretty_print=True)



