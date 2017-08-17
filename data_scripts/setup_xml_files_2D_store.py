
# construct xml dataset for 2D

import os
import numpy as np
import lxml.etree as etree
from tqdm import *
import glob
import subprocess

# seed numpy for same rotation every run
np.random.seed(0)

# params to run
base_path = '../data_2D/'
size = [128, 512]
num_simulations = 3000
num_obj_range = [1, 7] 
obj_size_range = [16, 48]

# helper for making circle vector
def rand_circle():
  circle    = np.zeros(3)
  circle[0] = np.random.randint(obj_size_range[0]/2,
                                obj_size_range[1]/2)
  circle[1] = np.random.randint(obj_size_range[1]+2,
                                size[0]-obj_size_range[1]-2)
  circle[2] = np.random.randint(obj_size_range[1]+size[1]/8,
                                size[1]/2)
  return circle

# helper for making rectangle vector
def rand_rectangle():
  rectangle    = np.zeros(4)
  rectangle[0] = np.random.randint(2,
                                   size[0]-obj_size_range[1]-obj_size_range[0]-2)
  rectangle[1] = np.random.randint(obj_size_range[1]+size[1]/8,
                                   size[1]/2)
  rectangle[2] = np.random.randint(rectangle[0]+obj_size_range[0],
                                   rectangle[0]+obj_size_range[1])
  rectangle[3] = np.random.randint(rectangle[1]+obj_size_range[0],
                                   rectangle[1]+obj_size_range[1])
  return rectangle 

# helper for saving xml
def save_xml(filename, sim_id, circles, rectangles):
  single_root = etree.Element("Param")
  # save info
  etree.SubElement(single_root, "sim_id").text = str(sim_id)
  etree.SubElement(single_root, "save_path").text = base_path + "simulation_data/run_" + str(sim_id) + "/"
  etree.SubElement(single_root, "size_x").text = str(size[0])
  etree.SubElement(single_root, "size_y").text = str(size[1])
  # add list of circles
  for i in xrange(len(circles)):
    circle_root = etree.SubElement(single_root, "circles_" + str(i))
    etree.SubElement(circle_root, "radius").text = str(circles[i][0])
    etree.SubElement(circle_root, "pos_x").text = str(circles[i][1])
    etree.SubElement(circle_root, "pos_y").text = str(circles[i][2])
  # add list of circles
  for i in xrange(len(rectangles)):
    rectangle_root = etree.SubElement(single_root, "rectangles_" + str(i))
    etree.SubElement(rectangle_root, "pos_x_1").text = str(rectangles[i][0])
    etree.SubElement(rectangle_root, "pos_y_1").text = str(rectangles[i][1])
    etree.SubElement(rectangle_root, "pos_x_2").text = str(rectangles[i][2])
    etree.SubElement(rectangle_root, "pos_y_2").text = str(rectangles[i][3])
  # save simulation info (will run simulation later)
  flow_data = etree.SubElement(single_root, "flow_data")
  etree.SubElement(flow_data, "availible").text = str(False)
  tree = etree.ElementTree(single_root)
  tree.write(filename, pretty_print=True)

if __name__ == "__main__":
  # root for main xml file
  main_root = etree.Element("experiments")

  # make dir to save xml files
  try:
    os.mkdir(base_path + "xml_files")
  except:
    pass

  # remove any previous xml files just to be safe
  for rm_file in glob.glob(base_path + "xml_files/*.xml"):
    os.remove(rm_file)

  for i in tqdm(xrange(num_simulations)):
    sim_id = str(i).zfill(5)  

    # random objs
    circles = []
    rectangles = []
    num_obj = np.random.randint(*num_obj_range)
    for k in xrange(num_obj):
      obj_type = np.random.randint(0,2)
      if obj_type == 0:
        circles.append(rand_circle())
      elif obj_type == 1:
        rectangles.append(rand_rectangle())
      
    # xml filename
    xml_filename = base_path + "xml_files/run_" + sim_id + ".xml"

    # save xml file for specific vox
    save_xml(xml_filename, sim_id, circles, rectangles)

    # make xml element for main
    main_run = etree.SubElement(main_root, "run")
    etree.SubElement(main_run, "id").text = sim_id
    etree.SubElement(main_run, "xml_filename").text = xml_filename
  
# save main xml file
tree = etree.ElementTree(main_root)
tree.write(base_path + "experiment_runs_master.xml", pretty_print=True)

