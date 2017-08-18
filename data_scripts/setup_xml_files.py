
# construct xml dataset of objects

import os
import numpy as np
import lxml.etree as etree
from tqdm import *
import glob
import subprocess

# seed numpy for same rotation every run
np.random.seed(0)

# params to run
#base_path = os.path.abspath("../data/") + "/"
base_path = os.path.abspath("../data/") + "/"
num_flips_per_type = 4
num_samples_per_type = 20
sizes = [16, 32, 64, 96]

# helper for voxelizing
def voxelize_file(filename, size, flip_x, flip_z):
  new_filename = filename[:-4] + "_size_" + str(size) + "_rotation_" + save_str + ".binvox"
  if os.path.isfile(new_filename):
    return
  flip_str = " "
  save_str = str(flip_x) + str(flip_z)
  flip_str += flip_x*"-rotx "
  flip_str += flip_z*"-rotz "
  vox_cmd = "../vox_utils/binvox -d " + str(size) + " -cb" + flip_str + "-e " + filename
  with open(os.devnull, 'w') as devnull:
    ret = subprocess.check_call(vox_cmd.split(' '), stdout=devnull, stderr=devnull)
  rename_cmd = "mv " + filename[:-4] + ".binvox"  + " " 
  ret = subprocess.check_call((rename_cmd + new_filename).split(' '))
  return new_filename

# helper for saving xml
def save_xml(filename, class_id, obj_id, vox_filename, dim, size, flip_x, filp_z):
  single_root = etree.Element("Param")
  # save info
  etree.SubElement(single_root, "class_id").text = ids
  etree.SubElement(single_root, "obj_id").text = obj_id
  etree.SubElement(single_root, "binvox_name").text = vox_filename
  save_path = (base_path + "simulation_data_" + str(dim) + "D/" 
              + vox_filename.split('/')[-2] + "_" 
              + vox_filename.split('/')[-1][:-7] + "/")
  etree.SubElement(single_root, "save_path").text = save_path
  etree.SubElement(single_root, "dim").text = str(dim)
  etree.SubElement(single_root, "size").text = str(size)
  etree.SubElement(single_root, "flip_x").text = str(flip_x)
  etree.SubElement(single_root, "flip_z").text = str(flip_z)
  # save simulation info (will run simulation later)
  flow_data = etree.SubElement(single_root, "flow_data")
  etree.SubElement(flow_data, "availible").text = str(False)
  tree = etree.ElementTree(single_root)
  tree.write(filename, pretty_print=True)

if __name__ == "__main__":
  # get all class labels
  class_ids = os.walk(base_path + "train/").next()[1]

  # root for main xml file
  main_root = etree.Element("experiments")

  # make dir to save xml files
  try:
    os.mkdir(base_path + "xml_files")
  except:
    pass

  # remove any previous xml files just to be safe
  #for rm_file in glob.glob(base_path + "xml_files/*.xml"):
  #  os.remove(rm_file)

  for ids in tqdm(class_ids):
    # get list of all objects in class
    objs_in_class = glob.glob(base_path + "train/" + ids + "/*.obj")
    objs_in_class.sort() # sort them for reproducability
    #for rm_file in glob.glob(base_path + "train/" + ids + "/*.binvox"):
    #  os.remove(rm_file)

    for i in xrange(num_samples_per_type):
      # get obj id
      obj_id = objs_in_class[i].split('/')[-1][6:-4]
  
      for j in xrange(num_flips_per_type):
        # generate random rotation values
        flip_x = np.random.randint(0, 4)
        flip_z = np.random.randint(0, 4)
  
        for k in xrange(len(sizes)):
          for dim in [2, 3]:
            # xml filename
            xml_filename = (base_path + "xml_files/" + ids + "_" + obj_id 
                           + "_" + str(dim) + "_" + str(sizes[k]).zfill(4) 
                           + "_" + str(flip_x) + str(flip_z) + ".xml")

            if not os.path.isfile(xml_filename):
              # generate binvox element
              vox_filename = voxelize_file(objs_in_class[i], sizes[k], flip_x, flip_z)
  
              # save xml file for specific vox
              save_xml(xml_filename, ids, obj_id, vox_filename, dim, sizes[k], flip_x, flip_z)
  
            # make xml element for main
            main_run = etree.SubElement(main_root, "run")
            etree.SubElement(main_run, "class_id").text = ids
            etree.SubElement(main_run, "obj_id").text = obj_id
            etree.SubElement(main_run, "dim").text = str(dim)
            etree.SubElement(main_run, "size").text = str(sizes[k])
            etree.SubElement(main_run, "xml_filename").text = xml_filename
  
  # save main xml file
  tree = etree.ElementTree(main_root)
  tree.write(base_path + "experiment_runs_master.xml", pretty_print=True)

