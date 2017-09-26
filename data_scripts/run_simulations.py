
import sys
sys.path.append('../')

import os
import queue_runner.que as que
from lxml import etree
import glob
import subprocess

#dim = 2
#size = 64
dim = 3
size = 32

def initialize_script(xml_file):
  tree = etree.parse(xml_file)
  root = tree.getroot()
  base_path = root.find("save_path").text
  with open(os.devnull, 'w') as devnull:
    try:
      subprocess.check_call(("rm -r " + base_path).split(' '), stdout=devnull, stderr=devnull)
    except:
      pass

def finish_script(xml_file):
  tree = etree.parse(xml_file)
  root = tree.getroot()
  flow_data = root.find("flow_data")
  base_path = root.find("save_path").text
  flow_data.find("availible").text = "True"
  etree.SubElement(flow_data, "geometry_file").text = base_path + "steady_state_flow_boundary.npy"
  etree.SubElement(flow_data, "flow_file").text = base_path + "steady_state_flow_steady_flow.npz"
  tree = etree.ElementTree(root)
  tree.write(xml_file, pretty_print=True)

def should_run(root):
  xml_dim  = int(root.find("dim").text)
  xml_size = int(root.find("size").text)
  xml_file = root.find("xml_filename").text
  tree = etree.parse(xml_file)
  root = tree.getroot()
  is_availible = root.find("flow_data").find("availible").text
  no_vtm = False 
  if is_availible == "True":
    flow_file = root.find("flow_data").find("flow_file").text
    if os.path.isfile(flow_file):
      no_vtm = True
  if   ((xml_size != size) or (xml_dim != dim) or (no_vtm)):
    return False
  else:
    return True
  
q = que.Que([1])
q.enque_file("../data/experiment_runs_master.xml", should_run, initialize_script, finish_script)
q.start_que_runner()





