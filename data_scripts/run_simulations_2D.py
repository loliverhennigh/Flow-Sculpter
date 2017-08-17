
import sys
sys.path.append('../')

import queue_runner.que as que
from lxml import etree
import glob

size = 32

def xml_update_script(xml_file):
    tree = etree.parse(xml_file)
    root = tree.getroot()
    flow_data = root.find("flow_data")
    base_path = root.find("save_path_2d").text
    flow_data.find("availible").text = "True"
    etree.SubElement(flow_data, "geometry_file").text = base_path + "vtkData/geometry_iT0000000.vtm"
    print(base_path + "vtkData/data/*.vtm")
    etree.SubElement(flow_data, "flow_file").text = glob.glob(base_path + "vtkData/data/*.vtm")[0]
    etree.SubElement(flow_data, "drag_file").text = base_path + "gnuplotData/data/drag.dat"
    tree = etree.ElementTree(root)
    tree.write(xml_file, pretty_print=True)

def should_run(xml_filename):
  tree = etree.parse(xml_filename)
  root = tree.getroot()
  sim_size = int(root.find("size").text)
  if sim_size == size:
    return True
  else:
    return False
  
q = que.Que("../steady_state_flow_2D/steady_state_flow_2D", 4)
q.enque_file("../data/experiment_runs_master.xml", should_run, xml_update_script)
q.start_que_runner()





