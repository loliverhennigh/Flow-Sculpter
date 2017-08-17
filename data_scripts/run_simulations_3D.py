
import sys
sys.path.append('../')

import queue_runner.que as que
from lxml import etree

size = 32

def should_run(xml_filename):
  tree = etree.parse(xml_filename)
  root = tree.getroot()
  sim_size = int(root.find("size").text)
  if sim_size == size:
    return True
  else:
    return False
  
q = que.Que("../steady_state_flow_3D/steady_state_flow_3D", 4)
q.enque_file("../data/experiment_runs_master.xml", should_run)
q.start_que_runner()





