
import sys
sys.path.append('../')

import queue_runner.que as que

q = que.Que("../steady_state_flow_2D/steady_state_flow_2D", 4)
q.enque_file("../data_2D/experiment_runs_master.xml")
q.start_que_runner()





