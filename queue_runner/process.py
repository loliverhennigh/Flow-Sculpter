
import sys
import psutil as ps
import os
import time
import glob
import lxml.etree as etree
from termcolor import colored

class Process:
  def __init__(self, command, xml_filename, finish_script):
    self.cmd = command
    self.xml_file = xml_filename
    self.finish_script = finish_script
    # check if simulaiton is already ran
    tree = etree.parse(self.xml_file)
    root = tree.getroot()
    availible = root.find("flow_data").find("availible").text
    if availible == "False":
      self.status = "Not Started"
      self.return_status = "NONE"
      availible = root.find("flow_data").find("availible").text
      
    else:
      self.status = "Finished"
      self.return_status = "SUCCESS"

    self.process = None
    self.run_time = 0

  def start(self):
    with open(os.devnull, 'w') as devnull:
      self.process = ps.subprocess.Popen([self.cmd, self.xml_file], stdout=devnull, stderr=devnull)
    self.pid = self.process.pid

    self.status = "Running"
    self.start_time = time.time()

  def update_status(self):
    if self.status == "Running":
      self.run_time = time.time() - self.start_time
      if self.process.poll() is not None:
        self.status = "Finished"
        if self.process.poll() == 0:
          self.return_status = "SUCCESS"
          self.finish_script(self.xml_file)
        else:
          self.return_status = "FAIL"
  def get_pid(self):
    return self.pid

  def get_status(self):
    return self.status

  def get_return_status(self):
    return self.return_status

  def print_info(self):
    print_string = (colored('cmd is ', 'blue') + ' '.join(self.cmd) + '\n').ljust(40)
    print_string = print_string + (colored('status ', 'blue') + self.status + '\n').ljust(30)
    if self.return_status == "SUCCESS":
      print_string = print_string + (colored('return status ', 'blue') + colored(self.return_status, 'green') + '\n').ljust(40)
    elif self.return_status == "FAIL":
      print_string = print_string + (colored('return status ', 'blue') + colored(self.return_status, 'red') + '\n').ljust(40)
    else:
      print_string = print_string + (colored('return status ', 'blue') + colored(self.return_status, 'yellow') + '\n').ljust(40)
    print_string = print_string + (colored('run time ', 'blue') + str(self.run_time)).ljust(40)
    print(print_string)


