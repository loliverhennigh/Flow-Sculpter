
import os
import process
import time
from lxml import etree
from termcolor import colored

class Que:
  def __init__(self, command, num_processes=1):
    self.pl = []
    self.running_pl = []
    self.num_processes = num_processes
    self.command = command
    self.available_processes = num_processes
    self.start_time = 0

  def enque_file(self, xml_filename, should_run, xml_update_script):
    # read runs from master xml file
    root = etree.parse(xml_filename)
    list_of_runs = root.findall("run") 
    for run in list_of_runs:
      xml_filename = run.find("xml_filename").text
      if should_run(xml_filename):
        self.pl.append(process.Process(self.command, xml_filename, xml_update_script))

  def start_next(self):
    for i in xrange(len(self.pl)):
      if self.pl[i].get_status() == "Not Started":
        self.pl[i].start()
        break

  def num_free_processes(self):
    num_running_processes = 0
    for i in xrange(len(self.pl)):
      if self.pl[i].get_status() == "Running":
        num_running_processes += 1 
    return self.available_processes - num_running_processes

  def num_finished_processes(self):
    proc = 0
    for i in xrange(len(self.pl)):
      if self.pl[i].get_status() == "Finished" and self.pl[i].get_return_status() == "SUCCESS":
        proc += 1
    return proc

  def num_failed_processes(self):
    proc = 0
    for i in xrange(len(self.pl)):
      if self.pl[i].get_status() == "Finished" and self.pl[i].get_return_status() == "FAIL":
        proc += 1
    return proc

  def num_running_processes(self):
    proc = 0
    for i in xrange(len(self.pl)):
      if self.pl[i].get_status() == "Running":
        proc += 1
    return proc

  def num_unstarted_processes(self):
    proc = 0
    for i in xrange(len(self.pl)):
      if self.pl[i].get_status() == "Not Started":
        proc += 1
    return proc

  def percent_complete(self):
    rc = -.01
    if self.num_finished_processes() > 0:
      rc = self.num_finished_processes() / float(len(self.pl))
    return rc * 100.0

  def run_time(self):
    return time.time() - self.start_time

  def time_left(self):
    tl = -1
    pc = self.percent_complete()
    if pc > 0:
      tl = (time.time() - self.start_time) * (1.0/(pc/100.0)) - self.run_time()
    return tl

  def time_string(self, tl):
    tl = int(tl)
    if tl < 0:
      tl = 0
    seconds = tl % 60 
    tl = (tl - seconds)/60
    mins = tl % 60 
    tl = (tl - mins)/60
    hours = tl % 24
    days = (tl - hours)/24
    return    ("(" + str(days).zfill(3) + ":" + str(hours).zfill(2) 
             + ":" + str(mins).zfill(2) + ":" + str(seconds).zfill(2) + ")")
 
  def update_pl_status(self):
    for i in xrange(len(self.pl)):
      self.pl[i].update_status()

  def print_que_status(self):
    os.system('clear')
    print("QUE STATUS")
    print(colored("Num Finished Success: " + str(self.num_finished_processes()), 'green'))
    print(colored("Num Finished Fail:    " + str(self.num_failed_processes()), 'red'))
    print(colored("Num Running:          " + str(self.num_running_processes()), 'yellow'))
    print(colored("Num Left:             " + str(self.num_unstarted_processes()), 'blue'))
    print(colored("Percent Complete:     " + str(self.percent_complete()), 'blue'))
    print(colored("Time Left (D:H:M:S):  " + self.time_string(self.time_left()), 'blue'))
    print(colored("Run Time  (D:H:M:S):  " + self.time_string(self.run_time()), 'blue'))
 
  def start_que_runner(self):
    self.start_time = time.time()
    while True:
      time.sleep(0.2)
      num_free = self.num_free_processes()
      if num_free > 0:
        self.start_next()
      self.update_pl_status()
      self.print_que_status()
      


