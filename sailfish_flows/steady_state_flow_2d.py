"""2D flow around a object in a channel.

Lift and drag coefficients of the object are measured using the
momentum exchange method.

Fully developed parabolic profile is prescribed at the inflow and
a constant pressure condition is prescribed at the outflow.

The results can be compared with:
    [1] M. Breuer, J. Bernsdorf, T. Zeiser, F. Durst
    Accurate computations of the laminar flow past a object
    based on two different methods: lattice-Boltzmann and finite-volume
    Int. J. of Heat and Fluid Flow 21 (2000) 186-196.
"""
from __future__ import print_function

import sys
sys.path.append('../sailfish/')

import numpy as np
from sailfish.subdomain import Subdomain2D
from sailfish.node_type import NTHalfBBWall, NTEquilibriumVelocity, NTEquilibriumDensity, DynamicValue, NTFullBBWall
from sailfish.controller import LBSimulationController
from sailfish.lb_base import ForceObject
from sailfish.lb_single import LBFluidSim
from sailfish.sym import S
import binvox_rw
import matplotlib.pyplot as plt
import glob
import os

def floodfill(image, x, y):
    edge = [(x, y)]
    image[x,y] = -1
    while edge:
        newedge = []
        for (x, y) in edge:
            for (s, t) in ((x+1, y), (x-1, y), (x, y+1), (x, y-1)):
                if     ((0 <= s) and (s < image.shape[0])
                   and (0 <= t) and (t < image.shape[1])
                   and (image[s, t] == 0)):
                    image[s, t] = -1 
                    newedge.append((s, t))
        edge = newedge

def clean_files(filename):
  files = glob.glob(filename + ".0.*")
  files.sort()
  rm_files = files[:-1]
  for f in rm_files:
    os.remove(f)
  os.rename(files[-1], filename + "_steady_flow.npz")

class BoxSubdomain(Subdomain2D):
  #bc = NTHalfBBWall
  bc = NTFullBBWall
  max_v = 0.08

  def boundary_conditions(self, hx, hy):

    walls = (hy == 0) | (hy == self.gy - 1)
    self.set_node(walls, self.bc)

    H = self.config.lat_ny
    hhy = S.gy - self.bc.location
    self.set_node((hx == 0) & np.logical_not(walls),
                  NTEquilibriumVelocity(
                  DynamicValue(4.0 * self.max_v / H**2 * hhy * (H - hhy), 0.0)))
    self.set_node((hx == self.gx - 1) & np.logical_not(walls),
                  NTEquilibriumDensity(1))

    L = self.config.vox_size
    model = self.load_vox_file(self.config.vox_filename)
    model = np.pad(model, ((L/2,L/2),(L, 6*L)), 'constant', constant_values=False)
    np.save(self.config.output + "_boundary", model)
    self.set_node(model, self.bc)


  def initial_conditions(self, sim, hx, hy):
    H = self.config.lat_ny
    sim.rho[:] = 1.0
    sim.vy[:] = 0.0

    hhy = hy - self.bc.location
    sim.vx[:] = 4.0 * self.max_v / H**2 * hhy * (H - hhy)

  def load_vox_file(self, vox_filename):
    with open(vox_filename, 'rb') as f:
      model = binvox_rw.read_as_3d_array(f)
      model = model.data[:,:,model.dims[2]/2]
    model = np.array(model, dtype=np.int)
    model = np.pad(model, ((1,1),(1, 1)), 'constant', constant_values=0)
    floodfill(model, 0, 0)
    model = np.greater(model, -0.1)
    return model

class BoxSimulation(LBFluidSim):
  subdomain = BoxSubdomain

  @classmethod
  def add_options(cls, group, defaults):
    group.add_argument('--vox_filename',
            help='name of vox file to run ',
            type=str, default="test.vox")
    group.add_argument('--vox_size',
            help='size of vox file to run ',
            type=int, default=32)

  @classmethod
  def update_defaults(cls, defaults):
    defaults.update({
      'max_iters': 300000,
      'output_format': 'npy',
      'output': 'test_flow',
      'every': 5000,
      })

  @classmethod
  def modify_config(cls, config):
    config.lat_nx = config.vox_size*8
    config.lat_ny = config.vox_size*2
    config.visc   = 0.05 * (config.lat_ny/100.0)
    print(config.visc)

  def __init__(self, *args, **kwargs):
    super(BoxSimulation, self).__init__(*args, **kwargs)

    L = self.config.vox_size
    margin = 5
    self.add_force_oject(ForceObject(
      (L-margin,  L/2 - margin),
      (2*L + margin, 3*L/2 + margin)))

    subdomain = BoxSubdomain
    print('Re = %2.f' % (BoxSubdomain.max_v * L / self.config.visc))

  def record_value(self, iteration, force, C_D, C_L):
    print("iteration")
    print(iteration)
    print("force_x")
    print(force[0])
    print("force_y")
    print(force[1])
    print("c_d")
    print(C_D)
    print("c_l")
    print(C_L)

  prev_f = None
  every = 500
  def after_step(self, runner):
    if self.iteration % self.every == 0:
      runner.update_force_objects()
      for fo in self.force_objects:
        runner.backend.from_buf(fo.gpu_force_buf)
        f = fo.force()

        # Compute drag and lift coefficients.
        C_D = (2.0 * f[0] / (self.config.lat_nx * BoxSubdomain.max_v**2))
        C_L = (2.0 * f[1] / (self.config.lat_ny * BoxSubdomain.max_v**2))
        self.record_value(runner._sim.iteration, f, C_D, C_L)

        if self.prev_f is None:
          self.prev_f = np.array(f)
        else:
          f = np.array(f)

          # Terminate simulation when steady state has
          # been reached.
          diff = np.abs(f - self.prev_f) / (np.abs(f) + 0.001)

          print(diff)
          if np.all(diff < 1e-4) or (self.config.max_iters < self.iteration + 501):
            clean_files(self.config.output)
            runner._quit_event.set()
          self.prev_f = f


if __name__ == '__main__':
  ctrl = LBSimulationController(BoxSimulation)
  ctrl.run()
