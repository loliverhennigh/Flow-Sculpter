#!/usr/bin/env python
# coding=utf-8
"""
Velocity-driven flow through a duct of rectangular cross-section.

This case has an analytical solution:
(from F. M. White, "Viscous Fluid Flow", 2nd, p. 120, Eq. 3.48):

    -a <= y <= a
    -b <= z <= b

    u(y, z) = \\frac{16 a^2}{\mu \pi^3} \left(- \\frac{dp}{dx} \\right) \sum_{i =
        1,3,5,...}^{\infty} (-1)^{(i-1)/2} \left(1 - \\frac{\cosh(i \pi z / 2a)}{\cosh({i
        \pi b / 2a)}} \\right) \\frac{\cos(i \pi y/2a)}{i^3}
"""

import sys
sys.path.append('../sailfish/')

import numpy as np
import time

from sailfish.subdomain import Subdomain3D
from sailfish.node_type import NTFullBBWall,NTRegularizedDensity, NTRegularizedVelocity
from sailfish.controller import LBSimulationController
from sailfish.lb_base import ForceObject
from sailfish.lb_single import LBFluidSim
import binvox_rw
import matplotlib.pyplot as plt
import glob
import os

def floodfill(image, x, y, z):
    edge = [(x, y, z)]
    image[x,y,z] = -1
    while edge:
        newedge = []
        for (x, y, z) in edge:
            for (s, t, k) in ((x+1, y, z), (x-1, y, z), (x, y+1, z), (x, y-1, z), (x, y, z+1), (x, y, z-1)):
                if    ((0 <= s) and (s < image.shape[0])
                   and (0 <= t) and (t < image.shape[1])
                   and (0 <= k) and (k < image.shape[2])
                   and (image[s, t, k] == 0)):
                    image[s, t, k] = -1 
                    newedge.append((s, t, k))
        edge = newedge

def clean_files(filename, size):

  # list all saved files
  print("waiting for files saving to chetch up")
  time.sleep(4.0)
  files = glob.glob(filename + ".0.*")
  files.sort()
  steady_flow_file = files[-1]

  # convert the last flow to the proper save formate
  steady_flow_array = np.load(steady_flow_file)
  velocity_array = steady_flow_array.f.v
  pressure_array = np.expand_dims(steady_flow_array.f.rho, axis=0)
  velocity_array[np.where(np.isnan(velocity_array))] = 0.0
  pressure_array[np.where(np.isnan(pressure_array))] = 1.0
  pressure_array = pressure_array - 1.0
  steady_flow_array = np.concatenate([velocity_array, pressure_array], axis=0)
  steady_flow_array = np.swapaxes(steady_flow_array, 0, 1)
  steady_flow_array = np.swapaxes(steady_flow_array, 1, 2)
  steady_flow_array = np.swapaxes(steady_flow_array, 2, 3)
  steady_flow_array = steady_flow_array[size/4:7*size/4]
  np.nan_to_num(steady_flow_array, False)
  steady_flow_array = steady_flow_array.astype(np.float32)
  np.save(filename + "_steady_flow", steady_flow_array) 

  """
  # convert boundary
  geometry_array = np.load(filename + "_boundary.npy")
  geometry_array = geometry_array.astype(np.uint8)
  geometry_array = geometry_array[size/4+1:7*size/4+1,1:-1,1:-1]
  geometry_array = np.expand_dims(geometry_array, axis=-1)
  np.save(filename + "_boundary", geometry_array)
  """

  # clean files
  rm_files = files
  for f in rm_files:
    os.remove(f)

class DuctSubdomain(Subdomain3D):
  max_v = 0.1
  #wall_bc = NTHalfBBWall
  wall_bc = NTFullBBWall

  def boundary_conditions(self, hx, hy, hz):
    wall_map = (hx == 0) | (hx == self.gx - 1) | (hy == 0) | (hy == self.gy - 1)
    self.set_node(wall_map, self.wall_bc)

    vel = self.analytical(hx, hy)[0,1:-1,1:-1]
    vel = np.core.records.fromarrays([np.zeros_like(vel), np.zeros_like(vel), vel])
    vel = vel.flatten()
    self.set_node((hz == 0) & np.logical_not(wall_map),
                  NTRegularizedVelocity(vel))
 
    self.set_node((hz == self.gz - 1) & np.logical_not(wall_map),
                  NTRegularizedDensity(1))

    L = self.config.vox_size
    model = self.load_vox_file(self.config.vox_filename)
    model = np.pad(model, ((L/2,6*L/4),(L/4,L/4),(L/4, L/4)), 'constant', constant_values=False)
    self.set_node(model, self.wall_bc)

    # convert boundary
    geometry_array = model.astype(np.uint8)
    geometry_array = geometry_array[L/4+1:7*L/4+1,1:-1,1:-1]
    geometry_array = np.expand_dims(geometry_array, axis=-1)
    np.save(self.config.output + "_boundary", geometry_array)

  def initial_conditions(self, sim, hx, hy, hz):
    sim.rho[:] = 1.0
    sim.vz[:] = self.analytical(hx, hy)

  @classmethod
  def width(cls, config):
    return config.lat_ny - 1 - 2 * cls.wall_bc.location

  @classmethod
  def accel(cls, config):
    # The maximum velocity is attained at the center of the channel,
    # i.e. x = y = 0.
    ii = np.arange(1, 20, 2)
    ssum = np.sum((-1)**((ii - 1)/2.0) * (1 - np.cosh(0) / np.cosh(ii * np.pi / 2)) *
                  np.cos(0) / ii**3)
    a = cls.width(config) / 2.0
    prefactor = 16 * a**2 / (config.visc * np.pi**3)
    return cls.max_v / (prefactor * ssum)

  def analytical(self, hx, hy):
    a = self.width(self.config) / 2.0
    hy = hy - self.wall_bc.location
    hx = hx - self.wall_bc.location
    ry = np.abs(a - hy)
    rx = np.abs(a - hx)

    prefactor = 16 * a**2 / (self.config.visc * np.pi**3)
    ii = np.arange(1, 100, 2)
    ret = np.zeros_like(hy)
    for i in ii:
      ret += ((-1)**((i - 1)/2.0) *
      (1 - np.cosh(i * np.pi * rx / (2.0 * a)) /
           np.cosh(i * np.pi / 2)) *
      np.cos(i * np.pi * ry / (2.0 * a)) / i**3)

    return self.accel(self.config) * prefactor * ret

  def load_vox_file(self, vox_filename):
    if vox_filename[-3:] == "vox":
      with open(vox_filename, 'rb') as f:
        model = binvox_rw.read_as_3d_array(f)
        model = model.data
    # if the file is .npy assume that it is 2D 0s and 1s
    elif vox_filename[-3:] == "npy":
      model = np.load(vox_filename)
      model = model[...,0]
    #model = np.array(model, dtype=np.int)
    model = np.pad(model, ((1,1,),(1,1),(1,1)), 'constant', constant_values=0)
    #floodfill(model, 0, 0, 0)
    plt.imshow(model[:,:,48])
    plt.show()
    model = np.greater(model, 0.1)
    plt.imshow(model[:,:,48])
    plt.show()
    return model

class DuctSim(LBFluidSim):
  subdomain = DuctSubdomain

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
      'max_iters': 600000,
      'output_format': 'npy',
      'output': 'test_flow',
      'every': 2500,
      'grid': 'D3Q15'
      })


  @classmethod
  def modify_config(cls, config):
    config.lat_nx = 3*config.vox_size/2
    config.lat_ny = 3*config.vox_size/2
    config.lat_nz = config.vox_size*3
    config.visc   = 0.03 * (config.lat_ny/100.0)
    print(config.visc)


  def __init__(self, config):
    super(DuctSim, self).__init__(config)
    # The flow is driven by a body force.

    L = self.config.vox_size
    margin = 5
    self.add_force_oject(ForceObject(
        (3,  3,  3),
        (3*L/2 - 3, 3*L/2 - 3, 2*L + 3)))

    a = DuctSubdomain.width(config) / 2.0
    self.config.logger.info('Re = %.2f, width = %d' % (a * DuctSubdomain.max_v / config.visc, a))

  def record_value(self, iteration, force):
    print("iteration")
    print(iteration)
    print("force_x")
    print(force[0])
    print("force_y")
    print(force[1])
    print("force_z")
    print(force[2])

  prev_f = None
  every = 500
  def after_step(self, runner):
    if self.iteration % self.every == 0:
      runner.update_force_objects()
      for fo in self.force_objects:
        runner.backend.from_buf(fo.gpu_force_buf)
        f = fo.force()

        # Compute drag and lift coefficients.
        #C_D = (2.0 * f[0] / (self.config.lat_nx * BoxSubdomain.max_v**2 ))
        #C_L = (2.0 * f[1] / (self.config.lat_nx * BoxSubdomain.max_v**2 ))
        self.record_value(runner._sim.iteration, f)

        if self.prev_f is None:
          self.prev_f = np.array(f)
        else:
          f = np.array(f)

          # Terminate simulation when steady state has
          # been reached.
          diff = np.abs(f - self.prev_f) / (np.abs(f) + 1.0e-1)
          print("diff")
          print(diff)

          if (np.all(diff < 1e-4) or (self.config.max_iters < self.iteration + 501)) and (not ( 2000 > self.iteration)):
            clean_files(self.config.output, self.config.vox_size)
            runner._quit_event.set()
          self.prev_f = f

if __name__ == '__main__':
  ctrl = LBSimulationController(DuctSim)
  ctrl.run()
