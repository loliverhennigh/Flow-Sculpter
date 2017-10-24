from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time
import cv2
import csv
import re
from glob import glob as glb

import numpy as np
import tensorflow as tf

import sys
sys.path.append('../')

import matplotlib.pyplot as plt

res_net = np.load("./figs/store_flow_accuracy_values/residual_network.npz")
res_p_drag_x_data = res_net.f.p_drag_x_data
res_t_drag_x_data = res_net.f.t_drag_x_data
res_p_drag_y_data = res_net.f.p_drag_x_data
res_t_drag_y_data = res_net.f.t_drag_x_data
res_p_max_vel_data = res_net.f.p_max_vel_data
res_t_max_vel_data = res_net.f.t_max_vel_data

res_net = np.load("./figs/store_flow_accuracy_values/xiao_network.npz")
xiao_p_drag_x_data = res_net.f.p_drag_x_data
xiao_t_drag_x_data = res_net.f.t_drag_x_data
xiao_p_drag_y_data = res_net.f.p_drag_x_data
xiao_t_drag_y_data = res_net.f.t_drag_x_data
xiao_p_max_vel_data = res_net.f.p_max_vel_data
xiao_t_max_vel_data = res_net.f.t_max_vel_data

fig = plt.figure(figsize = (15,15))
a = fig.add_subplot(2,2,1)
font_size_axis=6
s = 2.5
plt.scatter(xiao_t_drag_x_data, xiao_p_drag_x_data, color="green", label="(Guo et al., 2016)", s=s)
plt.scatter(res_t_drag_x_data, res_p_drag_x_data, label="Our Network", s=s, color="blue")
plt.plot(res_t_drag_x_data, res_t_drag_x_data, color="red", linewidth=0.7)
plt.title("X Force", fontsize=38)
plt.legend()
a = fig.add_subplot(2,2,2)
plt.scatter(xiao_t_drag_y_data, xiao_p_drag_y_data, color="green", s=s)
plt.scatter(res_t_drag_y_data, res_p_drag_y_data, s=s, color="blue")
plt.plot(res_t_drag_y_data, res_t_drag_y_data, color="red", linewidth=0.7)
plt.title("Y Force", fontsize=38)
a = fig.add_subplot(2,2,3)
plt.scatter(xiao_t_max_vel_data, xiao_p_max_vel_data, color="green", s=s)
plt.scatter(res_t_max_vel_data, res_p_max_vel_data, s=s, color="blue")
plt.plot(res_t_max_vel_data, res_t_max_vel_data, color="red", linewidth=0.7)
plt.title("Max Velocity", fontsize=38)
plt.ylabel("Predicted", fontsize=26)
plt.xlabel("True", fontsize=26)
plt.savefig("./figs/flow_accuracy_2d.pdf")
plt.show()

