# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 12:26:10 2022

@author: kucer
"""

import numpy as np
import time
import fields
import thermodynamics as td
import cloud
import scipy.constants

#%% testing meshgrid
xmin, xmax, dx = 0.0, 10, 1
ymin, ymax, dy = 30, 40, 1
zmin, zmax, dz = 60, 70, 1

x = np.arange(xmin, xmax, dx)
y = np.arange(ymin, ymax, dy)
z = np.arange(zmin, zmax, dz)
xx, yy, zz = np.meshgrid(x, y, z, indexing = 'ij')

i, j, k = 2, 5,5
print(x[i], y[j], z[k], xx[i,j,k], yy[i,j,k], zz[i,j,k])

vectors = np.stack((xx, yy, zz), axis=3)
vector_plane = vectors[0]
plane_shape = vector_plane.shape

positions = vector_plane.reshape((plane_shape[0]*plane_shape[1], 3))
print(vector_plane)
print(positions)
print(positions[0])

#%% testing integration speed
x = np.random.rand(100000)

t1 = time.time()
a = np.sum(np.exp(x) * 5)
t2= time.time()
print(a)
print(t2 - t1)

t1 = time.time()
b = np.sum(np.exp(x)) * 5
t2= time.time()
print(b)
print(t2 - t1)

# as expected, the second one is much faster!

#%% testing the cutoff calculator
# create dipolar fields
odt1 = fields.DipoleBeam(power=6.0, waist_y = 22.5E-6, waist_z=25.5E-6, theta_z = 0, theta_x = 0)
odt2 = fields.DipoleBeam(power=2.0, waist_y = 131.45E-6, waist_z=38.8E-6, theta_z = np.pi/2, theta_x = np.pi/2 - 15 * np.pi / 180)

field = fields.DipoleField()
field.add_beam(odt1)
field.add_beam(odt2)

# specify cloud parameters
mass_Er = 165.93 * scipy.constants.physical_constants['atomic mass constant'][0]
atoms = cloud.Cloud(temperature=1E-6, N_particles = 1E6, mass = mass_Er, polarizability=-2.9228E-39)


cloud_size = td.calculate_cloud_size(field, atoms, gamma_cutoff = 10, epsilon=1E-8)
print(cloud_size)

#%% testing density calculatory
xx, yy, zz, dV = td.generate_integration_grid(field, atoms, gamma_cutoff=10, dl=0.5E-6)
grid = xx, yy, zz
n0 = td.calculate_peak_density(grid, dV, field, atoms)
print(n0)
n0_harm = td.calculate_ideal_peak_density(field, atoms)
print(n0_harm)