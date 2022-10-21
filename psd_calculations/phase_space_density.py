# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import fields
import cloud
import thermodynamics as td
import scipy.constants


#%% SETUP

# create dipolar fields
P1 = 0.5 # 0.442
fudge_1 = 0.67
P2 = 2 # 1.602
fudge_2 = 0.78
odt1 = fields.DipoleBeam(power=P1 * fudge_1, waist_y = 22.5E-6, waist_z=25.5E-6, theta_z = 0, theta_x = 0)
odt2 = fields.DipoleBeam(power=P2 * fudge_2, waist_y = 131.45E-6, waist_z=38.8E-6, theta_z = np.pi/2, theta_x = (np.pi/2 - 15 * np.pi/2))

field = fields.DipoleField()
field.add_beam(odt1)
field.add_beam(odt2)

# specify cloud parameters
N_particles = 1E6
temperature = 1E-6
mass_Er = 165.93 * scipy.constants.physical_constants['atomic mass constant'][0]
atoms = cloud.Cloud(temperature=temperature, N_particles = N_particles, mass = mass_Er, polarizability=-2.9228E-39)

Ud=td.calculate_potential_depth(field, atoms)
print(Ud / scipy.constants.k * 1E6)

#%% calculate peak density, psd and compare with the ideal harmonic case
'''
gamma_cutoff * k_b * T = the energy cutoff used to determine integration region
dl = size of integration cell  
'''
# determine dl based on typical lengthscale in the harmonic trap
harmonic_frequencies, eigen_directions = field.calculate_trapping_frequencies(atoms.calculate_intensity_prefactor(), atoms.m)
approx_thermal_size = np.sqrt(2 * scipy.constants.k * atoms.T / atoms.m) / harmonic_frequencies
dl = min(approx_thermal_size) / 10
# create integration grid
# testing showed that gamma = 10 is sufficient for 0.1% accuracy and so is dl = thermal size / 10
xx, yy, zz, dV = td.generate_integration_grid(field, atoms, gamma_cutoff=10, dl=dl)
grid = xx, yy, zz
box_size = np.array([np.max(xx), np.max(yy), np.max(zz)])
print("integration box size: ", box_size, "m")
# peak density
n0 = td.calculate_peak_density(grid, dV, field, atoms)
#peak density in the harmonic approximation
n0_harm = td.calculate_ideal_peak_density(field, atoms)

print("peak density is = ", n0,"m^-3, equivalent harmonic trap peak density = ", n0_harm,"m^-3")
print("peak PSD is = ", td.calculate_PSD(n0, atoms),"m^-3, equivalent harmonic trap peak density = ", td.calculate_PSD(n0_harm, atoms),"m^-3")


