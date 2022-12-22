# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 14:29:11 2022

@author: kucer
"""
#%% imports
import simulation as sim
import numpy as np
import plotting_utils as pltutils
import ramps
import matplotlib.pyplot as plt

#%% test a ramp
piecewise_ramp = ramps.Ramp()
piecewise_ramp.add_ramp_segment(ramps.LinearSegment(4,3,1))
piecewise_ramp.add_ramp_segment(ramps.ConstantSegment(3,3,2))
piecewise_ramp.add_ramp_segment(ramps.LinearSegment(3,5,0.5))
piecewise_ramp.add_ramp_segment(ramps.ExponentialSegment(5,2,3,-1))

time = np.linspace(-2, 8, 100)
values = np.zeros(len(time))
for i, t in enumerate(time):
    values[i] = piecewise_ramp.get_value(t)

plt.scatter(time, values)
plt.show()
#%%

cloud = sim.AtomCloud(10000, 1E6)

# initialize fields
dipole_field = sim.DipoleField()
dipole_beam_horizontal = sim.DipoleBeam()
dipole_field.add_beam(dipole_beam_horizontal)

# define beam ramps
start_time = -100
end_time = 100

power_ramp_segment = ramps.RampSegment()
power_ramp_segment.start_time = start_time
power_ramp_segment.end_time = end_time
power_ramp_segment.ramp_type = ramps.ConstantRamp
power_ramp_segment.start_value = 12.0
dipole_beam_horizontal.power_ramp.add_ramp_segment(power_ramp_segment) 

waisty_ramp_segment = ramps.RampSegment()
waisty_ramp_segment.start_time = start_time
waisty_ramp_segment.end_time = end_time
waisty_ramp_segment.ramp_type = ramps.ConstantRamp
waisty_ramp_segment.start_value = 24E-6
dipole_beam_horizontal.waist_y_ramp.add_ramp_segment(waisty_ramp_segment)

waistz_ramp_segment = ramps.RampSegment()
waistz_ramp_segment.start_time = start_time
waistz_ramp_segment.end_time = end_time
waistz_ramp_segment.ramp_type = ramps.ConstantRamp
waistz_ramp_segment.start_value = 21E-6
dipole_beam_horizontal.waist_z_ramp.add_ramp_segment(waistz_ramp_segment)

# thermalize cloud
T = 1E-6
cloud.thermalize_momenta(T)
cloud.thermalize_positions(T, dipole_field, 0.0)

pltutils.xz_summed_image(cloud.positions, 5.8E-6, 1.944, image_size=(200, 200))