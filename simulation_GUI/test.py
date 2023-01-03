# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 14:29:11 2022

@author: kucer
"""
#%% imports
import simulation as sim
import fields
import ramps
import numpy as np
import matplotlib.pyplot as plt

#%% test a ramp
piecewise_ramp = ramps.Ramp()
piecewise_ramp.add_ramp_segment(ramps.LinearSegment(4,3,1))
piecewise_ramp.add_ramp_segment(ramps.ConstantSegment(3,3,2))
piecewise_ramp.add_ramp_segment(ramps.LinearSegment(3,5,0.5))
piecewise_ramp.add_ramp_segment(ramps.ExponentialSegment(5,2,3,-1))
piecewise_ramp.add_ramp_segment(ramps.GaussianNoiseSegment(2,2,2,0.1))

time = np.linspace(-2, 9, 100)
values = np.zeros(len(time))
for i, t in enumerate(time):
    values[i] = piecewise_ramp.get_value(t)

plt.scatter(time, values)
plt.show()

#%% test the simulation
# setup a small atom cloud in a box
atoms = sim.AtomCloud(10000, 10000000)
atoms.initialize_in_a_box(np.array([100E-6, 100E-6, 100E-6]))
atoms.thermalize_momenta(10.0E-6)

#%% setup the sim itself
simulation = sim.Simulation(atoms, "first_test")
simulation.plotter.plot_2D_image(5E-6, 1, image_size = (100, 200)) ## magnification 2, pixel size 5E-6

#%% initialize fields
no_ramp = ramps.Ramp()

#### B fields ####
feshbach_ramp = ramps.Ramp()
feshbach_ramp.add_ramp_segment(ramps.ConstantSegment(-1.16 * 2.03 * 1E-4, -1.16 * 2.03 * 1E-4, 100))
B_uniform = fields.UniformMagneticField(Bx_ramp=no_ramp, By_ramp=no_ramp, Bz_ramp=feshbach_ramp)

gradient_ramp = ramps.Ramp()
gradient_ramp.add_ramp_segment(ramps.ConstantSegment(8.3 * 100 * 1E-4, 8.3 * 100 * 1E-4, 100))
B_grad = fields.GradientMagneticField(gradient_ramp=gradient_ramp)

B_field = fields.MagneticField(uniform_field=B_uniform, gradient_field=B_grad)

#### ODT beams ####
P_odt1_ramp = ramps.Ramp()
P_odt1_ramp.add_ramp_segment(ramps.ConstantSegment(16,16,100))
wy_odt1_ramp = ramps.Ramp()
wy_odt1_ramp.add_ramp_segment(ramps.ConstantSegment(25E-6,25E-6,100))
wz_odt1_ramp = ramps.Ramp()
wz_odt1_ramp.add_ramp_segment(ramps.ConstantSegment(25E-6,25E-6,100))
focus_odt1_ramp = no_ramp
ODT_1 = fields.DipoleBeam(P_odt1_ramp, wy_odt1_ramp, wz_odt1_ramp, focus_odt1_ramp)

dipole_field = fields.DipoleField()
dipole_field.add_beam(ODT_1)

#### MOT beams ####
MOT_detuning_ramp = ramps.Ramp()
MOT_detuning_ramp.add_ramp_segment(ramps.ConstantSegment(-2*np.pi*5.0E6, -2*np.pi*5.0E6, 100))

MOT_power_ramp = ramps.Ramp()
MOT_power_ramp.add_ramp_segment(ramps.ConstantSegment(20E-6, 20E-6, 100)) ## 40E-6

MOT_beam_x_plus = fields.ResonantBeam(MOT_detuning_ramp, MOT_power_ramp, np.array([1,0,0]), polarisation=-1, waist=1E-2)
MOT_beam_x_minus = fields.ResonantBeam(MOT_detuning_ramp, MOT_power_ramp, np.array([-1,0,0]), polarisation=-1, waist=1E-2)
MOT_beam_y_plus = fields.ResonantBeam(MOT_detuning_ramp, MOT_power_ramp, np.array([0,1,0]), polarisation=-1, waist=1E-2)
MOT_beam_y_minus = fields.ResonantBeam(MOT_detuning_ramp, MOT_power_ramp, np.array([0,-1,0]), polarisation=-1, waist=1E-2)
MOT_beam_z_up = fields.ResonantBeam(MOT_detuning_ramp, MOT_power_ramp, np.array([0,0,1]), polarisation=1, waist=1E-2)
MOT_beam_z_down = fields.ResonantBeam(MOT_detuning_ramp, MOT_power_ramp, np.array([0,0,-1]), polarisation=1, waist=1E-2)

resonant_field = fields.ResonantField()
resonant_field.add_beam(MOT_beam_x_minus)
resonant_field.add_beam(MOT_beam_x_plus)
resonant_field.add_beam(MOT_beam_y_minus)
resonant_field.add_beam(MOT_beam_y_plus)
resonant_field.add_beam(MOT_beam_z_down)
resonant_field.add_beam(MOT_beam_z_up)

#%% add fields to simulation and specify other sim parameters
simulation.fields.append(B_field)
simulation.fields.append(resonant_field)
simulation.fields.append(dipole_field)

simulation.delta_t = 1E-6
simulation.total_simulation_time = 5E-3

simulation.sampling_delta_t = 1E-3
simulation.plot_it = True

simulation.gravity = True

simulation.collisions_on = True
simulation.simulation_box = sim.SimulationBox(ramps.Ramp(), 0.0006 * np.ones(3))

#%% run simulation
simulation.run_simulation()