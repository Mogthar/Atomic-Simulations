# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 15:17:24 2022

@author: kucer
"""
import numpy as np
import scipy.constants
from numba import jit
import ramps

class Simulation:
    def __init__(self):
        self.atom_cloud = AtomCloud(100, 100)
        self.simulation_box = SimulationBox(np.zeros(3), np.ones(3))
        self.fields = []
        self.effectors = []
        self.gravity = True
        self.delta_t = 1E-6
        self.sampling_delta_t = 1E-3
        self.total_simulation_time = 1.0
        self.current_simulation_time = 0.0
        self.save_file_name = "dummy"
    
    # there will be a gui button that will load data from gui into the simulation
    # and that will initialize the particles
    
    def propagate(self):
        # old_forces = Forces(positions, P1, waist_y1, waist_z1, gravity, current_foc_pos)
        old_forces = self.calculate_forces(self.atom_cloud.positions, self.current_simulation_time)
    
        new_positions = self.atom_cloud.positions + \
                        self.atom_cloud.momenta * self.delta_t / self.atom_cloud.particle_mass + \
                        1/2 * old_forces / self.atom_cloud.particle_mass * (self.delta_t**2)
        # is this line necessary? maybe instead of new positions above just redefine the atom_cloud positions                
        self.atom_cloud.positions = new_positions
        
        new_forces = self.calculate_forces(self.atom_cloud.positions, self.current_simulation_time + self.delta_t)
        
        new_momenta = self.atom_cloud.momenta + 1/2 * (old_forces + new_forces) * self.delta_t
        # same comment as above
        self.atom_cloud.momenta = new_momenta
    
    def calculate_forces(self, particle_positions, time):
        # go through all the field objects
        # calculate the forces
        # first detunings, then forces
        # dont forget about gravityyyyy
        return
    
    def collide_particles(self):
        return
    
    def recalculate_delta_t(self):
        return
    
    def initialize_cloud_from_gui(self, settings):
        return
    
    def initialize_cloud_from_file(self, file):
        return
    
    def sample_trajectory(self, file):
        # sample positons, particle number, temperature, time
        return
    
    def save_final_configuration(self, file):
        # save final positions and momenta
        return 
    
    def save_simulation_settings(self, file):
        # save simulation settings
        # this should include all the ramp info etc
        return
    
    def check_ramps(self):
        # check that all ramps are defined and have appropriate length
        # if not then raise error and maybe fill the rest of the ramp with a constant ramp
        return
    
    def run_simulation(self):
        sampling_timer = 0.0
        while(self.current_simulation_time < self.total_simulation_time):
            self.propagate()
            self.collide_particles()
            # maybe move the simulation box or have simulation box also follow a ramp! 
            
            self.current_simulation_time += self.delta_t
            sampling_timer += self.delta_t
            self.recalculate_delta_t()
            
            if sampling_timer > self.sampling_delta_t:
                self.sample_trajectory(self.file_name + ".traj")
                sampling_timer = 0.0
        
        self.sample_trajectory(self.file_name + ".traj")
        self.save_final_configuration(self.file_name + ".fin")
        self.save_simulation_settings(self.file_name + ".set")
    
# each effector/field should have a calc force, calc shift method
# each effector should have it own ramp object!!!

@jit(nopython=True, parallel=True)
def Forces(positions, P1, waist_y1, waist_z1, gravity, focal_position):
    force_odt1 = calc_force1(positions, waist_y1, waist_z1, P1, focal_position)
    if (gravity):
        gravitational_force = np.zeros(positions.shape)
        gravitational_force[:,2] = - particle_mass * scipy.constants.g
        return force_odt1 + gravitational_force
    else:
        return force_odt1
    
#this needs redoing because scattering is along the beam 
@jit(nopython=True, parallel=True)
def ScatteringHeating(n_scatt_events, wavelength):
    momentum_magnitude = scipy.constants.hbar * (2 * np.pi / wavelength) * np.sqrt(n_scatt_events)
    random_phi = np.random.rand(len(n_scatt_events)) * 2 * np.pi
    random_theta = np.arccos(1 - 2 * np.random.rand(len(n_scatt_events)))
    random_momenta = momentum_magnitude.reshape((len(n_scatt_events),1)) * np.stack((np.sin(random_theta)*np.sin(random_phi),np.sin(random_theta)*np.cos(random_phi),np.cos(random_theta)), axis=1)
    return random_momenta

@jit(nopython=True, parallel=True)
def ScatteringForce(MOT_intensity, detuning, B_field, position, momentum, P1, P2, waist_y1, waist_z1, waist_y2, waist_z2, angle1, angle2, time, dith_ampl, del_t):
    # calculate local detuning i.e. magnetic field and stark
    B_magnitude = np.sqrt((B_field*B_field).sum(axis=1))
    magnetic_detuning = -MagneticFieldShift(B_field, mj_excited) / scipy.constants.hbar
    stark_detuning = -StarkShift(P1, P2, position, waist_y1, waist_z1, waist_y2, waist_z2, angle1, angle2, time, dith_ampl) / scipy.constants.hbar
    local_detuning = detuning + magnetic_detuning + stark_detuning
    
    # could be taken out and made into a constant
    scattering_force_prefactor = scipy.constants.hbar * MOT_linewidth * np.pi / MOT_wavelength
    # loop through beams
    total_scattering_force = np.zeros(position.shape)
    total_scattering_events = np.zeros(len(position))
    for i in range(len(MOT_beam_direction)):
        # the doppler detuning is -k*v
        doppler_detuning = - 2 * np.pi / (MOT_wavelength) * np.dot(momentum / particle_mass, MOT_beam_direction[i])
        # calculate s parameter. 2 rabi**2 / gamma**2
        # angle between beam and magnetic field
        # the polarisation comes in because the beam in the reference frame of the field doesnt necessarily look
        # like circularly polarized; only the sigma - component of the beam can drive the transition and thats basically
        # what is being worked out here; since the atoms sit bellow the zero of the field i.e. they are always in a region
        # where all the other mj states are just too far detuned because of zeeman shift, we only care about the -7 transition
        # and hence only the sigma - component of each beam!!!!
        cos_B_k = np.dot(B_field, MOT_beam_direction[i]) / B_magnitude
        s_prefactor = 4 * MOT_intensity[i] / (scipy.constants.c * scipy.constants.epsilon_0 * (scipy.constants.hbar * MOT_linewidth)**2)
        s_dipole = dipole_elem_squared * ((1 - cos_B_k * MOT_beam_polarisation[i])**2) / 4 
        s = s_prefactor * s_dipole
        scatt_frac = ScatteringFraction(s, local_detuning + doppler_detuning)
        total_scattering_events += MOT_linewidth / 2 * scatt_frac * del_t
        total_scattering_force += scattering_force_prefactor * MOT_beam_direction[i] * scatt_frac.reshape((len(position),1))
        # total_scattering_force += MOT_beam_direction[i] * total_scattering_events.reshape((len(position),1)) * (2 * scattering_force_prefactor / (del_t * MOT_linewidth))
        # Milan's thesis suggests using s' because the other beams deplete the lower level but we are in the low s regime anyway
        # so the depletion is not that strong and there are always atoms available
    
    return total_scattering_force, total_scattering_events


# instead of positions and cloud separately just pass in the cloud? doesnt matter I guess
class Field:
    def __init__(self):
        pass
    
    def calculate_force(self, positions, time, cloud):
        pass
    
    def calculate_detuning(self, positions, time, cloud):
        pass
    
    def calcualte_scattering_heating(self, positions, time, cloud):
        pass
    
    
class DipoleField(Field):
    def __init__(self):
        self.dipole_beams = []
    
    def add_beam(self, beam):
        self.dipole_beams.append(beam)
    
    def calculate_force(self, positions, time, cloud):
        pass
    
    def calculate_detuning(self, positions, time, cloud):
        pass
    
    def calcualte_scattering_heating(self, positions, time, cloud):
        pass
    
    def calculate_trapping_frequencies():
        pass
    
# TODO - add a way of changing the wavelength via the GUI
class DipoleBeam():
    def __init__(self, beam_ramp = Ramp(), waist_ramp = Ramp(), focus_ramp = Ramp()):
        self.wavelength = 1030E-9
        self.beam_ramp = beam_ramp
        self.waist_ramp = waist_ramp
        self.focus_ramp = focus_ramp
        self.theta_z = 0
        self.theta_x = 0
    
    def rotate_beam(self, theta_z, theta_x):
        self.theta_z = theta_z
        self.theta_x = theta_x

    
    def reset_beam_rotation(self):
        self.theta_x = 0
        self.theta_z = 0
    
    def get_transformation_matrix(self):
        R_z = np.array([[np.cos(self.theta_z), -np.sin(self.theta_z), 0],[np.sin(self.theta_z), np.cos(self.theta_z), 0],[0,0,1]])
        R_x = np.array([[1, 0, 0],[0, np.cos(self.theta_x), -np.sin(self.theta_x)],[0, np.sin(self.theta_x), np.cos(self.theta_x)]])
        return np.matmul(R_x, R_z)
    
    def convert_global_coordinates_to_local(self, positions):
        M_transform_inverse = np.transpose(self.get_transformation_matrix())
        return np.transpose(np.matmul(M_transform_inverse, np.transpose(positions)))
        
    def calculate_waist(self, waist_0, z_rayleigh, x):
        return waist_0 * np.sqrt(1 + (x / z_rayleigh)**2)
    
    def calculate_rayleigh_range(self, waist_0):
        return np.pi * (waist_0**2) / self.wavelength
    
    
    
    def calculate_detuning(self, positions, time, cloud):
    
        
class ResonantField(Field):
class ResonantBeam():
    

    
###### MAGNETIC FIELD ######
    
class MagneticField(Field):
    def __init__(self, uniform_field = UniformMagneticField(), gradient_field = GradientMagneticField()):
        self.uniform_field = uniform_field
        self.gradient_field = gradient_field
    
    # because of the order of operations can possibly chache the field magnitude
    def calculate_detuning(self, positions, time, cloud):
        field = self.uniform_field.calculate_field(positions, time) + self.gradient_field.calculate_field(positions, time)
        field_magnitude = np.sqrt((field * field).sum(axis = 1))
        return cloud.get_Zeeman_shift_prefactor() * field_magnitude
        
    def calculate_force(self, positions, time, cloud):
        field = self.uniform_field.calculate_field(positions, time) + self.gradient_field.calculate_field(positions, time)
        field_magnitude = np.sqrt((field * field).sum(axis = 1))
        gradient_value = self.gradient_field.gradient_ramp.get_value(time)
        
        magnitude_gradient_x = field[:,0] * (-gradient_value / 2) / field_magnitude
        magnitude_gradient_y = field[:,0] * (-gradient_value / 2) / field_magnitude
        magnitude_gradient_z = field[:,0] * (gradient_value) / field_magnitude
        
        return cloud.get_ground_state_moment() * np.stack((magnitude_gradient_x, magnitude_gradient_y, magnitude_gradient_z), axis=1)
        
    def calculate_scattering_heating(self, positions, time, cloud):
        pass
    
class UniformMagneticField:
    # initialize with empty ramps
    # empty ramp returns 0 when asked for value
    def __init__(self, x_ramp = Ramp(), y_ramp = Ramp(), z_ramp = Ramp()):
        self.x_ramp = x_ramp
        self.y_ramp = y_ramp
        self.z_ramp = z_ramp
        
    def calculate_field(self, positions, time)
        x_value = x_ramp.get_value(time)
        y_value = y_ramp.get_value(time)
        z_value = z_ramp.get_value(time)
        
        field = np.array([[x_value, y_value, z_value]])
        
        return np.repeat(field, np.size(positions, axis = 0), axis = 0)
    
class GradientMagneticField:
    def __init__(self, gradient_ramp = Ramp()):
        self.gradient_ramp = gradient_ramp
    
    def calculate_field(self, positions, time):
        x_pos = positions[:,0]
        y_pos = positions[:,1]
        z_pos = positions[:,2]
        
        gradient_value = self.gradient_ramp.get_value(time)
        Bx = -gradient_value /2 * x
        By = -gradient_value /2 * y
        Bz =  gradient_value * z
        
        return np.stack((Bx,By,Bz), axis=1)

####### RAMPS #######

class Ramp:
    def __init__(self):
        self.ramp_segments = []
        
    def add_ramp_segment(self, ramp_segment):
        self.ramp_segments.append(ramp_segment)
        
    # empty ramp returns 0 when asked for value
    def get_value(self, time):
        value = 0.0
        for ramp_segment in self.ramp_segments:
            if(ramp_segment.is_time_in_segment(time)):
                value += ramp_segment.get_value(time)
        return value

class RampSegment:
    def __init__(self):
        self.start_time = 0
        self.end_time = 0
        self.ramp_type = None
        self.start_value = 0
        self.end_value = 0
    
    def is_time_in_segment(self, time):
        if(time >= self.start_time and time <= self.end_time):
            return True
        else:
            return False
    
    def get_value(self, time):
        return self.ramp_type(time - self.start_time, self.end_time - self.start_time, self.start_value, self.end_value)
    
####### CLOUD ######

class AtomCloud:
    # try and initilize as many of these as possible so that we get no errors!
    # 
    def __init__(self, number_of_simulated_particles, number_of_real_particles):
        self.positions = np.zeros((number_of_simulated_particles, 3))
        self.momenta = np.zeros((number_of_simulated_particles, 3))
        self.alpha = np.ones((number_of_simulated_particles, 3)) * (number_of_real_particles / number_of_simulated_particles)
        
        self.particle_mass = 167.9323702 * scipy.constants.physical_constants['atomic mass unit-kilogram relationship'][0]
        
        self.lande_g_ground = 1.163801
        self.mj_ground = -6
        self.lande_g_excited = 1.195
        self.mj_excited = -7
        self.magneton = scipy.constants.physical_constants['Bohr magneton'][0]
        
        self.scatt_cross_section = 8 * np.pi * (150 * scipy.constants.physical_constants['atomic unit of length'][0])**2
    
    # initilaize momenta from boltzmann distribution at certain temperature    
    def thermalize_momenta(self, temperature):
        self.momenta = self.particle_mass  * np.random.normal(loc = 0.0, 
                                                              scale = np.sqrt(scipy.constants.k * temperature / self.particle_mass), 
                                                              size = (len(self.positions), 3))
    
    # initializes particles in a thermal equilibrium in a given potential using harmonic approximation     
    def thermalize_positions(self, temperature, potential):
        omega_x, omega_y, omega_z = potential.get_trap_frequencies()
        omega_squared = np.array([omega_x**2, omega_y**2, omega_z**2])
        self.positions = np.random.normal(loc = 0.0, 
                                          scale = np.sqrt(scipy.constants.k * temperature / (self.particle_mass * omega_squared)), 
                                          size = (len(self.positions), 3))
        
    def calculate_cloud_temperature(self, bounding_box):
        # bounding box defined by center and size 
        culling_mask = np.logical_not(np.any((self.positions > (bounding_box.center + bounding_box.size / 2)) | 
                                            (self.positions < (bounding_box.center - bounding_box.size / 2))))
        
        remaining_momenta = self.momenta[culling_mask]
        momenta_magnitude = np.linalg.norm(remaining_momenta, axis=1)
        return np.average(momenta_magnitude**2) / (3 * self.particle_mass * scipy.constants.k)
    
    def get_Zeeman_shift_prefactor(self):
        return self.magneton * (self.mj_excited * self.lande_g_excited - self.mj_ground * self.lande_g_ground)
    
    def get_ground_state_moment(self):
        return - self.magneton * self.lande_g_ground * self.mj_ground
        

class SimulationBox:
    def __init__(self, center, size):
        self.center = center
        self.size = size
        # have the simulation box follow the same ramp as the odt
        self.ramp = None
