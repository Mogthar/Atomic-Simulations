# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 15:17:24 2022

@author: kucer
"""
import numpy as np
import scipy.constants
from numba import jit
from ramps import Ramp
import fields
import os
import plotting_utils as plutils
import time


#TODO - define plotter and saver classes 
# TODO - parallelisation
# TODO - stats -define a separate file with class and simulation
# NOTE: we are actually in high field seeking state! U = -mu B and for electrons mu = -g * bohr * mj
# so we have U = g * bohr * B_magnitude * mj and we have mj < 0 !!
# TODO - add method into simulation itself to initialize box at a given number of beam waists/ rayleigh ranges
# TODO - implement scattering cross section as a function of B field in the cloud class
# TODO - change plotter such that when it is defined it is given an image and pixel size
# TODO - deal with box, masking etc
# TODO - implement particle multiplication
# TODO - implement particle collision

##### PERFORMANCE NOTES
# NOTE is better with numpy arrays to use array = array + another array

class Simulation:
    def __init__(self, atom_cloud, simulation_name):
        self.atom_cloud = atom_cloud
        # initialize with simple simulation box
        self.simulation_box = SimulationBox(Ramp(), np.ones(3))
        self.gamma_cell = 1.0 # fraction of mean free path used as box sizes
        self.fields = []
        self.gravity = True
        
        ''' time related parameters'''
        self.delta_t = 1E-6
        self.gamma_t_oscillation = 1E-3
        self.gamma_t_collision = 0.1
        self.sampling_delta_t = 1E-3
        self.collision_delta_t = 1E-3
        
        self.total_simulation_time = 100E-3
        self.current_simulation_time = 0.0
        
        self.name = simulation_name
        self.data_sampler = DataSampler(self, self.name)
        
        self.plotter = Plotter(self)
        self.plot_it = False
    
    # TODO - need to be careful about the order of things here. How does scattering momentum play in? 
    # since its a non-conservative foce I will leave it out of the verlet algorithm
    def propagate(self):
        old_forces = self.calculate_forces(self.atom_cloud.positions, self.current_simulation_time)
    
        new_positions = self.atom_cloud.positions + \
                        self.delta_t / self.atom_cloud.particle_mass * self.atom_cloud.momenta + \
                        1/2 * (self.delta_t**2) / self.atom_cloud.particle_mass * old_forces
                     
        self.atom_cloud.positions = new_positions
        
        new_forces = self.calculate_forces(self.atom_cloud.positions, self.current_simulation_time + self.delta_t)
        
        # calculate necessary quantites for the non-conservative scattering force
        scattering_momenta = np.zeros(self.atom_cloud.positions.shape)
        detunings = np.zeros(len(self.atom_cloud.positions))
        B_field_directions = np.zeros(self.atom_cloud.positions.shape)       
        for field in self.fields:
            detunings = detunings + \
                        field.calculate_detuning(self.atom_cloud.positions, self.atom_cloud.momenta, self.current_simulation_time, self.atom_cloud)
            if isinstance(field, fields.MagneticField):
                B_field_directions = field.calculate_field_direction(self.atom_cloud.positions, self.current_simulation_time)
        for field in self.fields:
            scattering_momenta = scattering_momenta + \
                                 field.calculate_scattering_momenta(self.atom_cloud.positions, self.atom_cloud.momenta, 
                                                                    B_field_directions, detunings, self.current_simulation_time, 
                                                                    self.delta_t, self.atom_cloud)
        
        new_momenta = self.atom_cloud.momenta + 1/2 * (old_forces + new_forces) * self.delta_t + scattering_momenta
        self.atom_cloud.momenta = new_momenta
    
    def calculate_forces(self, positions, time):
        forces = np.zeros(positions.shape)
        for field in self.fields:
            forces = forces + field.calculate_force(positions, time, self.atom_cloud)
        if self.gravity:
            grav_force = np.array([0,-9.81 * self.atom_cloud.particle_mass,0])
            forces = forces + grav_force
        return forces
    
    def collide_particles(self):
        return
    
    # TODO add collision based dt
    def recalculate_delta_t(self):
        oscillation_dt = self.delta_t
        for field in self.fields:
            if isinstance(field, fields.DipoleField):
                trappinq_freq, _ = field.calculate_trapping_frequencies(self.current_simulation_time, self.atom_cloud)
                maximum_trapping_frequency = np.amax(trappinq_freq)
                oscillation_dt = self.gamma_t_oscillation * 2 * np.pi / maximum_trapping_frequency
        self.delta_t = oscillation_dt
    
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
    
    def run_simulation(self):
        sampling_timer = 0.0
        collision_timer = 0.0
        # first calculate delta t
        self.recalculate_delta_t()
        
        t_start = time.time()
        while(self.current_simulation_time < self.total_simulation_time):
            self.propagate()
            
            self.current_simulation_time += self.delta_t
            sampling_timer += self.delta_t
            collision_timer += self.delta_t
            
            if collision_timer > self.collision_delta_t:
                self.collide_particles()
                collision_timer -= self.collision_delta_t
            
            if sampling_timer > self.sampling_delta_t:
                self.data_sampler.sample_data()
                sampling_timer -= self.sampling_delta_t
                print(100 * self.current_simulation_time / self.total_simulation_time)
                print(self.atom_cloud.calculate_cloud_temperature(self.simulation_box, self.current_simulation_time))
                if self.plot_it:
                    self.plotter.plot_2D_image(5E-6, 2, image_size = (200, 200))
        
            # recalculate dt based on new trapping frequencies
            self.recalculate_delta_t()
        t_end = time.time()
        print(t_end - t_start)
        
        self.data_sampler.save_final_info()
        
    ''' set of methods that calculate cloud properties given the trapping fields '''
    
    def calculate_average_density_harmonicApprox(self):
        omega_bar = 0.0
        for field in self.fields:
            if isinstance(field, fields.DipoleField):
                trapping_frequencies = field.calculate_trapping_frequencies(self.current_simulation_time, self.atom_cloud)
                omega_bar = np.power(trapping_frequencies[0] * trapping_frequencies[1] * trapping_frequencies[2], 1/3)
        
        number_of_particles = self.atom_cloud.get_real_particle_number()
        cloud_temperature = self.atom_cloud.calculate_cloud_temperature()
        average_density = 2**(-3/2) * number_of_particles * omega_bar**3 * \
                          np.power(self.atom_cloud.particle_mass / (2 * np.pi * scipy.constants.k * cloud_temperature), 3/2)
        return average_density

#%% DATA SAMPLER
class DataSampler:
    def __init__(self, simulation, file_name):
        self.simulation = simulation
        self.file_name = file_name
        
    '''
    def sample_trajectory(self):
        if not os.path.isdir(self.file_name):
            os.mkdir(self.file_name)
        else:
            path = self.file_name + "/trajectory.xyz"
            positions = self.simulation.atom_cloud.positions
            with open(path, "a") as f:
                particle_number_string = str(len(positions))
                f.write(particle_number_string)
                f.write('\n')
                f.write("frame")
                f.write('\n')
                for i in range(len(positions)):
                    atom_name = "atom%d " %(i)
                    position_string = "%.9f %.9f %.9f" %(positions[i][0],positions[i][1],positions[i][2])
                    f.write(atom_name + position_string)
                    f.write('\n')
                #for j in range(defaultLen - len(positions)):
                 #   atom_name = "atom%d " %(j + len(positions))
                  #  position_string = "%.9f %.9f %.9f" %(0.0,0.0,0.0)
                   # f.write(atom_name + position_string)
                    #f.write('\n')
    '''
    def sample_trajectory(self):
        pass
    
    def sample_physical_quantities(self):
        pass
    
    def sample_data(self):
        self.sample_trajectory()
        self.sample_physical_quantities()
    
    def save_simulation_settings(self):
        pass

    
    def save_final_info(self):
        self.sample_trajectory()
        self.sample_physical_quantities()
        self.save_simulation_settings()
        pass
    
#%% PLOTTER #####
class Plotter:
    def __init__(self, simulation):
        self.simulation = simulation

    def plot_2D_image(self, pixel_size, magnification, image_size):
        plutils.xz_summed_image(self.simulation.atom_cloud.positions, pixel_size, magnification, image_size)
    
#%% SIMULATIONN BOX #####

class SimulationBox:
    ''' center position stores a vector pointing to the box center, 
    size stores a vector of the form (length, width, height)'''
    
    def __init__(self, center_position_ramp, size):
        self.center_position_ramp = center_position_ramp
        self.size = size
    
    def set_box_size(self, x_size, y_size, z_size):
        self.size = np.array([x_size, y_size, z_size])
        
    def get_center_position(self, time):
        return np.array([self.center_position_ramp.get_value(time), 0.0, 0.0])
    
#%% CLOUD
# checked 3.10.2022

class AtomCloud:
    def __init__(self, number_of_simulated_particles, number_of_real_particles):
        self.positions = np.zeros((number_of_simulated_particles, 3))
        self.momenta = np.zeros((number_of_simulated_particles, 3))
        self.alpha = np.ones(number_of_simulated_particles) * (number_of_real_particles / number_of_simulated_particles)
        
        self.particle_mass = 167.9323702 * scipy.constants.physical_constants['atomic mass unit-kilogram relationship'][0]
        
        self.lande_g_ground = 1.163801
        self.mj_ground = -6
        self.lande_g_excited = 1.195
        self.mj_excited = -7
        self.magneton = scipy.constants.physical_constants['Bohr magneton'][0]
        
        #TODO check this a0 value
        self.scatt_cross_section = 8 * np.pi * (78 * scipy.constants.physical_constants['atomic unit of length'][0])**2  # need to check this!!
        self.polarizability = 2.9228099E-39
        self.intensity_prefactor = - self.polarizability / (2 * scipy.constants.epsilon_0 * scipy.constants.c)
        self.scattering_rate = 1.79341521457477E-10   # in units of s-1 per W/m2 i.e. multiply by intensity to get the real scattering rate
    
    # initilaize momenta from boltzmann distribution at certain temperature    
    def thermalize_momenta(self, temperature):
        self.momenta = self.particle_mass  * np.random.normal(loc = 0.0, 
                                                              scale = np.sqrt(scipy.constants.k * temperature / self.particle_mass), 
                                                              size = (len(self.positions), 3))
    
    # initializes particles in a thermal equilibrium in a given potential using harmonic approximation     
    def thermalize_positions(self, temperature, dipole_field, time):
        omega = dipole_field.calculate_trapping_frequencies(time, self)
        self.positions = np.random.normal(loc = 0.0, 
                                          scale = np.sqrt(scipy.constants.k * temperature / (self.particle_mass * (omega**2))), 
                                          size = (len(self.positions), 3))
        
    def initialize_in_a_box(self, box_dimensions):
        self.positions = box_dimensions * (np.random.rand(*self.positions.shape) - 0.5)
        
    def calculate_cloud_temperature(self, bounding_box, time):
        # bounding box defined by center and size 
        culling_mask = np.logical_not(np.any((self.positions > (bounding_box.get_center_position(time) + bounding_box.size / 2)) | 
                                            (self.positions < (bounding_box.get_center_position(time) - bounding_box.size / 2)), axis=1))
        remaining_momenta = self.momenta[culling_mask]
        momenta_magnitude = np.linalg.norm(remaining_momenta, axis=1)
        return np.average(momenta_magnitude**2) / (3 * self.particle_mass * scipy.constants.k)
    
    ### utility functions 
    def get_real_particle_number(self):
        return np.sum(self.alpha)
    
    def get_ground_state_moment(self):
        return - self.magneton * self.lande_g_ground * self.mj_ground
    
    def get_excited_state_moment(self):
        return - self.magneton * self.lande_g_excited * self.mj_excited
    