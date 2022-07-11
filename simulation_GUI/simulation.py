# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 15:17:24 2022

@author: kucer
"""
import numpy as np
import scipy.constants
from numba import jit

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
        self.file_name = "dummy"
    
    # there will be a gui button that will load data from gui into the simulation
    # and that will initialize the particles
    
    def propagate(self):
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
        return
        
    
    def run_simulation(self):
        sampling_timer = 0.0
        while(self.current_simulation_time < self.total_simulation_time):
            self.propagate()
            self.collide_particles()
            # maybe move the simulation box
            
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
def timeEvolutionBeeman(positions, momenta, old_positions, P1, waist_y1, waist_z1, next_waist_y1, next_waist_z1, previous_waist_y1, previous_waist_z1, gravity, delta_t, current_foc_pos, next_foc_pos, previous_foc_pos):
    old_forces = Forces(old_positions, P1, previous_waist_y1, previous_waist_z1, gravity, previous_foc_pos)
    current_forces = Forces(positions, P1, waist_y1, waist_z1, gravity, current_foc_pos)
    new_positions = positions + momenta * delta_t / particle_mass + 1/6 * ((4 * current_forces - old_forces) / particle_mass) * delta_t**2
    # new_momenta = np.zeros((len(new_positions, 3)))
    for i in range(2):
        new_forces = Forces(new_positions, P1, next_waist_y1, next_waist_z1, gravity, next_foc_pos)
        new_positions = positions + momenta / particle_mass * delta_t + 1/6 * (new_forces + 2* current_forces) / particle_mass * delta_t**2
        new_momenta = (new_positions - positions) * particle_mass / delta_t + 1/6 * (2 * new_forces + current_forces)*delta_t

    return new_positions, new_momenta, positions


class Field:
    def __init__(self):
        self.whateva = None

class Effector:
    def __init__(self):
        self.whateva = None

class Ramp:
    def __init__(self):
        self.type = None
        
    def get_value(self, time):
        return
        
# should have ramp subclass that inherits from ramp and that specifies the ramp?
# should define functions for the ramps on the side and then just pass them to the Ramp initilaizer as an argument

class AtomCloud:
    def __init__(self, number_of_simulated_particles, number_of_real_particles):
        self.positions = np.zeros((number_of_simulated_particles, 3))
        self.momenta = np.zeros((number_of_simulated_particles, 3))
        self.alpha = np.ones((number_of_simulated_particles, 3)) * (number_of_real_particles / number_of_simulated_particles)
        
        self.particle_mass = 167.9323702 * scipy.constants.physical_constants['atomic mass unit-kilogram relationship'][0]
        self.lande_g_ground = 1.163801
        self.mj_ground = -6
        self.lande_g_excited = 1.195
        self.mj_excited = -7
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

class SimulationBox:
    def __init__(self, center, size):
        self.center = center
        self.size = size
