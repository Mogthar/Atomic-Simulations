# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 22:46:34 2022

@author: kucer
"""

import numpy as np
    
class DipoleField():
    def __init__(self):
        self.dipole_beams = []
    
    def add_beam(self, beam):
        self.dipole_beams.append(beam)
    
    def calculate_intensity(self, positions):
        total_intensity = np.zeros(len(positions))
        for dipole_beam in self.dipole_beams:
            total_intensity += dipole_beam.calculate_beam_intensity(positions)
        return total_intensity
    
    def get_beams(self):
        return self.dipole_beams
    
    # experimental feature, needs a bit more thinking. Can do either analytical and assume harmonic potential
    # or can do some kind of hessian calculation
    def calculate_trapping_frequencies(self, intensity_prefactor, particle_mass):
        omega_tot_matrix = np.zeros((3,3))
        for dipole_beam in self.dipole_beams:
            M_transform = dipole_beam.get_transformation_matrix()
            omega_squared = (dipole_beam.calculate_trapping_frequencies(intensity_prefactor, particle_mass))**2
            omega_matrix = np.diag(omega_squared)
            omega_tot_matrix += np.linalg.multi_dot([M_transform, omega_matrix, np.transpose(M_transform)])
        eigen_val, eigen_vec = np.linalg.eig(omega_tot_matrix)
        return np.sqrt(eigen_val), eigen_vec            
    
class DipoleBeam():
    def __init__(self, power = 0.0, waist_y = 25E-6, waist_z = 25E-6, theta_z = 0, theta_x = 0):
        self.wavelength = 1030E-9
        self.power = power
        self.waist_y = waist_y
        self.waist_z = waist_z
        ''' The angles are defined such that the beam is first rotated by theta z about z
            and then theta x about x'''
        self.theta_z = theta_z
        self.theta_x = theta_x
    
    def get_transformation_matrix(self):
        R_z = np.array([[np.cos(self.theta_z), -np.sin(self.theta_z), 0],[np.sin(self.theta_z), np.cos(self.theta_z), 0],[0,0,1]])
        R_x = np.array([[1, 0, 0],[0, np.cos(self.theta_x), -np.sin(self.theta_x)],[0, np.sin(self.theta_x), np.cos(self.theta_x)]])
        return np.matmul(R_x, R_z)
    
    def convert_global_vectors_to_local(self, vectors):
        M_transform_inverse = np.transpose(self.get_transformation_matrix())
        return np.transpose(np.matmul(M_transform_inverse, np.transpose(vectors)))
    
    def convert_local_vectors_to_global(self, vectors):
        M_transform = self.get_transformation_matrix()
        return np.transpose(np.matmul(M_transform, np.transpose(vectors)))
    
    ## decide whether the input should be time or waist?
    def calculate_waist(self, waist_0, z_rayleigh, x):
        return waist_0 * np.sqrt(1 + (x / z_rayleigh)**2)
    
    def calculate_rayleigh_range(self, waist_0):
        return np.pi * (waist_0**2) / self.wavelength
    
    def calculate_beam_intensity(self, positions):
        local_positions = self.convert_global_vectors_to_local(positions)
        rayleigh_range_y = self.calculate_rayleigh_range(self.waist_y)
        rayleigh_range_z = self.calculate_rayleigh_range(self.waist_z)
        
        # calculate the waists along the beam using the local x coordinate
        waists_y = self.calculate_waist(self.waist_y, rayleigh_range_y, local_positions[:,0])
        waists_z = self.calculate_waist(self.waist_z, rayleigh_range_z, local_positions[:,0])
        
        intensity = self.power * (2 / np.pi) * (1 / (waists_y * waists_z)) * \
                    np.exp(- 2 * np.power(local_positions[:,1], 2) / np.power(waists_y, 2)) * \
                    np.exp(- 2 * np.power(local_positions[:,2], 2) / np.power(waists_z, 2))
        return intensity
    
    # the negative of the intensity gradient such that force is then F = alpha * (-grad I) since potential is U = alpha * I 
    # here alpha is the intensity prefactor equal to polarizability / (2 * c * epsilon0)
    def calculate_intensity_gradient(self, positions):
        local_positions = self.convert_global_vectors_to_local(positions)
        rayleigh_range_y = self.calculate_rayleigh_range(self.waist_y)
        rayleigh_range_z = self.calculate_rayleigh_range(self.waist_z)
        
        # calculate the waists along the beam using the local x coordinate
        waists_y = self.calculate_waist(self.waist_y, rayleigh_range_y, local_positions[:,0])
        waists_z = self.calculate_waist(self.waist_z, rayleigh_range_z, local_positions[:,0])
        
        gradient_y = (8 * self.power / np.pi) * local_positions[:,1] * (1 / waists_z) * np.power(waists_y, -3) * \
                     np.exp(- 2 * np.power(local_positions[:,1], 2) / np.power(waists_y, 2)) * \
                     np.exp(- 2 * np.power(local_positions[:,2], 2) / np.power(waists_z, 2))
        
        gradient_z = (8 * self.power / np.pi) * local_positions[:,2] * (1 / waists_y) * np.power(waists_z, -3) * \
                     np.exp(- 2 * np.power(local_positions[:,1], 2) / np.power(waists_y, 2)) * \
                     np.exp(- 2 * np.power(local_positions[:,2], 2) / np.power(waists_z, 2))
        
        gradient_x = (2 * self.power / np.pi) * local_positions[:,0] * (1 / (waists_y * waists_z)) * \
                     np.exp(- 2 * np.power(local_positions[:,1], 2) / np.power(waists_y, 2)) * \
                     np.exp(- 2 * np.power(local_positions[:,2], 2) / np.power(waists_z, 2)) * \
                     (np.power(self.waist_y / (waists_y * rayleigh_range_y), 2) * (1 - np.power(2 * local_positions[:,1] / waists_y, 2)) + \
                     np.power(self.waist_z / (waists_z * rayleigh_range_z), 2) * (1 - np.power(2 * local_positions[:,2] / waists_z, 2)))
        
        # gradient vector in local coordinates
        gradient_local = np.stack((gradient_x, gradient_y, gradient_z), axis = 1)
        
        return self.convert_local_vectors_to_global(gradient_local)
    
    def calculate_trapping_frequencies(self, intensity_prefactor, particle_mass):
        omega_x_squared = -(2 * intensity_prefactor * self.power) / \
                           (np.pi * particle_mass * self.waist_y * self.waist_z) * \
                           (np.power(self.calculate_rayleigh_range(self.waist_y),-2) + \
                            np.power(self.calculate_rayleigh_range(self.waist_z),-2))
                   
        omega_y_squared = -(8 * intensity_prefactor * self.power) / \
                           (np.pi * np.power(self.waist_y, 3) * self.waist_z * particle_mass)
                           
        omega_z_squared = -(8 * intensity_prefactor * self.power) / \
                           (np.pi * np.power(self.waist_z, 3) * self.waist_y * particle_mass)
    
        return np.sqrt(np.array([omega_x_squared, omega_y_squared, omega_z_squared]))
    