# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 09:47:42 2022

@author: kucer
"""

import numpy as np

# field can consist of several components (think laser beams or independent coils), each of which has their value determined by a ramp
class Field:
    def __init__(self):
        pass
    
    # in general, any field can exert conservative force, change detuning of the energy levels or cause some heating 
    def calculate_force(self, positions, time, cloud):
        pass
    
    def calculate_detuning(self, positions, time, cloud):
        pass
    
    def calcualte_scattering_heating(self, positions, time, delta_t, cloud):
        pass
    
    
#%% RESONANT FIELD

class ResonantField(Field):
    def __init__(self):
        self.resonant_beams = []
    
    def add_beam(self, beam):
        self.resonant_beams.append(beam)
        
    def calculate_force(self, positions, time, cloud):
        pass
        
    def calculate_detuning(self, positions, time, cloud):
        return np.zeros(positions.shape)
    
    def calculate_scattering_heating(self, positions, time, cloud):
        pass
    
class ResonantBeam():
    def __init__(self, detuning_ramp = Ramp(), intensity_ramp = Ramp(), beam_direction = np.array([1,0,0]), polarisation = -1):
        self.detuning_ramp = detuning_ramp
        self.intensity_ramp = intensity_ramp
        self.polarisation = polarisation
        self.beam_direction = beam_direction
        
        ''' constant parameters for a given transition '''
        self.wavelength = 582.84E-9                 # m
        self.saturation_intensity = 1.3             # W / m**2
        self.transition_linewidth = 2*np.pi*190*1E3 # rad /s
        self.dipole_element_squared = self.transition_linewidth * 3 * np.pi * scipy.constants.epsilon_0 * \
                                      scipy.constants.hbar * (self.wavelength / (2 * np.pi))**3
                                      
        
    #TODO implement detuning calculation (probably doppler)
    #TODO implement scattering heating calculation
    #TODO implement scattering force with the knowledge of detuning
    #TODO check that the heating is correct
    #TODO somehow implement saturation from other beams
        
    
#%% DIPOLAR FIELD
    
class DipoleField(Field):
    def __init__(self):
        self.dipole_beams = []
    
    def add_beam(self, beam):
        self.dipole_beams.append(beam)
    
    def calculate_force(self, positions, time, cloud):
        total_force = np.zeros(positions.shape)
        for dipole_beam in self.dipole_beams:
            total_force = total_force + cloud.intensity_prefactor * dipole_beam.calculate_intensity_gradient(positions, time)
        return total_force
    
    # assume top level doesnt shift!
    # need to be careful with the sign!
    def calculate_detuning(self, positions, time, cloud):
        detuning = np.zeros(len(positions))
        for dipole_beam in self.dipole_beams:
            detuning = detuning - cloud.intensity_prefactor * dipole_beam.calculate_beam_intensity(positions, time)
        return detuning
    
    def calcualte_scattering_heating(self, positions, time, delta_t, cloud):
        heating_momenta = np.zeros(positions.shape)
        for dipole_beam in self.dipole_beams:
            number_of_scattering_events = delta_t * cloud.scattering_rate * dipole_beam.calculate_beam_intensity(positions, time)
            heating_momenta = heating_momenta + dipole_beam.calculate_scattering_heating(number_of_scattering_events)
        return heating_momenta
    
    # experimental feature, needs a bit more thinking. Can do either analytical and assume harmonic potential
    # or can do some kind of hessian calculation
    def calculate_trapping_frequencies(self, time, cloud):
        omega_squared_total = np.zeros(3)
        for dipole_beam in self.dipole_beams:
            M_transform = dipole_beam.get_transformation_matrix()
            omega_squared_beam = dipole_beam.calculate_trapping_frequencies(time, cloud)**2
            omega_squared_total += np.matmul(M_transform**2, omega_squared_beam)
        
        return np.sqrt(omega_squared_total)            
    
# TODO - add a way of changing the wavelength via the GUI
class DipoleBeam():
    def __init__(self, power_ramp = Ramp(), waist_y_ramp = Ramp(), waist_z_ramp = Ramp(), focus_ramp = Ramp()):
        self.wavelength = 1030E-9
        self.power_ramp = power_ramp
        self.waist_y_ramp = waist_y_ramp
        self.waist_z_ramp = waist_z_ramp
        self.focus_ramp = focus_ramp
        ''' The angles are defined such that the beam is first rotated by theta z about z
            and then theta x about x'''
        self.theta_z = 0
        self.theta_x = 0
        
        # TODO - see if caching local positions and waists helps
        ''' cached variables for faster computing '''
        ''' NOTE: caching requires more memory!! '''
        self.cache_time_intensity = None
        self.intensity_cache = None
    
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
    
    def calculate_beam_intensity(self, positions, time):
        if self.cache_time_intensity == time:
            return self.intensity_cache
        local_positions = self.convert_global_vectors_to_local(positions)
        rayleigh_range_y = self.calculate_rayleigh_range(self.waist_y_ramp.get_value(time))
        rayleigh_range_z = self.calculate_rayleigh_range(self.waist_z_ramp.get_value(time))
        
        # calculate the waists along the beam using the local x coordinate
        waists_y = self.calculate_waist(self.waist_y_ramp.get_value(time), rayleigh_range_y, local_positions[:,0])
        waists_z = self.calculate_waist(self.waist_z_ramp.get_value(time), rayleigh_range_z, local_positions[:,0])
        
        self.intensity_cache = self.power_ramp.get_value(time) * (2 / np.pi) * (1 / (waists_y * waists_z)) * \
                               np.exp(- 2 * np.power(local_positions[:,1], 2) / np.power(waists_y, 2)) * \
                               np.exp(- 2 * np.power(local_positions[:,2], 2) / np.power(waists_z, 2))
        self.cache_time_intensity = time
        return self.intensity_cache
    
    # the negative of the intensity gradient such that force is then F = alpha * (-grad I) since potential is U = alpha * I 
    # here alpha is the intensity prefactor equal to polarizability / (2 * c * epsilon0)
    def calculate_intensity_gradient(self, positions, time):
        local_positions = self.convert_global_vectors_to_local(positions)
        rayleigh_range_y = self.calculate_rayleigh_range(self.waist_y_ramp.get_value(time))
        rayleigh_range_z = self.calculate_rayleigh_range(self.waist_z_ramp.get_value(time))
        
        # calculate the waists along the beam using the local x coordinate
        waists_y = self.calculate_waist(self.waist_y_ramp.get_value(time), rayleigh_range_y, local_positions[:,0])
        waists_z = self.calculate_waist(self.waist_z_ramp.get_value(time), rayleigh_range_z, local_positions[:,0])
        
        gradient_y = (8 * self.power_ramp.get_value(time) / np.pi) * local_positions[:,1] * (1 / waists_z) * np.power(waists_y, -3) * \
                  np.exp(- 2 * np.power(local_positions[:,1], 2) / np.power(waists_y, 2)) * \
                  np.exp(- 2 * np.power(local_positions[:,2], 2) / np.power(waists_z, 2))
        
        gradient_z = (8 * self.power_ramp.get_value(time) / np.pi) * local_positions[:,2] * (1 / waists_y) * np.power(waists_z, -3) * \
                  np.exp(- 2 * np.power(local_positions[:,1], 2) / np.power(waists_y, 2)) * \
                  np.exp(- 2 * np.power(local_positions[:,2], 2) / np.power(waists_z, 2))
        
        gradient_x = (2 * self.power_ramp.get_value(time) / np.pi) * local_positions[:,0] * (1 / (waists_y * waists_z)) * \
                     np.exp(- 2 * np.power(local_positions[:,1], 2) / np.power(waists_y, 2)) * \
                     np.exp(- 2 * np.power(local_positions[:,2], 2) / np.power(waists_z, 2)) * \
                     (np.power(self.waist_y_ramp.get_value(time) / (waists_y * rayleigh_range_y), 2) * (1 - np.power(2 * local_positions[:,1] / waists_y, 2)) + \
                     np.power(self.waist_z_ramp.get_value(time) / (waists_z * rayleigh_range_z), 2) * (1 - np.power(2 * local_positions[:,2] / waists_z, 2)))
        
        # gradient vector in local coordinates
        gradient_local = np.stack((gradient_x, gradient_y, gradient_z), axis = 1)
        
        return self.convert_local_vectors_to_global(gradient_local)
        
    
    def calculate_scattering_heating(self, number_of_scattering_events):
        # its a biased (absorbtion) and random (emission) walk in momentum space with size of each step determined by momentum of scattered photon
        # generally the scattering is slow so the "biased" part of the walk gets rethermalized anyway and we only treat the overall effect on mometum magnitude i.e. on the temperature (see notes) 
        # based on this we calculate the magnitude of the heating momentum and we chose a random direction for it 
        heating_momentum_magnitude = scipy.constants.hbar * (2 * np.pi / self.wavelength) * np.sqrt(number_of_scattering_events * (number_of_scattering_events + 1))
        random_phi = np.random.rand(len(number_of_scattering_events)) * 2 * np.pi
        random_theta = np.arccos(1 - 2 * np.random.rand(len(number_of_scattering_events)))
        random_heating_momenta = heating_momentum_magnitude.reshape((len(number_of_scattering_events),1)) * \
                                 np.stack((np.sin(random_theta)*np.sin(random_phi), np.sin(random_theta)*np.cos(random_phi), np.cos(random_theta)), axis=1)
        return random_heating_momenta
    
    def calculate_trapping_frequencies(self, time, cloud):
        omega_x_squared = -(2 * cloud.intensity_prefactor * self.power_ramp.get_value(time)) / \
                           (np.pi * cloud.particle_mass * self.waist_y_ramp.get_value(time) * self.waist_z_ramp.get_value(time)) * \
                           (np.power(self.calculate_rayleigh_range(self.waist_y_ramp.get_value(time)),-2) + \
                            np.power(self.calculate_rayleigh_range(self.waist_z_ramp.get_value(time)),-2))
                   
        omega_y_squared = -(8 * cloud.intensity_prefactor * self.power_ramp.get_value(time)) / \
                           (np.pi * np.power(self.waist_y_ramp.get_value(time), 3) * self.waist_z_ramp.get_value(time) * cloud.particle_mass)
                           
        omega_z_squared = -(8 * cloud.intensity_prefactor * self.power_ramp.get_value(time)) / \
                           (np.pi * np.power(self.waist_z_ramp.get_value(time), 3) * self.waist_y_ramp.get_value(time) * cloud.particle_mass)
    
        return np.sqrt(np.array([omega_x_squared, omega_y_squared, omega_z_squared]))

#%% MAGNETIC FIELD
class UniformMagneticField:
    # initialize with empty ramps
    # empty ramp returns 0 when asked for value
    def __init__(self, x_ramp = Ramp(), y_ramp = Ramp(), z_ramp = Ramp()):
        self.x_ramp = x_ramp
        self.y_ramp = y_ramp
        self.z_ramp = z_ramp
        
    def calculate_field(self, positions, time):
        x_value = self.x_ramp.get_value(time)
        y_value = self.y_ramp.get_value(time)
        z_value = self.z_ramp.get_value(time)
        
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
        Bx = -gradient_value /2 * x_pos
        By = -gradient_value /2 * y_pos
        Bz =  gradient_value * z_pos
        
        return np.stack((Bx,By,Bz), axis=1)
    
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
        return np.zeros(positions.shape)