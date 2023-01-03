# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 09:47:42 2022

@author: kucer
"""

import numpy as np
import scipy.constants
from abc import ABC, abstractmethod
import ramps

# TODO paralelization

# field can consist of several components (think laser beams or independent coils), each of which has their value determined by a ramp
class Field(ABC):
    def __init__(self):
        pass
    
    # in general, any field can exert conservative force, change detuning of the energy levels or cause some heating 
    @abstractmethod
    def calculate_force(self, positions, time, cloud):
        pass
    
    # detuning is calculated as delta = w - w0 = w - E_high + E_low
    # some fields shift only E_low, some fields shift both
    @abstractmethod
    def calculate_detuning(self, positions, momenta, time, cloud):
        pass
    
    # this is due to resonnant and off resonant scattering events
    @abstractmethod
    def calculate_scattering_momenta(self, positions, momenta, B_field_direction, local_detunings, time, delta_t, cloud):
        pass
    
    
#%% RESONANT FIELD

class ResonantField(Field):
    def __init__(self):
        self.resonant_beams = []
    
    def add_beam(self, beam):
        self.resonant_beams.append(beam)
        
    # resonant beams dont exert a conservative force
    def calculate_force(self, positions, time, cloud):
        return np.zeros(positions.shape)
    
    # resonant beams dont cause local detunings
    def calculate_detuning(self, positions, momenta, time, cloud):
        return np.zeros(len(positions))
    
    def calculate_scattering_momenta(self, positions, momenta, B_field_direction, local_detunings, time, delta_t, cloud):
        # first calculate saturation parameters
        # TODO can use parallelization here buthave to be careful about order of beams and things in the list. Maybe instead of list have an np array
        saturation_params = []
        for resonant_beam in self.resonant_beams:
            saturation_params.append(resonant_beam.calculate_saturation_parameter(positions, B_field_direction, time))
        
        # TODO most likely not worth parallelizing 
        combined_saturation = np.zeros(len(positions))
        for s in saturation_params:
            combined_saturation = combined_saturation + s
        
        # TODO suitable for parallelization. Maybe zip together saturaion params and beams
        scattering_momenta = np.zeros(positions.shape)
        for i, resonant_beam in enumerate(self.resonant_beams):
            s = saturation_params[i]
            # beam detuning + local detuning + doppler detuning 
            detuning = resonant_beam.detuning_ramp.get_value(time) + resonant_beam.calculate_doppler_detuning(momenta / cloud.particle_mass) + local_detunings
            fraction = s / (1 + combined_saturation + 4 * (detuning**2) / (resonant_beam.transition_linewidth**2))
            number_of_scattering_events = delta_t * resonant_beam.transition_linewidth / 2 * fraction
            scattering_momenta = scattering_momenta + resonant_beam.calculate_scattering_momenta(number_of_scattering_events)
        return scattering_momenta
    
# TODO check how polarisation is defined
class ResonantBeam():
    def __init__(self, detuning_ramp, power_ramp, beam_direction, polarisation = -1, waist = 2.5E-2):
        self.detuning_ramp = detuning_ramp
        self.power_ramp = power_ramp
        # direction is taken to be in the lab frame and needs to be a unit vector
        self.beam_direction = beam_direction
        self.polarisation = polarisation
        self.waist = waist
        
        ''' constant parameters for a given transition '''
        self.wavelength = 582.84E-9                 # m
        self.saturation_intensity = 1.3             # W / m**2
        self.transition_linewidth = 2*np.pi*190*1E3 # rad /s
        self.dipole_element_squared = self.transition_linewidth * 3 * np.pi * scipy.constants.epsilon_0 * \
                                      scipy.constants.hbar * (self.wavelength / (2 * np.pi))**3
    
    def calculate_beam_intensity(self, positions, time):
        # need to find the radial distance from the center of the beam
        projection = np.dot(positions, self.beam_direction).reshape(len(positions), 1)
        radial_distance = np.sqrt(np.sum(positions * positions, axis = 1) - np.sum(projection * projection, axis = 1))
        
        intensity = self.power_ramp.get_value(time) * (2 / np.pi) * (1 / (self.waist)**2) * \
                    np.exp(- 2 / np.power(self.waist, 2) * np.power(radial_distance, 2))
        return intensity
    
    def calculate_scattering_momenta(self, number_of_scattering_events):
        # impair momentum from photons along the beam and then from random re-emitted photons
        photon_momentum = scipy.constants.hbar * (2 * np.pi / self.wavelength)
        N_events =  number_of_scattering_events.reshape(len(number_of_scattering_events),1)
        random_phi = np.random.rand(len(N_events)) * 2 * np.pi
        random_theta = np.arccos(1 - 2 * np.random.rand(len(N_events)))
        emission_momenta = photon_momentum * np.sqrt(N_events) * \
                           np.stack((np.sin(random_theta)*np.sin(random_phi), np.sin(random_theta)*np.cos(random_phi), np.cos(random_theta)), axis=1)
        absorption_momenta = photon_momentum * N_events * self.beam_direction
        return absorption_momenta + emission_momenta
    
    def calculate_doppler_detuning(self, velocities):
        return - 2 * np.pi / (self.wavelength) * np.dot(velocities, self.beam_direction)
        
    def calculate_saturation_parameter(self, positions, B_field_direction, time):
        # projection of beam component along the field
        cos_B_k = np.dot(B_field_direction, self.beam_direction)
        beam_intensity = self.calculate_beam_intensity(positions, time)
        
        s_prefactor = 4 / (scipy.constants.c * scipy.constants.epsilon_0 * (scipy.constants.hbar * self.transition_linewidth)**2) * beam_intensity
        s_dipole = self.dipole_element_squared * ((1 - cos_B_k * self.polarisation)**2) / 4 
        return s_prefactor * s_dipole
        
        
        
#%% DIPOLAR FIELD
    
class DipoleField(Field):
    # dipolar field consists of dipole beams
    def __init__(self):
        self.dipole_beams = []
    
    def add_beam(self, beam):
        self.dipole_beams.append(beam)
    
    def calculate_force(self, positions, time, cloud):
        total_force = np.zeros(positions.shape)
        alpha = cloud.intensity_prefactor
        # TODO possibly parallelize
        for dipole_beam in self.dipole_beams:
            total_force = total_force + alpha * dipole_beam.calculate_intensity_gradient(positions, time)
        return total_force
    
    # detuning is calculated as delta = w - w0 = w - E_high + E_low
    # delE_high = 0 so we just add the change in the low level (which is negative)
    def calculate_detuning(self, positions, momenta, time, cloud):
        detuning = np.zeros(len(positions))
        prefactor = cloud.intensity_prefactor / scipy.constants.hbar
        # TODO possibly parallelize
        for dipole_beam in self.dipole_beams:
            detuning = detuning + prefactor * dipole_beam.calculate_beam_intensity(positions, time)
        return detuning
    
    def calculate_scattering_momenta(self, positions, momenta, B_field_direction, local_detunings, time, delta_t, cloud):
        scattering_momenta = np.zeros(positions.shape)
        # TODO possibly parallelize
        for dipole_beam in self.dipole_beams:
            number_of_scattering_events = delta_t * cloud.scattering_rate * dipole_beam.calculate_beam_intensity(positions, time)
            scattering_momenta = scattering_momenta + dipole_beam.calculate_scattering_momenta(number_of_scattering_events)
        return scattering_momenta
    
    # approximated based on harmonic potential expansion at the bottom of the trap
    def calculate_trapping_frequencies(self, time, cloud):
        omega_tot_matrix = np.zeros((3,3))
        for dipole_beam in self.dipole_beams:
            M_transform = dipole_beam.get_transformation_matrix()
            omega_squared = (dipole_beam.calculate_trapping_frequencies(time, cloud))**2
            omega_matrix = np.diag(omega_squared)
            omega_tot_matrix += np.linalg.multi_dot([M_transform, omega_matrix, np.transpose(M_transform)])
        eigen_val, eigen_vec = np.linalg.eig(omega_tot_matrix)
        return np.sqrt(eigen_val), eigen_vec        

class DipoleBeam():
    def __init__(self, power_ramp, waist_y_ramp, waist_z_ramp, focus_ramp):
        self.wavelength = 1030E-9
        self.power_ramp = power_ramp
        self.waist_y_ramp = waist_y_ramp
        self.waist_z_ramp = waist_z_ramp
        self.focus_ramp = focus_ramp
        ''' The angles are defined such that the beam is first rotated by theta z about z
            and then theta x about x'''
        self.theta_z = 0
        self.theta_x = 0
        
        self.cached_time = None
        self.cached_intensity = None
    
    def rotate_beam(self, theta_z, theta_x):
        self.theta_z = theta_z
        self.theta_x = theta_x
    
    def get_transformation_matrix(self):
        R_z = np.array([[np.cos(self.theta_z), -np.sin(self.theta_z), 0],[np.sin(self.theta_z), np.cos(self.theta_z), 0],[0,0,1]])
        R_x = np.array([[1, 0, 0],[0, np.cos(self.theta_x), -np.sin(self.theta_x)],[0, np.sin(self.theta_x), np.cos(self.theta_x)]])
        return np.matmul(R_x, R_z)
    
    def convert_global_vectors_to_local(self, vectors):
        # should be R_transp * vectors in the shape 3,N which is the same as vectors in the shape N,3 times R (not transposed) 
        return np.matmul(vectors, self.get_transformation_matrix())
    
    def convert_local_vectors_to_global(self, vectors):
        M_transform_transp = np.transpose(self.get_transformation_matrix())
        # inverse to the above
        return np.matmul(vectors, M_transform_transp)
    
    ## decide whether the input should be time or waist?
    # TODO cache?
    def calculate_waist(self, waist_0, z_rayleigh, x):
        return waist_0 * np.sqrt(1 + (x / z_rayleigh)**2)
    
    def calculate_rayleigh_range(self, waist_0):
        return np.pi * (waist_0**2) / self.wavelength
    
    def calculate_beam_intensity(self, positions, time):        
        if time == self.cached_time:
            return self.cached_intensity
        else:
            self.cached_time = time
            local_positions = self.convert_global_vectors_to_local(positions)
            waists = np.array([self.waist_y_ramp.get_value(time), self.waist_z_ramp.get_value(time)])
            rayleigh_range_yz = self.calculate_rayleigh_range(waists)
            
            # calculate the waists along the beam using the local x coordinate
            waists_yz = self.calculate_waist(waists, rayleigh_range_yz, np.stack((local_positions[:,0],local_positions[:,0]), axis=1))
            
            self.cached_intensity = self.power_ramp.get_value(time) * (2 / np.pi) * (1 / np.prod(waists_yz, axis=1)) * \
                                    np.exp(-2 * np.sum(np.power(local_positions[:,1:] / waists_yz, 2), axis=1))
            return self.cached_intensity
    
    # the negative of the intensity gradient such that force is then F = alpha * (-grad I) since potential is U = alpha * I 
    # here alpha is the intensity prefactor equal to polarizability / (2 * c * epsilon0)
    def calculate_intensity_gradient(self, positions, time):
        local_positions = self.convert_global_vectors_to_local(positions)        
        waists = np.array([self.waist_y_ramp.get_value(time), self.waist_z_ramp.get_value(time)])
        
        rayleigh_range_yz = self.calculate_rayleigh_range(waists)
        waists_yz = self.calculate_waist(waists, rayleigh_range_yz, np.stack((local_positions[:,0],local_positions[:,0]), axis=1))
        
        intensity = self.calculate_beam_intensity(positions, time)
        
        gradient_y = 4 * local_positions[:,1] * np.power(waists_yz[:,0], -2) * intensity
        
        gradient_z = 4 * local_positions[:,2] * np.power(waists_yz[:,1], -2) * intensity
        
        gradient_x =  local_positions[:,0] * intensity * \
                      np.sum(np.power(waists / (waists_yz * rayleigh_range_yz), 2) * (1 - np.power(2 * local_positions[:,1:] / waists_yz, 2)), axis=1)
        
        # gradient vector in local coordinates
        gradient_local = np.stack((gradient_x, gradient_y, gradient_z), axis = 1)
        
        return self.convert_local_vectors_to_global(gradient_local)
        
    
    def calculate_scattering_momenta(self, number_of_scattering_events):
        # impair momentum from photons along the beam and then from random re-emitted photons
        photon_momentum = scipy.constants.hbar * (2 * np.pi / self.wavelength)
        N_events =  number_of_scattering_events.reshape(len(number_of_scattering_events),1)
        random_phi = np.random.rand(len(N_events)) * 2 * np.pi
        random_theta = np.arccos(1 - 2 * np.random.rand(len(N_events)))
        emission_momenta = photon_momentum * np.sqrt(N_events) * \
                           np.stack((np.sin(random_theta)*np.sin(random_phi), np.sin(random_theta)*np.cos(random_phi), np.cos(random_theta)), axis=1)
        # absorption happens in the positive x direction along the beam
        absorption_direction = self.convert_local_vectors_to_global(np.array([1,0,0])) 
        absorption_momenta = photon_momentum * N_events * absorption_direction
        return absorption_momenta + emission_momenta
    
    def calculate_trapping_frequencies(self, time, cloud):
        power = self.power_ramp.get_value(time)
        waists = np.array([self.waist_y_ramp.get_value(time), self.waist_z_ramp.get_value(time)])
        
        omega_x_squared = -(2 * cloud.intensity_prefactor * power) / \
                           (np.pi * cloud.particle_mass * waists[0] * waists[1]) * \
                           (np.power(self.calculate_rayleigh_range(waists[0]),-2) + \
                            np.power(self.calculate_rayleigh_range(waists[1]),-2))
                   
        omega_y_squared = -(8 * cloud.intensity_prefactor * power) / \
                           (np.pi * np.power(waists[0], 3) * waists[1] * cloud.particle_mass)
                           
        omega_z_squared = -(8 * cloud.intensity_prefactor * power) / \
                           (np.pi * np.power(waists[1], 3) * waists[0] * cloud.particle_mass)
    
        return np.sqrt(np.array([omega_x_squared, omega_y_squared, omega_z_squared]))
    

#%% MAGNETIC FIELD
class UniformMagneticField:
    def __init__(self, Bx_ramp = ramps.Ramp(), By_ramp=ramps.Ramp(), Bz_ramp=ramps.Ramp()):
        self.Bx_ramp = Bx_ramp
        self.By_ramp = By_ramp
        self.Bz_ramp = Bz_ramp
        
    def calculate_field(self, positions, time):
        Bx_value = self.Bx_ramp.get_value(time)
        By_value = self.By_ramp.get_value(time)
        Bz_value = self.Bz_ramp.get_value(time)
        
        field = np.array([[Bx_value, By_value, Bz_value]])
        
        return np.repeat(field, len(positions), axis = 0)
    
class GradientMagneticField:
    def __init__(self, gradient_ramp = ramps.Ramp()):
        # needs to be in Tesla per meter!
        self.gradient_ramp = gradient_ramp
    
    def calculate_field(self, positions, time):
        gradient_value = self.gradient_ramp.get_value(time)
        grad = np.array([-gradient_value / 2, -gradient_value / 2, gradient_value])
        return grad * positions

# note that we are in the high field seeking state since mu > 0 and so U = - mu * mag(B) is minimized for maximal field
class MagneticField(Field):
    def __init__(self, uniform_field = UniformMagneticField(), gradient_field = GradientMagneticField()):
        self.uniform_field = uniform_field
        self.gradient_field = gradient_field
        
        # chaching
        self.cache_time = None
        self.cached_field = None
    
    # detuning is delta = w - E_high + E_low
    # E = - mu * mag(B)
    def calculate_detuning(self, positions, momenta, time, cloud):
        field_magnitude = self.calculate_field_magnitude(positions, time)
        return (1 / scipy.constants.hbar) * (- cloud.get_ground_state_moment() + cloud.get_excited_state_moment()) * field_magnitude
        
    def calculate_force(self, positions, time, cloud):
        gradient_value = self.gradient_field.gradient_ramp.get_value(time)
        grad = np.array([-gradient_value / 2, -gradient_value / 2, gradient_value])
        field_direction = self.calculate_field_direction(positions, time)
        return cloud.get_ground_state_moment() * grad * field_direction
        
    def calculate_scattering_momenta(self, positions, momenta, B_field_direction, local_detunings, time, delta_t, cloud):
        return np.zeros(momenta.shape)
    
    def calculate_field(self, positions, time):
        if time == self.cache_time:
            return self.cached_field
        else:
            self.cache_time = time    
            self.cached_field = self.uniform_field.calculate_field(positions, time) + self.gradient_field.calculate_field(positions, time)
            return self.cached_field

    def calculate_field_direction(self, positions, time):
        field = self.calculate_field(positions, time)
        field_magnitude = self.calculate_field_magnitude(positions, time)
        field_magnitude = field_magnitude.reshape(len(field), 1)
        return field / field_magnitude
    
    # TODO cache?
    def calculate_field_magnitude(self, positions, time):
        field = self.calculate_field(positions, time)
        return np.sqrt(np.sum(field * field, axis=1))
        
# TODO NOTE: tested all functions for 10k simulated atoms and no speed up with numexpr was possible