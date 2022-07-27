# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 15:17:24 2022

@author: kucer
"""
import numpy as np
import scipy.constants
from numba import jit
import ramps

# TODO - when initializing many of the objects initialize the ramps to something relevant to our experiment
# i.e. dipolar beams to the appropriate waists etc.
# TODO - add trap frequency calculation to the code
# NOTE: we are actually in high field seeking state! U = -mu B and for electrons mu = -g * bohr * mj
# so we have U = g * bohr * B_magnitude * mj and we have mj < 0 !!
# TODO - have a way of choosing the box size in the gui - one will be manual and the other one will be based on main dipole beam - just chose waists (default 3)
# TODO - add method into simulation itself to initialize box at a given number of beam waists/ rayleigh ranges

class Simulation:
    def __init__(self):
        self.atom_cloud = AtomCloud(100, 100)
        self.simulation_box = SimulationBox(Ramp(), np.ones(3))
        self.gamma_box = 1.0 # fraction of mean free path used as box size
        self.fields = []
        self.gravity = True
        
        ''' time related parameters'''
        self.delta_t = 1E-6
        self.gamma_t_oscillation = 1E-3
        self.gamma_t_collision = 0.1
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
    
    
    #### need to be super careful with the sign of detuning!
    def calculate_forces(self, particle_positions, time):
        # go through all the field objects
        # calculate the forces
        # first detunings, then forces
        # dont forget about gravityyyyy
        return
    
    def collide_particles(self):
        return
    
    # TODO add collision based dt
    def recalculate_delta_t(self):
        oscillation_dt = self.delta_t
        for field in self.fields:
            if isinstance(field, DipoleField):
                maximum_trapping_frequency = np.amax(field.calculate_trapping_frequencies())
                oscillation_dt = self.gamma_t_oscillation * 2 * np.pi / maximum_trapping_frequency
        self.delta_t = oscillation_dt
    
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
            # if collisions on then collide
            self.collide_particles()
            # maybe move the simulation box or have simulation box also follow a ramp! 
            
            # if heating is on then introduce scattering heating
            self.heat_particles()
            
            self.current_simulation_time += self.delta_t
            sampling_timer += self.delta_t
            self.recalculate_delta_t()
            
            if sampling_timer > self.sampling_delta_t:
                self.sample_trajectory(self.file_name + ".traj")
                sampling_timer = 0.0
        
        self.sample_trajectory(self.file_name + ".traj")
        self.save_final_configuration(self.file_name + ".fin")
        self.save_simulation_settings(self.file_name + ".set")
        
    ''' set of methods that calculate cloud properties given the trapping fields '''
    
    def calculate_average_density_harmonicApprox(self):
        omega_bar = 0.0
        for field in self.fields:
            if isinstance(field, DipoleField):
                trapping_frequencies = field.calculate_trapping_frequencies
                omega_bar = np.power(trapping_frequencies[0] * trapping_frequencies[1] * trapping_frequencies[2], 1/3)
        
        number_of_particles = self.atom_cloud.get_real_particle_number()
        cloud_temperature = self.atom_cloud.calculate_cloud_temperature()
        average_density = 2**(-3/2) * number_of_particles * omega_bar**3 * \
                          np.power(self.atom_cloud.particle_mass / (2 * np.pi * scipy.constants.k * cloud_temperature), 3/2)
        return average_density

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

##### SIMULATIONN BOX #####

class SimulationBox:
    def __init__(self, center_position_ramp = Ramp(), size):
        self.center_position_ramp = center_position_ramp
        # array of shape 3 storing the x, y and z box dimensions
        self.size = size


###### FIELDS ######

# TODO instead of positions and cloud separately just pass in the cloud? doesnt matter I guess
class Field:
    def __init__(self):
        pass
    
    def calculate_force(self, positions, time, cloud):
        pass
    
    def calculate_detuning(self, positions, time, cloud):
        pass
    
    def calcualte_scattering_heating(self, positions, time, delta_t, cloud):
        pass
    
    
###### RESONANT FIELD ######

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
        self.saturation_intensity = 1.3             # W / m
        self.transition_linewidth = 2*np.pi*190*1E3 # rad /s
        self.dipole_element_squared = self.transition_linewidth * 3 * np.pi * scipy.constants.epsilon_0 * scipy.constants.hbar * (self.wavelength / (2 * np.pi))**3
        
        
        
        
        
    
    
####### DIPOLAR FIELD #######
    
class DipoleField(Field):
    def __init__(self):
        self.dipole_beams = []
    
    def add_beam(self, beam):
        self.dipole_beams.append(beam)
    
    def calculate_force(self, positions, time, cloud):
        total_force = np.zeros(positions.shape)
        for dipole_beam in self.dipole_beams:
            total_force += cloud.intensity_prefactor * dipole_beam.calculate_intensity_gradient(positions, time)
        return total_force
    
    # assume top level doesnt shift!
    # need to be careful with the sign!
    def calculate_detuning(self, positions, time, cloud):
        detuning = np.zeros(len(positions))
        for dipole_beam in self.dipole_beams:
            detuning -= cloud.intensity_prefactor * dipole_beam.calculate_beam_intensity(positions, time)
        return detuning
    
    def calcualte_scattering_heating(self, positions, time, delta_t, cloud):
        heating_momenta = np.zeros(positions.shape)
        for dipole_beam in self.dipole_beams:
            number_of_scattering_events = delta_t * cloud.scattering_rate * dipole_beam.calculate_beam_intensity(positions, time)
            heating_momenta += dipole_beam.calculate_scattering_heating(number_of_scattering_events)
        return heating_momenta
    
    # experimental feature, needs a bit more thinking. Can do either analytical and assume harmonic potential
    # or can do some kind of hessian calculation
    def calculate_trapping_frequencies():
        omega_squared_total = np.zeros(3)
        for dipole_beam in self.dipole_beams:
            M_transform = dipole_beam.get_transformation_matrix()
            omega_squared_beam = dipole_beam.calculate_trapping_frequencies()**2
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
        local_positions = self.convert_global_vectors_to_local(positions)
        rayleigh_range_y = self.calculate_rayleigh_range(self.waist_y_ramp.get_value(time))
        rayleigh_range_z = self.calculate_rayleigh_range(self.waist_z_ramp.get_value(time))
        
        # calculate the waists along the beam using the local x coordinate
        waists_y = self.calculate_waist(self.waist_y_ramp.get_value(time), rayleigh_range_y, local_positions[:,0])
        waists_z = self.calculate_waist(self.waist_z_ramp.get_value(time), rayleigh_range_z, local_positions[:,0])
        
        return self.power_ramp.get_value(time) * (2 / np.pi) * (1 / (waists_y * waists_z)) * \
               np.exp(- 2 * np.power(local_positions[:,1], 2) / np.power(waists_y, 2)) * \
               np.exp(- 2 * np.power(local_positions[:,2], 2) / np.power(waists_z, 2))
    
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
                   
        omega_y_squared = -(8 * self.intensity_prefactor * self.power_ramp.get_value(time)) / \
                           (np.pi * np.power(self.waist_y_ramp.get_value(time), 3) * self.waist_z_ramp.get_value(time) * cloud.particle_mass)
                           
        omega_z_squared = -(8 * self.intensity_prefactor * self.power_ramp.get_value(time)) / \
                           (np.pi * np.power(self.waist_z_ramp.get_value(time), 3) * self.waist_y_ramp.get_value(time) * cloud.particle_mass)
    
        return np.sqrt(np.array([omega_x_squared, omega_y_squared, omega_z_squared]))

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
        return np.zeros(positions.shape)
    
class UniformMagneticField:
    # initialize with empty ramps
    # empty ramp returns 0 when asked for value
    def __init__(self, x_ramp = Ramp(), y_ramp = Ramp(), z_ramp = Ramp()):
        self.x_ramp = x_ramp
        self.y_ramp = y_ramp
        self.z_ramp = z_ramp
        
    def calculate_field(self, positions, time):
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
        self.alpha = np.ones(number_of_simulated_particles) * (number_of_real_particles / number_of_simulated_particles)
        
        self.particle_mass = 167.9323702 * scipy.constants.physical_constants['atomic mass unit-kilogram relationship'][0]
        
        self.lande_g_ground = 1.163801
        self.mj_ground = -6
        self.lande_g_excited = 1.195
        self.mj_excited = -7
        self.magneton = scipy.constants.physical_constants['Bohr magneton'][0]
        
        self.scatt_cross_section = 8 * np.pi * (150 * scipy.constants.physical_constants['atomic unit of length'][0])**2  # need to check this!!
        self.polarizability = 2.9228099E-39
        self.intensity_prefactor = - self.polarizability / (2 * scipy.constants.epsilon_0 * scipy.constants.c)
        self.scattering_rate = 1.79341521457477E-10   # in units of s-1 per W/m2 i.e. multiply by intensity to get the real scattering rate
    
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
    
    def get_real_particle_number(self):
        return np.sum(self.alpha)
    
    def get_Zeeman_shift_prefactor(self):
        return self.magneton * (self.mj_excited * self.lande_g_excited - self.mj_ground * self.lande_g_ground)
    
    def get_ground_state_moment(self):
        return - self.magneton * self.lande_g_ground * self.mj_ground
        
