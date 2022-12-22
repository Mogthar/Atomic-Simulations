# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 15:17:24 2022

@author: kucer
"""
import numpy as np
import scipy.constants
from numba import jit
from ramps import Ramp, RampSegment

# TODO - check detuning signs!
# TODO - chache data in the field objects. Before each force calculation call a reset function that sets everything to 0. maybe have another variable called cahe time and then in the call for
# the function ramp we can first check whether time is the cache time, if not then recalculate and cache! For that just define a cache method that does that.

# TODO - when initializing many of the objects initialize the ramps to something relevant to our experiment
# i.e. dipolar beams to the appropriate waists etc.
# NOTE: we are actually in high field seeking state! U = -mu B and for electrons mu = -g * bohr * mj
# so we have U = g * bohr * B_magnitude * mj and we have mj < 0 !!
# TODO - have a way of choosing the box size in the gui - one will be manual and the other one will be based on main dipole beam - just chose waists (default 3)
# TODO - add method into simulation itself to initialize box at a given number of beam waists/ rayleigh ranges

##### PERFORMANCE NOTES
# NOTE is better with numpy arrays to use array = array + another array

class Simulation:
    def __init__(self):
        self.atom_cloud = AtomCloud(100, 100)
        self.simulation_box = SimulationBox(Ramp(), np.ones(3))
        self.gamma_box = 1.0 # fraction of mean free path used as box sizes
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

'''
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
'''

#%% SIMULATIONN BOX #####

class SimulationBox:
    ''' center position stores a vector pointing to the box center, 
    size stores a vector of the form (length, width, height)'''
    
    def __init__(self, center_position, size):
        self.center = center_position
        self.size = size
        
    def set_box_size(self, x_size, y_size, z_size):
        self.size = np.array([x_size, y_size, z_size])
    
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
    #TODO check what potential is passed in here. most likely dipolar field
    def thermalize_positions(self, temperature, dipole_field, time):
        omega = dipole_field.calculate_trapping_frequencies(time, self)
        self.positions = np.random.normal(loc = 0.0, 
                                          scale = np.sqrt(scipy.constants.k * temperature / (self.particle_mass * (omega**2))), 
                                          size = (len(self.positions), 3))
        
    #TODO check the definition of boudning box
    def calculate_cloud_temperature(self, bounding_box):
        # bounding box defined by center and size 
        culling_mask = np.logical_not(np.any((self.positions > (bounding_box.center + bounding_box.size / 2)) | 
                                            (self.positions < (bounding_box.center - bounding_box.size / 2))))
        
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
        
