import numpy as np
import matplotlib.pyplot as plt
import scipy.constants
import random
import time
import os
from numba import jit

### Laser loading parameters and geometry ###
P_loading = 12   # loading power
wy_1 = 21.0E-6   # beam waist
wz_1 = 21.0E-6 

### misc parameters ###
particle_mass =  167.9323702 * scipy.constants.physical_constants['atomic mass unit-kilogram relationship'][0]
sLen = 72 * scipy.constants.physical_constants['atomic unit of length'][0]   # sWave scattering length
dipLen = 0.95 * sLen
c_section = 32*np.pi/45 * np.power(dipLen,2) + 8*np.pi*np.power(sLen,2) 
alpha = -5.50555551377629e-37    # polarizability prefactor
wavelength = 1030E-9             # laser wavelength

### set of functions to calculate trapping geometry and harmonic frequencies
@jit(nopython=True, parallel=True)
def calc_waist(w0, zr, x):
    return w0 * np.sqrt(1+np.power(x / zr, 2))

@jit(nopython=True, parallel=True)
def calc_RayleighLength(waist, wavelength):
    return np.pi * (waist**2) / wavelength

@jit(nopython=True, parallel=True)
def calc_OmegaSquared_1_x(P1, waist_y1, waist_z1):
    return -(2*alpha*P1) / (np.pi * particle_mass * waist_y1 * waist_z1) * (np.power(calc_RayleighLength(waist_y1, wavelength),-2) + np.power(calc_RayleighLength(waist_z1, wavelength),-2))

@jit(nopython=True, parallel=True)
def calc_OmegaSquared_1_y(P1, waist_y1, waist_z1):
    return -(8*alpha*P1) / (np.pi * np.power(waist_y1, 3) * waist_z1 * particle_mass)

@jit(nopython=True, parallel=True)
def calc_OmegaSquared_1_z(P1, waist_y1, waist_z1):
    return -(8*alpha*P1) / (np.pi * np.power(waist_z1, 3) * waist_y1 * particle_mass)

# define a box around the focus 
n_waists = 3
n_rayleigh = 2  
initial_box_size = np.array([2 * n_rayleigh * calc_RayleighLength(wy_1, wavelength), 2 * wy_1 * n_waists, 2 * wz_1 * n_waists])
initial_box_volume = initial_box_size[0] * initial_box_size[1] * initial_box_size[2]

# calculate dt based on oscillation period; later compare with average collision time and chose the smaller of the two
omega_max = np.sqrt(max(calc_OmegaSquared_1_x(P_loading, wy_1, wz_1), calc_OmegaSquared_1_y(P_loading, wy_1, wz_1), calc_OmegaSquared_1_z(P_loading, wy_1, wz_1)))
gamma_osc = 0.01   # fraction of oscillation period used to calculate dt
dt_osc_init = gamma_osc * 2 * np.pi / omega_max

# calculate dt based on mean collision time
N_loading = 3.0E5      # typical number of loaded particles
T_loading = 25.0E-6    # typical loading temperature
omega_bar = np.power(calc_OmegaSquared_1_x(P_loading, wy_1, wz_1) * calc_OmegaSquared_1_y(P_loading, wy_1, wz_1) * calc_OmegaSquared_1_z(P_loading, wy_1, wz_1), 1/6)
avg_dens = np.power(2, -3/2) * N_loading * np.power(particle_mass / (2 * np.pi * scipy.constants.k * T_loading), 3/2) * omega_bar**3  
mean_free_path = 1 / (avg_dens * c_section)
avg_rel_speed = np.sqrt(16 * scipy.constants.k * T_loading / (np.pi * particle_mass))
mean_collision_time = mean_free_path / avg_rel_speed
gamma_coll = 0.1       # fraction of mean collision time used to calculate dt
dt_coll_init = gamma_coll * mean_collision_time

print("dt based on oscillations: ", dt_osc_init, "dt based on collisions: ", dt_coll_init)

# calculate simulation cell dimensions - used for treating collisions
gamma_cell = 0.1   # fraction of mean free path used as cell size
approx_cell_size = mean_free_path * gamma_cell
initial_cell_numbers = np.ceil(initial_box_size / approx_cell_size).astype('int64')   # partition simulation box into cells
print("Cell numbers in each direction:", initial_cell_numbers)
cell_sizes = initial_box_size / initial_cell_numbers
cell_volume = cell_sizes[0] * cell_sizes[1] * cell_sizes[2]
N_cells = initial_cell_numbers[0] * initial_cell_numbers[1] * initial_cell_numbers[2]
print("Total number of cells:", N_cells)


### functions for use in simulations ###

# force from a laser beam at particular set of positions
@jit(nopython=True, parallel=True)
def calc_force1(position, waist_y, waist_z, P1, focal_position):
    x = position[:,0] - focal_position
    y = position[:,1]
    z = position[:,2]
    rayleigh_y = calc_RayleighLength(waist_y, wavelength)
    rayleigh_z = calc_RayleighLength(waist_z, wavelength)
    wy = calc_waist(waist_y, rayleigh_y, x)
    wz = calc_waist(waist_z, rayleigh_z, x)
    
    Fy = (8 * alpha * P1 / np.pi) * y * (1 / wz) * np.power(wy, -3) * np.exp(- 2 * np.power(y, 2) / np.power(wy, 2)) * np.exp(- 2 * np.power(z, 2) / np.power(wz, 2))
    Fz = (8 * alpha * P1 / np.pi) * z * (1 / wy) * np.power(wz, -3) * np.exp(- 2 * np.power(y, 2) / np.power(wy, 2)) * np.exp(- 2 * np.power(z, 2) / np.power(wz, 2))
    Fx_prefactor = (2 * alpha * P1 / np.pi) * x * (1 / (wz * wy)) * np.exp(- 2 * np.power(y, 2) / np.power(wy, 2)) * np.exp(- 2 * np.power(z, 2) / np.power(wz, 2))
    Fx_postfactor = np.power(waist_y / (wy * rayleigh_y), 2) * (-np.power(2 * y / wy, 2) + 1) + np.power(waist_z / (wz * rayleigh_z), 2) * (-np.power(2 * z / wz, 2) + 1)
    Fx = Fx_prefactor * Fx_postfactor
    return np.stack((Fx,Fy,Fz), axis=1)

# function that adds different force contributions; currently only dipole force + gravity; can add light scattering in the future
@jit(nopython=True, parallel=True)
def Forces(positions, P1, waist_y1, waist_z1, gravity, focal_position):
    force_odt1 = calc_force1(positions, waist_y1, waist_z1, P1, focal_position)
    if (gravity):
        gravitational_force = np.zeros(positions.shape)
        gravitational_force[:,2] = - particle_mass * scipy.constants.g
        return force_odt1 + gravitational_force
    else:
        return force_odt1
 
# verlet algorithm for time propagation of particles 
@jit(nopython=True, parallel=True)
def timeEvolutionVerlet(positions, momenta, P1, waist_y1, waist_z1, next_waist_y1, next_waist_z1, gravity, delta_t, current_foc_pos, next_foc_pos):
    old_forces = Forces(positions, P1, waist_y1, waist_z1, gravity, current_foc_pos)
    
    new_positions = positions + momenta * delta_t / particle_mass + 1/2 * old_forces / particle_mass * delta_t**2
    
    new_forces = Forces(new_positions, P1, next_waist_y1, next_waist_z1, gravity, next_foc_pos)
    new_momenta = momenta + 1/2 * (old_forces + new_forces) * delta_t
    return new_positions, new_momenta    

# removes particles that fly out of the simulation box    
def boxCheck(positions, momenta, box_size, focal_position):
    focal_position_vector = np.array([focal_position,0.0,0.0])
    stayed_in_box = np.logical_not(np.any(((positions-focal_position_vector) > box_size / 2) | ((positions-focal_position_vector) < -box_size / 2), axis = 1))
    return positions[stayed_in_box], momenta[stayed_in_box]

# returns a list of IDs where each entry corresponds to the ID of a cell that a given particle currently resides in
# this helps with dealing with collisions only within individual cells
@jit(nopython=True, parallel=True)
def AssignParticleToACell(positions, box_size, cell_numbers, focal_pos):
    focal_position_vector = np.array([focal_pos,0.0,0.0])
    indices = np.floor(((positions-focal_position_vector) + box_size/2) / (box_size / cell_numbers))
    is_out_of_box = np.full(len(indices), False)
    for i in range(len(is_out_of_box)):
        is_out_of_box[i] = np.any((indices[i] > (cell_numbers -1)) | (indices[i] < 0))
    cell_id = indices[:,2] * cell_numbers[0] * cell_numbers[1] + indices[:,1] * cell_numbers[1] + indices[:,0]
    cell_id[is_out_of_box] = -1
    return cell_id

# an implementation of DSMC algorithm for coliding particles
# needs paralelizing
def dictionaryCollisions(positions, momenta, delta_t, box_size, cell_numbers, cell_vol, alpha_N, focal_pos):
    indices = AssignParticleToACell(positions, box_size, cell_numbers, focal_pos)
    # combine momenta and positions into one array
    combined_pos_mom_array = np.concatenate((positions,momenta),axis=1)
    N_particles = len(positions)
    average_density = 0
    N_particles_trapped = 0        # stores number of particles that are still in the simulation box
    new_momenta = np.array([])     # array to store new positions and momenta after the collision step is evaluated
    new_positions = np.array([])
    
    # create cell dictionary where each key is a populated cell ID and it contains an array of positions and momenta of particles in the given cell
    # particles outside the box have cell ID = -1
    cell_dictionary = {}
    for i in range(len(indices)):
        if indices[i] in cell_dictionary:
            cell_dictionary[indices[i]] = np.append(cell_dictionary[indices[i]], 
                                                    combined_pos_mom_array[i].reshape(1,6), axis=0)
        else:
            cell_dictionary[indices[i]] = combined_pos_mom_array[i].reshape(1,6)
    
    # perform random collisions within each cell
    for cell_index in cell_dictionary:
        momenta_in_cell = cell_dictionary[cell_index][:,3:6]
        positions_in_cell = cell_dictionary[cell_index][:,0:3]
        N_part_cell = len(positions_in_cell)
        if cell_index != -1:
            average_density += (N_part_cell*alpha_N)**2
            N_particles_trapped += N_part_cell*alpha_N
            if N_part_cell > 1:
                particle_labels = np.linspace(0, N_part_cell, N_part_cell, endpoint=False, dtype=int)
                # maximum relative velocity of particles in a cell
                max_v_rel = 2 * np.amax(np.linalg.norm(momenta_in_cell / particle_mass, axis=1))

                # calculate the number of test collisions
                N_pairs = N_part_cell * (N_part_cell - 1) / 2
                N_test_collisions = np.ceil(N_pairs * alpha_N * delta_t / cell_volume * c_section * max_v_rel).astype('int')
                
                # for each test collision pick a pair of particles, test whether they collide and then change momenta based on ellastic collision dynamics
                for i in range(N_test_collisions):
                    pair_choice = np.random.choice(particle_labels, 2)
                    pair_momenta = momenta_in_cell[pair_choice]
                    relative_speed = np.linalg.norm((pair_momenta[0] - pair_momenta[1]) / particle_mass)
                    ## test the collision ##
                    collision_probability = N_pairs * (alpha_N * delta_t * c_section * relative_speed / cell_volume) / N_test_collisions
                    if  np.random.rand() < collision_probability:
                        velocity_com = (pair_momenta[0] + pair_momenta[1]) / (2 * particle_mass)
                        velocity_relative = (pair_momenta[0] - pair_momenta[1]) / particle_mass
                        rand_theta = np.arccos(2 * np.random.rand() - 1)
                        rand_phi = np.random.rand() * 2 * np.pi
                        c = relative_speed * np.array([np.sin(rand_theta) * np.cos(rand_phi), np.sin(rand_theta) * np.sin(rand_phi), np.cos(rand_theta)])
                        # calculate new momenta
                        pair_momenta[0] = particle_mass * (velocity_com + 0.5 * c)
                        pair_momenta[1] = particle_mass * (velocity_com - 0.5 * c)
                        # update momenta in cell with new momenta
                        momenta_in_cell[pair_choice] = pair_momenta

        # after a cell has been evaluated append the momenta and positions to the new momentum/position array
        new_momenta = np.append(new_momenta, momenta_in_cell)
        new_positions = np.append(new_positions, positions_in_cell)
    
    new_momenta = np.reshape(new_momenta, (N_particles, 3))
    new_positions = np.reshape(new_positions, (N_particles, 3))
    average_density *= (1 / (N_particles_trapped * cell_vol))
    return new_positions, new_momenta, average_density


### different types of transport ramps  ###
### return focal position at an arbitrary time given the total ramp time and total ramp distance ###

@jit(nopython=True, parallel=True)
def LinearRamp(ramp_distance, ramp_time, time):
    if time <= 0:
        return 0
    elif time > 0 and time < ramp_time:
        return ramp_distance * time / ramp_time
    else:
        return ramp_distance
    
@jit(nopython=True, parallel=True)
def ConstAccRamp(ramp_distance, ramp_time, time):
    if time <= 0:
        return 0
    elif time > 0 and time <= (ramp_time / 2):
        return (1/2) * (4 * ramp_distance / (ramp_time**2)) * (time**2)
    elif time > (ramp_time / 2) and time <= ramp_time:
        return -(1/2) * (4 * ramp_distance / (ramp_time**2)) * ((time-ramp_time)**2) + ramp_distance
    else:
        return ramp_distance

@jit(nopython=True, parallel=True)
def CubicRamp(ramp_distance, ramp_time, time):
    if time <= 0:
        return 0
    elif time > 0 and time <= ramp_time:
        return -2*ramp_distance * (time / ramp_time)**3 + 3*ramp_distance * (time / ramp_time)**2
    else:
        return ramp_distance

@jit(nopython=True, parallel=True)
def ConstJerkRamp(ramp_distance, ramp_time, time):
    if time <= 0:
        return 0
    elif time > 0 and time <= (ramp_time / 4):
        return 16/3 * ramp_distance * (time / ramp_time)**3
    elif time > (ramp_time / 4) and time <= (3* ramp_time / 4):
        return -ramp_distance*16/3*(time/ramp_time)**3 + 8*ramp_distance*(time/ramp_time)**2 - 2*ramp_distance*(time/ramp_time)+ramp_distance/6
    elif time > (3* ramp_time / 4) and time <= ramp_time:
        return 16/3 * ramp_distance * ((time - ramp_time) / ramp_time)**3 + ramp_distance
    else:
        return ramp_distance

#### #### #### #### #### ####
        
### functions to deal with logging and loading data ###
        
# log particle positions in an XYZ formatted file
def WriteXYZFile(filename, positions, defaultLen):
    path = "./Trajectories/" + filename
    with open(path, "a") as f:
        # particle_number_string = str(len(positions))
        # f.write(particle_number_string)
        f.write(str(defaultLen))
        f.write('\n')
        f.write("frame")
        f.write('\n')
        for i in range(len(positions)):
            atom_name = "atom%d " %(i)
            position_string = "%.9f %.9f %.9f" %(positions[i][0],positions[i][1],positions[i][2])
            f.write(atom_name + position_string)
            f.write('\n')
        for j in range(defaultLen - len(positions)):
            atom_name = "atom%d " %(j + len(positions))
            position_string = "%.9f %.9f %.9f" %(0.0,0.0,0.0)
            f.write(atom_name + position_string)
            f.write('\n')
    
# log particle number and temperature at a given time
def LogTempAndPartNum(filename, positions, momenta, time, alph):
    path = "./Trajectories/" + filename
    with open(path, "a") as f:
        # particle_number_string = str(len(positions))
        # f.write(particle_number_string)
        particle_number = len(positions) * alph
        mom_magnitude = np.linalg.norm(momenta, axis=1)
        temp = np.average(mom_magnitude**2) / (3 * particle_mass * scipy.constants.k)
        string = "%.5f %.5f %.8f" %(time, particle_number, temp)
        f.write(string)
        f.write('\n')

# load positions from an XYZ file
def LoadAtomPosition(filename):
    path = "./Trajectories/" + filename
    with open(path,"r") as f:
        atom_number = int(f.readline())
        atomic_coordinates = np.zeros((atom_number,3))
        lines = f.readlines()[-(atom_number):-1]
    
    for index, line in enumerate(lines):
        coordinates = [float(x) for x in line.split()[1:4]]
        atomic_coordinates[index] = coordinates
    return atomic_coordinates[np.logical_not(np.all(atomic_coordinates==0, axis=1))]

# do a final log of both positions and momenta which can be used as a starting point for next simulation
def FinalLog(filename, positions, momenta, alph):
    path = "./Trajectories/" + filename
    with open(path, "a") as f:
        particle_number = len(positions)
        mom_magnitude = np.linalg.norm(momenta, axis=1)
        temp = np.average(mom_magnitude**2) / (3 * particle_mass * scipy.constants.k)
        string = "%.6f %.6f %.8f" %(particle_number, alph, temp)
        f.write(string)
        f.write('\n')
        for i in range(len(positions)):
            coordinate_string = "%.9f %.9f %.9f %.9f %.9f %.9f" %(positions[i][0],positions[i][1],positions[i][2], 
                                                                        momenta[i][0], momenta[i][1], momenta[i][2])
            f.write(coordinate_string)
            f.write('\n')

# load positions and momenta from a final log file
def LoadFromFinalLog(filename):
    path = "./Trajectories/" + filename
    with open(path, "r") as f:
        first_line = f.readline()
        atom_number = int(float(first_line.split()[0]))
        alph = float(first_line.split()[1])
        atomic_coordinates = np.zeros((atom_number,3))
        atomic_momenta = np.zeros((atom_number,3))
        lines = f.readlines()[1:-1]
    
    for index, line in enumerate(lines):
        coordinates = [float(x) for x in line.split()[0:3]]
        momenta = [float(x) for x in line.split()[3:6]]
        atomic_coordinates[index] = coordinates
        atomic_momenta[index] = momenta
    return atomic_coordinates, atomic_momenta, alph

### ### ### ### ### ### ### ### ###
 
    



### HERE BEGINS THE ACTUAL SIMULATION BIT ###
transport_times = np.linspace(2,5,11)   # simulate specific transport ramp with varying length in time
transport_distance = 0.1

# arrays to store final temperatures and transport efficiency
temperatures = np.zeros(len(transport_times))
initial_temperatures = np.zeros(len(transport_times))
efficiency = np.zeros(len(transport_times))

N_particles = 10000                     # number of simulated particles
alpha_N = N_loading / N_particles       # number of particles each simulated particle represents
# current_T = T_loading
P_loading = 12.0                        # power in the beam

# trap frequencies squared to initialize particles
omega_squared = np.array([calc_OmegaSquared_1_x(P_loading, wy_1, wz_1), 
                          calc_OmegaSquared_1_y(P_loading, wy_1, wz_1), 
                          calc_OmegaSquared_1_z(P_loading, wy_1, wz_1)])

for i, t_trans in enumerate(transport_times):
    
    # dt is the simulation time step and is defined as min(dt_osc, dt_coll)
    dt = dt_osc_init
    dt_osc = dt_osc_init
    dt_coll = dt_coll_init
    
    # initialize particles based on maxwell boltzmann distribution in a harmonic trap
    positions = np.random.normal(loc = 0.0, 
                                 scale = np.sqrt(scipy.constants.k * T_loading / (particle_mass * omega_squared)), 
                                 size = (N_particles, 3))

    momenta = particle_mass  * np.random.normal(loc = 0.0, 
                                            scale = np.sqrt(scipy.constants.k * T_loading / particle_mass), 
                                            size = (N_particles, 3))

    # logging filenames
    filename = "tevz_cubic_harmonic_comparison_long.xyz"
    filename_log = "log_" + filename

    # total collision time and a timer that keeps track of time since last collision timestep
    total_time = 0.0
    collision_timer = 0.0
    # set up a sampling timer and an interval which determines how often we log the positions of particles
    sampling_timer = 0.0
    sampling_interval = 1E-3
    WriteXYZFile(filename, positions, N_particles)   # log initial positions
    
    # initially equilibrate for 20 oscillation times because the trap is actually not harmonic but we initialize it as such
    while total_time < dt_osc_init * 20:
        positions, momenta = timeEvolutionVerlet(positions, momenta, P_loading, wy_1, wz_1, wy_1, wz_1, 
                                                     True, dt, 0.0, 0.0)
        total_time += dt
    
    # calculate the equilibrated temperature of particles left in the trap
    dummy_positions, dummy_momenta = boxCheck(positions, momenta, initial_box_size, 0.0)
    mom_magnitude = np.linalg.norm(dummy_momenta, axis=1)
    init_temp = np.average(mom_magnitude**2) / (3 * particle_mass * scipy.constants.k)
    initial_temperatures[i] = init_temp
    
    # begin transport
    total_time = 0.0
    while total_time < t_trans:
        #### 1. propagate ###
        # first calculate the focal positions and the waists (keep the waist constant for now)        
        focal_position = CubicRamp(transport_distance, t_trans, total_time)
        next_focal_position = CubicRamp(transport_distance, t_trans, total_time + dt)
        wy, wz = wy_1, wz_1
        wy_next, wz_next = wy_1, wz_1
        
        positions, momenta = timeEvolutionVerlet(positions, momenta, P_loading, wy, wz, wy_next, wz_next, 
                                                 True, dt, focal_position, next_focal_position)

        #### 2. collide particles ####
        collision_timer += dt
        # if time from last colision is larger than the collision timestep proceed with collisions
        if collision_timer >= dt_coll:
            positions, momenta, avg_dens = dictionaryCollisions(positions, momenta, collision_timer, initial_box_size, 
                                                         initial_cell_numbers, cell_volume, alpha_N, focal_position)
            collision_timer -= dt_coll
        
            # recalculate the collision time
            dummy_positions, dummy_momenta = boxCheck(positions, momenta, initial_box_size, next_focal_position)
            mom_magnitude = np.linalg.norm(dummy_momenta, axis=1)
            current_T = np.average(mom_magnitude**2) / (3 * particle_mass * scipy.constants.k)
        
            mean_free_path = 1 / (avg_dens * c_section)
            avg_rel_speed = np.sqrt(16 * scipy.constants.k * current_T / (np.pi * particle_mass))
            mean_collision_time = mean_free_path / avg_rel_speed
            dt_coll = gamma_coll * mean_collision_time
            
        
        #### 3. increment total time and recalculate dt if need be ####
        total_time += dt 
        
        # recalculate dt_osc (note that this only changes if either total power or the beam waist changes)
        omega_max = np.sqrt(max(calc_OmegaSquared_1_x(P_loading, wy_next, wz_next), calc_OmegaSquared_1_y(P_loading, wy_next, wz_next), calc_OmegaSquared_1_z(P_loading, wy_next, wz_next)))
        dt_osc = gamma_osc * 2 * np.pi / omega_max
        
        if dt_osc > dt_coll:
            print("collision time step is dominating. Watch out")
        dt = dt_osc 
        
        #### 4. sample if sampling timer exceeds chosen value
        sampling_timer += dt
        if sampling_timer > sampling_interval:
            WriteXYZFile(filename, positions, N_particles)
            LogTempAndPartNum(filename_log, positions, momenta, total_time, alpha)
            sampling_timer = 0.0
    
    
    # equilibrate for 200 ms at the end    
    total_time = 0.0
    while total_time < 0.2:
        positions, momenta = timeEvolutionVerlet(positions, momenta, P_loading, wy_1, wz_1, wy_1, wz_1, 
                                                     True, dt, transport_distance, transport_distance)
        total_time += dt
    
    # finally check how many particles are left in the trap and calculate their temperature
    positions, momenta = boxCheck(positions, momenta, initial_box_size, transport_distance)
    temp  = 0.0
    if len(positions) > 0:
        mom_magnitude = np.linalg.norm(momenta, axis=1)
        temp = np.average(mom_magnitude**2) / (3 * particle_mass * scipy.constants.k)
    temperatures[i] = temp
    
    efficiency[i] = len(positions) / N_particles
    print("done")

plt.plot(transport_times, efficiency)
plt.show()
plt.plot(transport_times, temperatures)
plt.show()    
