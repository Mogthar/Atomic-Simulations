# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 11:45:50 2022

@author: kucer
"""
import numpy as np
import scipy.constants
import scipy.optimize

def calculate_potential_depth(field, atom_cloud):
    intensity_at_focus = field.calculate_intensity(np.array([[0.0, 0.0, 0.0]]))
    return -intensity_at_focus * atom_cloud.calculate_intensity_prefactor()

def calculate_potential_energy(field, atom_cloud, positions):
    potential_energy = field.calculate_intensity(positions) * atom_cloud.calculate_intensity_prefactor()
    potential_energy += calculate_potential_depth(field, atom_cloud)
    return potential_energy

def calculate_peak_density(grid, dV, field, atom_cloud):
    xx, yy, zz = grid
    vectors = np.stack((xx, yy, zz), axis=3)
    print("shape of vectors: ", vectors.shape)
    
    partition_function = 0.0
    
    #integrate along x (break down into planes to reduce memory usage)
    for vector_plane in vectors:
        plane_shape = vector_plane.shape
        positions = vector_plane.reshape((plane_shape[0] * plane_shape[1], 3))
        potential = calculate_potential_energy(field, atom_cloud, positions)
        partition_function += np.sum(np.exp(-potential / (scipy.constants.k * atom_cloud.T))) * dV
    
    return atom_cloud.N / partition_function

'''
def calculate_beam_depths(beams, atom_cloud):
    depths= []
    for beam in beams:
        intensity_at_focus = beam.calculate_beam_intensity(np.array([[0.0, 0.0, 0.0]]))
        depths.append(-intensity_at_focus * atom_cloud.calculate_intensity_prefactor())
    return depths
    
def calculate_cloud_size(field, atom_cloud, gamma_cutoff = 10):
    beams = field.get_beams()
    total_depth = calculate_potential_depth(field, atom_cloud)
    beam_depths = calculate_beam_depths(beams, atom_cloud)
    
    if gamma_cutoff * scipy.constants.k * atom_cloud.T > total_depth:
        raise Exception("Cut off is too high!")
    
    cloud_sizes = np.zeros((len(beams), 3))
    for i, beam in enumerate(beams):
        beam_depth = beam_depths[i]
        if gamma_cutoff * scipy.constants.k * atom_cloud.T > total_depth - beam_depth:
            energy_cutoff = gamma_cutoff * scipy.constants.k * atom_cloud.T - total_depth
            # size of the thermal cloud in the beams native coordinates
            beam_cloud_size = calculate_size_along_beam(beam, atom_cloud, energy_cutoff)
            # determine the bounding box for the rotated beam
            bounding_box_size = np.max(abs(np.matmul(beam.get_transformation_matrix(), np.diag(beam_cloud_size))),axis=1)
            cloud_sizes[i] = bounding_box_size
    
    max_bounding_box = np.max(cloud_sizes, axis=0)
    
    # need to check for 0 in the bounding box entry
    if not np.all(max_bounding_box):
        print("No spilling in the wings")
        harmonic_frequencies, eigen_directions = field.calculate_trapping_frequencies(atom_cloud.calculate_intensity_prefactor(), atom_cloud.m)
        # size of cloud along eigen directions
        beam_cloud_size = np.sqrt(2 * gamma_cutoff * scipy.constants.k * atom_cloud.T / atom_cloud.m) / harmonic_frequencies
        max_bounding_box = np.max(abs(np.matmul(eigen_directions, np.diag(beam_cloud_size))), axis=1)
    return max_bounding_box
'''

def calculate_cloud_size(field, atom_cloud, gamma_cutoff = 10, epsilon=1E-8):
    beams = field.get_beams()
    total_depth = calculate_potential_depth(field, atom_cloud)
    
    if gamma_cutoff * scipy.constants.k * atom_cloud.T > total_depth:
        raise Exception("Cut off is too high!")
    
    cloud_sizes = np.zeros((len(beams), 3))
    for i, beam in enumerate(beams):
        try:
            # size of the thermal cloud in the beams native coordinates
            beam_cloud_size = calculate_size_along_beam(beam, field, atom_cloud, gamma_cutoff, epsilon)
            # transform this into the lab frame and determine maximum components along the lab axis to define a bounding box
            bounding_box_size = np.max(abs(np.matmul(beam.get_transformation_matrix(), np.diag(beam_cloud_size))),axis=1)
            print("beam", i, "bounding box is", bounding_box_size)
            cloud_sizes[i] = bounding_box_size
        except:
            print("beam", i, "did not converge")
    max_bounding_box = np.max(cloud_sizes, axis=0)
    return max_bounding_box

def generate_integration_grid(field, atom_cloud, gamma_cutoff = 10, dl = 1E-6):
    box_size = calculate_cloud_size(field, atom_cloud, gamma_cutoff)
    x = np.arange(-box_size[0], box_size[0], dl)
    y = np.arange(-box_size[1], box_size[1], dl)
    z = np.arange(-box_size[2], box_size[2], dl)
    
    xx, yy, zz = np.meshgrid(x, y, z, indexing = 'ij')
    
    return xx, yy, zz, dl**3
        
'''
def calculate_size_along_beam(beam, field, atom_cloud, gamma_cutoff = 10):
    # need to solve potential = energy_cutoff alond principal axes of a beam
    
    #x solution
    Ux = lambda x : calculate_potential_energy(field, atom_cloud, beam.convert_local_vectors_to_global(np.array([[x, 0.0, 0.0]], dtype=np.double))) - gamma_cutoff * scipy.constants.k * atom_cloud.T
    solution = scipy.optimize.root(Ux, [0.0])
    if not solution.success:
        raise Exception("cloud radius not found")
    x_radius = solution.x[0]
    
    #y solution
    Uy = lambda y : calculate_potential_energy(field, atom_cloud, beam.convert_local_vectors_to_global(np.array([[0.0, y, 0.0]], dtype=np.double))) - gamma_cutoff * scipy.constants.k * atom_cloud.T
    solution = scipy.optimize.root(Uy, [0.0])
    if not solution.success:
        raise Exception("cloud radius not found")
    y_radius = solution.x[0]
    
    #z solution
    Uz = lambda z: calculate_potential_energy(field, atom_cloud, beam.convert_local_vectors_to_global(np.array([[0.0, 0.0, z]], dtype=np.double))) - gamma_cutoff * scipy.constants.k * atom_cloud.T
    solution = scipy.optimize.root(Uz, [0.0])
    if not solution.success:
        raise Exception("cloud radius not found")
    z_radius = solution.x[0]
    
    return np.array([x_radius, y_radius, z_radius])
'''

def calculate_size_along_beam(beam, field, atom_cloud, gamma_cutoff = 10, epsilon=1E-8):
    # estimate first step based on harmonic guess
    target_energy = gamma_cutoff * scipy.constants.k * atom_cloud.T
    search_step_size = np.sqrt(2 * gamma_cutoff * scipy.constants.k * atom_cloud.T / atom_cloud.m) / beam.calculate_trapping_frequencies(atom_cloud.calculate_intensity_prefactor(), atom_cloud.m)
    
    x_direction = potential_line_search(beam, field, atom_cloud, target_energy, search_step_size[0], search_direction = np.array([[1,0,0]]), epsilon)
    y_direction = potential_line_search(beam, field, atom_cloud, target_energy, search_step_size[1], search_direction = np.array([[0,1,0]]), epsilon)
    z_direction = potential_line_search(beam, field, atom_cloud, target_energy, search_step_size[2], search_direction = np.array([[0,0,1]]), epsilon)
    size = x_direction + y_direction + z_direction
    return size[0]

# get rid off the hard coded max iter number
def potential_line_search(beam, field, atom_cloud, target_energy, initial_step_size, search_direction = np.array([[1,0,0]]), epsilon = 1E-8):
    iternum = 0
    step_size = initial_step_size
    current_position = np.array([[0,0,0]])
    while step_size > epsilon:
        next_position = current_position + search_direction * step_size
        next_potential = calculate_potential_energy(field, atom_cloud, beam.convert_local_vectors_to_global(next_position))
        if next_potential < target_energy:
            current_position = next_position
        else:
            step_size = step_size / 2
        iternum += 1
        if iternum > 1000:
            raise Exception("search did not converge")
    return current_position
    
def calculate_ideal_peak_density(field, atom_cloud):
    trapping_frequencies, vectors = field.calculate_trapping_frequencies(atom_cloud.calculate_intensity_prefactor(), atom_cloud.m)
    omega_bar_cubed = trapping_frequencies[0]* trapping_frequencies[1] * trapping_frequencies[2]
    return atom_cloud.N * (omega_bar_cubed) * np.power(atom_cloud.m / (2 * np.pi * scipy.constants.k * atom_cloud.T), 3/2)

def calculate_PSD(density, atom_cloud):
    lambda_DB = np.sqrt(2 * np.pi * scipy.constants.hbar**2 / (atom_cloud.m * scipy.constants.k * atom_cloud.T))
    return density * lambda_DB**3
    

