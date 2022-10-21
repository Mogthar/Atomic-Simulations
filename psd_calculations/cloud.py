# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 11:56:29 2022

@author: kucer
"""
import scipy.constants

class Cloud():
    
    def __init__(self, temperature = 1E-6, N_particles = 1E6, mass = 0.0, polarizability = 0.0):
        self.T = temperature
        self.N = N_particles
        self.m = mass
        self.alpha = polarizability
        
    def calculate_intensity_prefactor(self):
        return self.alpha / (2 * scipy.constants.epsilon_0 * scipy.constants.c)