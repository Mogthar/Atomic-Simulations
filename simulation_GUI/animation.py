# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 22:21:43 2022

@author: kucer
"""

import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class Animation:
    def __init__(self, canvas):
        self.frames = []
        self.canvas = canvas
        self.plot_figure = plt.figure(figsize = (5,3), dpi = 100)
        self.axis = self.plot_figure.add_subplot(111)
        self.figure_to_canvas = FigureCanvasTkAgg(self.plot_figure, self.canvas)
        self.figure_to_canvas.get_tk_widget().pack()
        
    
    
    def add_frame(self, frame):
        self.frames.append(frame)
        return
    
    def get_frame(self, index):
        return self.frames[index]
    
    def render_frame(self, frame):
        positions_2d = frame.get_2d_positions()
        self.axis.scatter(positions_2d[:,0], positions_2d[:,1])
        self.figure_to_canvas.draw()
        return 
    
    def get_number_of_frames(self):
        return len(self.frames)
    
    def load_frames_from_file(self, file):
        with file as f:
            while True:
                atom_number_str = f.readline()
                # check for EoF
                if atom_number_str == '':
                    break
                else:
                    atom_number = int(atom_number_str)
                    atomic_coordinates = np.zeros((atom_number,3))
                    for i in range(atom_number):
                        line = f.readline()
                        atomic_coordinates[i] = [float(x) for x in line.split()]
                    self.add_frame(Frame(atomic_coordinates))
        return
        
    def reset(self):
        self.frames = []
        return
    
class Frame:
    
    def __init__(self, positions):
        self.positions = positions
    
    def get_2d_positions(self):
        return self.positions[:,[0,2]]
        
        
# what file format will we use?
# maybe a frame should also have a timestamp?
# atom number
# x y z
#atom number 
# x y z
# etc ....
