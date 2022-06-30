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
        # self.plot_figure = plt.figure(figsize = (5,3), dpi = 100)
        # self.axis = self.plot_figure.add_axes([0.1,0.1,0.8,0.8])
        self.plot_figure, self.axis = plt.subplots()
        self.figure_to_canvas = FigureCanvasTkAgg(self.plot_figure, self.canvas)
        self.figure_to_canvas.get_tk_widget().pack(fill=tk.BOTH)
        self.current_frame_index = None
    
    
    def add_frame(self, frame):
        self.frames.append(frame)
        return
    
    def get_frame(self, index):
        return self.frames[index]
    
    def render_frame(self, frame_index):
        self.current_frame_index = frame_index
        positions_2d = self.get_frame(frame_index).get_2d_positions()
        self.axis.clear()
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
        self.axis.clear()
        return
    
class Frame:
    
    def __init__(self, positions):
        self.positions = positions
    
    def get_2d_positions(self):
        return self.positions[:,[0,2]]
        

class Player:
    def __init__(self, animation):
        self.animation = animation
        self.animation_speed = 1 # frames per second
        self.is_playing = False
        
    
    def play_animation(self):
        total_frame_number = self.animation.get_number_of_frames()
        current_frame = 2 ####
        self.is_playing = True
        while self.is_playing:
            # render next image and wait
            # if last image then is_playing = false
            # do a coroutine
            print("playing")
        
    def pause_animation(self):
        self.is_playing = False
        return
    
    def reset_animation(self):
        # render image 0;
        # is playing false
        return
        
    # have some kind of state isPlaying 
    # if yes then loop through frames of animation at certain speed (variable of player)
    
# what file format will we use?
# maybe a frame should also have a timestamp?
# atom number
# x y z
#atom number 
# x y z
# etc ....
