# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 22:21:43 2022

@author: kucer
"""

import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import asyncio

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
        self.slider = None
        self.startPause_button = None
        
        self.player = Player(self)
    
    
    def add_frame(self, frame):
        self.frames.append(frame)
        return
    
    def get_frame(self, index):
        return self.frames[index]
    
    def get_current_frame_index(self):
        return self.current_frame_index
    
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
        self.reset()
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
        self.animation_speed = 2 # frames per second
        self.is_playing = False
        
    
    def play_animation(self):
        self.is_playing = True
        self.animation.startPause_button['text'] = 'PAUSE'
        # asyncio.run(self.loop_images())
        self.loop_images()
    
    #async def loop_images(self):
    def loop_images(self):
        while self.is_playing:
            next_frame_index = self.animation.get_current_frame_index() + 1
            if next_frame_index >= self.animation.get_number_of_frames():
                self.is_playing = False
                self.animation.startPause_button['text'] = 'START'
            else:
                self.animation.render_frame(next_frame_index)
                self.animation.slider.set(next_frame_index)
                self.animation.render_frame(next_frame_index)
                # await asyncio.sleep(1 / self.animation_speed)
                
        
        
        
    def pause_animation(self):
        self.is_playing = False
        self.animation.startPause_button['text'] = 'START'
        return
    
    def stop_animation(self):
        self.is_playing = False
        self.animation.startPause_button['text'] = 'START'
        self.animation.slider.set(0)
        self.animation.render_frame(0)
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
