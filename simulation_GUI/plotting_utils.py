# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 14:36:31 2022

@author: kucer
"""

import numpy as np
import matplotlib.pyplot as plt

# input image size in pixels
def xz_summed_image(atomic_positions, real_pixel_size, magnification, image_size=np.array([100,100])):
    pixel_max = np.array([int(image_size[0] / 2), int(image_size[1] / 2)])
    x_coords = np.linspace(-pixel_max[0], pixel_max[0], 2 * pixel_max[0] + 1)
    y_coords = np.linspace(-pixel_max[1], pixel_max[1], 2 * pixel_max[1] + 1)
    x_grid, y_grid = np.meshgrid(x_coords, y_coords, indexing='ij')

    pixel_value = np.zeros(x_grid.shape)
    
    # assign positions to the nearest pixel
    apparent_pixel_size = real_pixel_size / magnification
    twoD_pixel_positions = np.rint(atomic_positions[:,[0,2]] / apparent_pixel_size)

    for pixel_position in twoD_pixel_positions:
        if np.all((pixel_position <= pixel_max) & (pixel_position >= -pixel_max)):
            index_x, index_y = (pixel_position + pixel_max).astype(int)
            pixel_value[index_x][index_y] += 1
        
    # now plot the heatmap
    fig, ax = plt.subplots()
    pix_min, pix_max = pixel_value.min(), pixel_value.max()
    c = ax.pcolormesh(x_grid, y_grid, pixel_value, cmap='viridis', vmin=pix_min, vmax=pix_max)
    ax.set_title('2D XZ summed image')
    ax.axis([x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()])
    fig.colorbar(c, ax=ax)
    plt.show()