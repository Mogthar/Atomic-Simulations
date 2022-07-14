# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 08:33:49 2022

@author: kucer
"""
# define ramps in the format: ramp(time, ramp_time, start_value, end_value)

import numpy as np

def ConstantRamp(time, ramp_time, start_value, end_value):
    if time < 0:
        return 0
    elif time >= 0 and time <= ramp_time:
        return start_value
    else:
        return 0

def LinearRamp(time, ramp_time, start_value, end_value):
    if time < 0:
        return 0
    elif time >= 0 and time <= ramp_time:
        return start_value + (end_value - start_value) * time / ramp_time
    else:
        return 0

def ConstAccRamp(time, ramp_time, start_value, end_value):
    ramp_distance = end_value - start_value
    if time < 0:
        return 0
    elif time > 0 and time <= (ramp_time / 2):
        return (1/2) * (4 * ramp_distance / (ramp_time**2)) * (time**2) + start_value
    elif time > (ramp_time / 2) and time <= ramp_time:
        return -(1/2) * (4 * ramp_distance / (ramp_time**2)) * ((time-ramp_time)**2) + ramp_distance + start_value
    else:
        return 0

def CubicRamp(time, ramp_time, start_value, end_value):
    ramp_distance = end_value - start_value
    if time < 0:
        return 0
    elif time >= 0 and time <= ramp_time:
        return -2 * ramp_distance * (time / ramp_time)**3 + 3*ramp_distance * (time / ramp_time)**2 + start_value
    else:
        return 0

def ConstJerkRamp(time, ramp_time, start_value, end_value):
    ramp_distance = end_value - start_value
    if time < 0:
        return 0
    elif time > 0 and time <= (ramp_time / 4):
        return 16/3 * ramp_distance * (time / ramp_time)**3 + start_value
    elif time > (ramp_time / 4) and time <= (3* ramp_time / 4):
        return -ramp_distance*16/3*(time/ramp_time)**3 + 8*ramp_distance*(time/ramp_time)**2 - 2*ramp_distance*(time/ramp_time)+ramp_distance/6 + start_value
    elif time > (3* ramp_time / 4) and time <= ramp_time:
        return 16/3 * ramp_distance * ((time - ramp_time) / ramp_time)**3 + ramp_distance + start_value
    else:
        return 0
    
def ExponentialRamp(time, ramp_time, start_value, end_value):
    return 0