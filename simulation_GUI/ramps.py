# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 08:33:49 2022

@author: kucer
"""

import numpy as np

class Ramp:
    def __init__(self):
        self.ramp_segments = []
        self.segment_times = []
        
        self.cache_time = None
        self.cache_value = None
        
    def add_ramp_segment(self, ramp_segment):
        self.ramp_segments.append(ramp_segment)
        if len(self.segment_times) > 0:
            self.segment_times.append(self.segment_times[-1] + ramp_segment.length)
        else:
            self.segment_times.append(ramp_segment.length)

    
    def get_value(self, time):
        if time == self.cache_time:
            return self.cache_value
        else:
            self.cache_time = time
            # empty ramp returns 0 when asked for value
            if len(self.ramp_segments) == 0:
                self.cache_value = 0
                return self.cache_value
            else:
                # negative times return initial value
                if time < 0:
                    self.cache_value = self.ramp_segments[0].start_value
                    return self.cache_value
                
                # at his point we have some segments and the time is positive
                for i, end_time in enumerate(self.segment_times):
                    if time < end_time:
                        #special case of the first segment
                        if i == 0:
                            self.cache_value = self.ramp_segments[i].get_value(time)
                            return self.cache_value
                        else:
                            self.cache_value = self.ramp_segments[i].get_value(time - self.segment_times[i - 1])
                            return self.cache_value
                
                # if we are at the end of the ramp then we return end value of the last segment
                self.cache_value = self.ramp_segments[-1].end_value
                return self.cache_value
    
class RampSegment:
    def __init__(self, start_value, end_value, length):
        self.start_value = start_value
        self.end_value = end_value
        self.length = length
    
    def get_value(self, time):
        pass

class ConstantSegment(RampSegment):
    def __init__(self, start_value, end_value, length):
        super(ConstantSegment, self).__init__(start_value, end_value, length)
        assert (self.start_value == self.end_value)
    
    def get_value(self, time):
        return self.start_value

class LinearSegment(RampSegment):
    def __init__(self, start_value, end_value, length):
        super(LinearSegment, self).__init__(start_value, end_value, length)
        self.slope = (self.end_value - self.start_value) / self.length
    
    def get_value(self, time):
        return self.start_value + self.slope * time
        

class ExponentialSegment(RampSegment):
    def __init__(self, start_value, end_value, length, tau):
        super(ExponentialSegment, self).__init__(start_value, end_value, length)
        self.tau = tau
        self.A = (self.end_value -self.start_value * np.exp(-self.length/ self.tau)) / (1 - np.exp(-self.length / self.tau))
        self.B = (self.end_value - self.start_value) / (np.exp(-self.length / self.tau) - 1)
        
    def get_value(self, time):
        return self.A + self.B * np.exp(-time/self.tau)
    

#TODO possibly make these into modulation of a segment rather than its own segment (LOW)
class GaussianNoiseSegment(RampSegment):
    def __init__(self, start_value, end_value, length, stdev):
        super(GaussianNoiseSegment, self).__init__(start_value, end_value, length)
        assert (self.start_value == self.end_value)
        self.stdev = stdev
        
    def get_value(self, time):
        return self.start_value + np.random.normal(loc=0.0, scale=self.stdev)
    
class OscillationSegment(RampSegment):
    def __init__(self, start_value, end_value, length, amplitude, frequency, phase):
        super(GaussianNoiseSegment, self).__init__(start_value, end_value, length)
        assert (self.start_value == self.end_value)
        self.amplitude = amplitude
        self.frequency = frequency
        self.phase = phase
    
    def get_value(self, time):
        return self.start_value + self.amplitude * np.sin(2 * np.pi * self.frequency * time + self.phase)


### OBSOLETE (for now) ###

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