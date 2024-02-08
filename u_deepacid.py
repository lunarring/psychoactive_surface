#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 10:01:50 2020

@author: jjj
"""

import os, sys
import numpy as np
import torch

dp_git = os.path.join(os.path.dirname(os.path.realpath(__file__)).split("git")[0]+"git")
sys.path.append(os.path.join(dp_git,'garden4'))

import general as gs
from u_torch import torch_resample, torch_resize, GaussianSmoothing, tshow, apply_kernel_lowres, apply_kernel, get_kernel_cdiff, get_kernel_gauss, get_kernel, get_cartesian_resample_grid

#%%

class AcidMan():
    def __init__(self, gpu, midi_man, music_man=None, time_man=None):
        self.midi_man = midi_man
        self.music_man = music_man
        self.gpu = gpu
        self.do_acid = self.do_acid_not_init
        
        self.osc_amp_modulator = 0
        self.osc_kumulator = 0
        
        if time_man is None:
            self.tm = gs.TimeMan()
        else:
            self.tm = time_man
        
        with torch.no_grad():
            self.cartesian_resample_grid = None
            self.gkernel = get_kernel_gauss(gpu=gpu)
            self.ckernel = get_kernel_cdiff(gpu=gpu)
        self.ran_once = False
        self.fract_features = 1
        
        
    def do_acid_not_init(self):
        print("ACID NOT INITIALIZED! run e.g. init_j01")
        
    
    def get_resample_grid(self, shape_hw):
        # if self.cartesian_resample_grid is None:
        #     self.cartesian_resample_grid = get_cartesian_resample_grid(shape_hw, self.gpu)
        # resample_grid = self.cartesian_resample_grid.clone() 
        resample_grid = get_cartesian_resample_grid(shape_hw, self.gpu)
        return resample_grid

    
    def init(self, profile_name='j01'):
        init_function = "init_{}".format(profile_name)
        assert hasattr(self, init_function), "acid_man: unknown profile! {} not found".format(init_function)
        
        self.init_function = getattr(self, init_function)
        self.init_function()
        acid_function = "do_acid_{}".format(profile_name)
        assert hasattr(self, acid_function), "acid_man: unknown profile! {} not found".format(acid_function)
        self.do_acid = getattr(self, acid_function)
        

    def init_a01(self):
        self.phase_rot = 0
        self.freq_rot = 0
        self.is_fluid = False
        
    def get_active_resolutions(self, maxdim=1024):
        
        # resolution_list = np.array(2**np.arange(3,11))
        if maxdim == 1024:
            resolution_mods = np.zeros((8,), dtype=np.float32)
            offset = 0
        else:
            resolution_mods = np.zeros((9,), dtype=np.float32)
            offset = 1
        
        for idx, l in enumerate([chr(x) for x in range(65,73)]):
            amp = self.midi_man.get(l+"5",val_min=0,val_max=1,val_default=0)
            resolution_mods[idx+offset] = amp
            
        return resolution_mods
    
        
    def do_acid_a01(self, source, amp):
        
        dt = self.tm.get_dt()
        frame_sum = torch.sum(source, axis=2)
        
        frame_sum -= frame_sum.min()
        frame_sum /= frame_sum.max()
        
        do_apply_kernel_lowres = True
        if source.shape[0] < 256:
            do_apply_kernel_lowres = False
        
        if do_apply_kernel_lowres:
            edges = apply_kernel_lowres(frame_sum, self.ckernel,1)
        else:
            edges = apply_kernel(frame_sum, self.ckernel)
    
        # where
        edges = edges.abs()
        edges /= edges.max()
        factor = int(self.midi_man.get("E5",val_min=1,val_max=100,val_default=25))
        
        if do_apply_kernel_lowres:
            edges = apply_kernel_lowres(edges, self.gkernel, factor, post_kernel=True)
        else:
            edges = apply_kernel(edges, self.gkernel, factor)
        edges /= edges.max()
        
        
        edges = 1 - edges
        edges *= self.midi_man.get("D5",val_min=0.0,val_max=10,val_default=2.5)
        
        
        # which phase
        factor = int(self.midi_man.get("C5",val_min=1,val_max=100,val_default=50))
        
        if do_apply_kernel_lowres:
            fsum_amp = apply_kernel_lowres(frame_sum, self.gkernel, factor)
        else:
            fsum_amp = apply_kernel(frame_sum, self.gkernel, factor)
        fsum_amp -= fsum_amp.min()
        fsum_amp /= fsum_amp.max()
        fsum_amp *= 2*np.pi
        
        # xy modulatioself.nS: frequency
        freq_rot_new = self.midi_man.get("B5",val_min=0,val_max=10.5,val_default=0.0)
        if freq_rot_new != self.freq_rot:
            self.phase_rot += dt*(self.freq_rot - freq_rot_new)
            self.freq_rot = freq_rot_new    
        
        # if self.midi_man.get_value("G4"):
        #     self.osc_kumulator += dt*self.osc_amp_modulator*1e-3
        
        y_wobble = torch.sin(dt * self.freq_rot  + fsum_amp + self.osc_kumulator)
        x_wobble = torch.cos(dt * self.freq_rot  + fsum_amp + self.osc_kumulator)
        
        kibbler_coef = self.midi_man.get("A5",val_min=0,val_max=10.9, val_default=0.0)
        
        v_edges = edges * y_wobble
        h_edges = edges * x_wobble
        
        # if self.midi_man.get_value("D3"):
        #     h_edges *= 0
        
        # if self.midi_man.get_value("D4"):
        #     vmod = torch.linspace(1.5,-0.5,v_edges.shape[0], device=v_edges.device)
        #     v_edges *= vmod.unsqueeze(1) 
        #     h_edges *= vmod.unsqueeze(1) 
        
        shape_hw = source.shape
        resample_grid = self.get_resample_grid(shape_hw)
        self.identity_resample_grid = resample_grid.clone()
        resample_grid[:,:,0] += v_edges * amp * (kibbler_coef + self.osc_amp_modulator)
        resample_grid[:,:,1] += h_edges * amp * (kibbler_coef + self.osc_amp_modulator)
        
        return resample_grid
    
    

                
                