#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 10:01:50 2020

@author: jjj
"""

import os, sys
import numpy as np
import torch

from torch.nn import functional as F

#%%

def get_kernel(kernel_weights, gpu=None):
    assert gpu is not None, "pythoniac maniac gpuiac"
    assert len(kernel_weights.shape) == 2, '2D conv!'
    assert kernel_weights.shape[0] == kernel_weights.shape[1], 'square!'
    padding = int((kernel_weights.shape[0]-1) / 2)
    m = torch.nn.Conv2d(1, 1, kernel_weights.shape[0], padding=padding, stride=1)
    m.bias[0] = 0
    m.weight[0,0,:,:] = torch.from_numpy(kernel_weights.astype(np.float32))
    m = m.cuda(gpu)
    return m

def get_kernel_gauss(ksize=3, gpu=None):
    x, y = np.meshgrid(np.linspace(-1,1,ksize), np.linspace(-1,1,ksize))
    d = np.sqrt(x*x+y*y)
    sigma, mu = 1.0, 0.0
    g = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )
    g = g/np.sum(g)
    return get_kernel(g, gpu)

def get_kernel_cdiff(gpu=None):
    g = np.ones((3,3))
    g[1,1] = -8
    return get_kernel(g, gpu)

def apply_kernel(img, kernel, nmb_repetitions=1):
    if len(img.shape)==2:
        img = img.expand(1,1,img.shape[0],img.shape[1])
    else:
        print("WARNING! 3D NEEDS 3D KERNEL!")
        img = img.permute([2,0,1])
        img = img.expand(1,img.shape[0],img.shape[1],img.shape[2])
        
    for i in range(nmb_repetitions):
        img = kernel(img)
        
    return img.squeeze()

def torch_resize(img, new_sz=None, mode='bilinear', factor=None):
    if factor is not None:
        new_sz  = [int(img.shape[0]*factor), int(img.shape[1]*factor)]
    if len(img.shape) == 2:
        return F.interpolate(img.expand(1,1,img.shape[0],img.shape[1]), new_sz, mode=mode).squeeze()
    elif len(img.shape) == 3:                        # add singleton to batch dim
        return F.interpolate(img.unsqueeze(0).permute([0,3,1,2]), new_sz, mode=mode).permute([0,2,3,1])
    else:
        return F.interpolate(img.permute([0,3,1,2]), new_sz, mode=mode).permute([0,2,3,1])

def apply_kernel_lowres(img, kernel, factor_downscaling=10, mode='bilinear', post_kernel=False):
    old_sz = img.shape
    new_sz = [int(img.shape[0]/factor_downscaling), int(img.shape[1]/factor_downscaling)]
    img = torch_resize(img, new_sz, mode=mode)
    img = apply_kernel(img, kernel, 1)
    img = torch_resize(img, old_sz, mode=mode)
    if post_kernel:
        img = apply_kernel(img, kernel, 1)
    return img

def get_cartesian_resample_grid(shape_hw, gpu, use_half_precision=False):
    # initialize reesampling Cartesian grid
    theta = torch.zeros((1,2,3)).cuda(gpu)
    theta[0,0,0] = 1
    theta[0,1,1] = 1
    
    basegrid = F.affine_grid(theta, (1, 2, shape_hw[0], shape_hw[1]))
    iResolution = torch.tensor([shape_hw[0], shape_hw[1]]).float().cuda(gpu).unsqueeze(0).unsqueeze(0)

    grid = (basegrid[0,:,:,:] + 1) / 2
    grid *= iResolution
    cartesian_resample_grid = grid / iResolution

    if use_half_precision:
        cartesian_resample_grid = cartesian_resample_grid.half()
        
    return cartesian_resample_grid

class AcidMan():
    def __init__(self, gpu, midi_man, music_man=None, time_man=None):
        self.midi_man = midi_man
        self.music_man = music_man
        self.gpu = gpu
        self.do_acid = self.do_acid_not_init
        
        self.osc_amp_modulator = 0
        self.osc_kumulator = 0
        
        self.kum_t = 0
        
        with torch.no_grad():
            self.cartesian_resample_grid = None
            self.gkernel = get_kernel_gauss(gpu=gpu)
            self.ckernel = get_kernel_cdiff(gpu=gpu)
        self.ran_once = False
        self.fract_features = 1
        
        
    def do_acid_not_init(self):
        print("ACID NOT INITIALIZED! run e.g. init_j01")
        
    
    def get_resample_grid(self, shape_hw):
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
        freq_rot_new = self.midi_man.get("B5",val_min=0,val_max=3,val_default=0.0)
        freq_rot_new = freq_rot_new ** 2
        if freq_rot_new != self.freq_rot:
            self.phase_rot += self.kum_t*(self.freq_rot - freq_rot_new)
            self.freq_rot = freq_rot_new    
        
        # if self.midi_man.get_value("G4"):
        #     self.osc_kumulator += dt*self.osc_amp_modulator*1e-3
        
        self.kum_t += self.freq_rot * 0.1
        
        y_wobble = torch.sin(self.kum_t  + fsum_amp + self.osc_kumulator)
        x_wobble = torch.cos(self.kum_t  + fsum_amp + self.osc_kumulator)
        
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
    
    

                
                