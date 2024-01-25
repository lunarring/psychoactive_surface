#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 22:13:24 2023

@author: lunar
"""

#%%
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image
import torch

from diffusers import AutoencoderTiny
from sfast.compilers.stable_diffusion_pipeline_compiler import (compile, CompilationConfig)
from diffusers.utils import load_image
import lunar_tools as lt
import numpy as np
from PIL import Image
from lunar_tools.comms import OSCSender, OSCReceiver

mainOSCReceiver = OSCReceiver('10.20.17.122', port_receiver=8005)

torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

use_maxperf = False
use_image_mode = False

if use_image_mode:
    pipe = AutoPipelineForImage2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
else:
    pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")

pipe.to("cuda")
pipe.vae = AutoencoderTiny.from_pretrained('madebyollin/taesdxl', torch_device='cuda', torch_dtype=torch.float16)
pipe.vae = pipe.vae.cuda()
pipe.set_progress_bar_config(disable=True)

if use_maxperf:
    config = CompilationConfig.Default()
    config.enable_xformers = True
    config.enable_triton = True
    config.enable_cuda_graph = True
    config.enable_jit = True
    config.enable_jit_freeze = True
    config.trace_scheduler = True
    config.enable_cnn_optimization = True
    config.preserve_parameters = False
    config.prefer_lowp_gemm = True
    
    pipe = compile(pipe, config)

#%%
torch.manual_seed(1)

noise_level = 0

#%%
# Image generation pipeline
sz = (512*2, 512*2)
renderer = lt.Renderer(width=sz[1], height=sz[0])
latents = torch.randn((1,4,64,64)).half().cuda() # 64 is the fastest
latents_additive = torch.randn((1,4,64,64)).half().cuda() # 64 is the fastest


# Iterate over blended prompts
# blended = blender.blend_prompts(blended_prompts[0], blended_prompts[1], 0)
# prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = blended

#%%#
while True:
    osc_data = mainOSCReceiver.get_last_value("/env1")
    print(f'data {osc_data}')
    
    # if data is not None:
    #     print(f'data {len(data)}')    
    
    # latents_modulated = latents + osc_data * latents_additive * 5*1e-1
    # latents_modulated = latents + torch.roll(latents_additive, int(osc_data*100)) * 0.2
    latents_modulated = torch.roll(latents_additive, int(osc_data*20))
    
    image = pipe(prompt='cat', guidance_scale=0.0, num_inference_steps=1, latents=latents_modulated).images[0]
        
    # Render the image
    image = np.asanyarray(image)
    image = np.uint8(image)
    renderer.render(image)
        
        
    
#%%
        
