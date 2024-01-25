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
from lunar_tools.control_input import MidiInput

from prompt_blender import PromptBlender

akai_lpd8 = MidiInput(device_name="akai_midimix")
mainOSCReceiver = OSCReceiver('10.20.17.122', port_receiver=8004)

torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

use_maxperf = True
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

prompts = ['cat', 'cat and dog']

# Iterate over blended promptsblender = PromptBlender(pipe)

prompts = []
prompts.append('red strange carpet full of wool')
prompts.append('green tower full of mossy reird plants')

blender = PromptBlender(pipe)

#%%#
n_steps = 2500
blended_prompts = blender.blend_sequence_prompts(prompts, n_steps)


while True:
    
    # if data is not None:
    #     print(f'data {len(data)}')    
    
    # latents_modulated = latents + osc_data * latents_additive * 5*1e-1
    # latents_modulated = latents + torch.roll(latents_additive, int(osc_data*100)) * 0.2
    # latents_modulated = torch.roll(latents_additive, int(osc_data*20))
    
    for i in range(len(blended_prompts) - 1):
        osc_data = mainOSCReceiver.get_last_value("/env1")
        print(f'data {osc_data}')        
        
        # fract = float(i) / (len(blended_prompts) - 1)
        fract = 0
        
        # modulation gain
        a0 = akai_lpd8.get("A0", val_min=0, val_max=1, val_default=0)
        a1 = akai_lpd8.get("A1", val_min=0, val_max=10, val_default=0)
        
        i_b = i
        i_b += (int(osc_data*10000*a0)-0.5)
        i_b = int(np.clip(i_b, 0, len(blended_prompts)-2))
        
        height = latents.shape[2]
        vert_ramp = torch.linspace(0,1,height).to(latents.device)
        latents_modulated = latents + latents_additive * vert_ramp.reshape([1,1,height,1]) * a1
        latents_modulated = latents_modulated.half()
        
        blended = blender.blend_prompts(blended_prompts[i_b], blended_prompts[i_b+1], fract)
    
        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = blended
            
        image = pipe(guidance_scale=0.0, num_inference_steps=1, latents=latents_modulated, prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds, pooled_prompt_embeds=pooled_prompt_embeds, negative_pooled_prompt_embeds=negative_pooled_prompt_embeds).images[0]
        
        # Render the image
        image = np.asanyarray(image)
        image = np.uint8(image)
        renderer.render(image)
        
        
    
#%%
        
