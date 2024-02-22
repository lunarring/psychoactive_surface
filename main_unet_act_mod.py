#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 22:13:24 2023

@author: lunar
"""

#%%
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image, StableDiffusionXLControlNetPipeline
import torch

from diffusers import AutoencoderTiny
from sfast.compilers.stable_diffusion_pipeline_compiler import (compile, CompilationConfig)
from diffusers import ControlNetModel
from diffusers.utils import load_image
import lunar_tools as lt
import numpy as np
from PIL import Image
from lunar_tools.comms import OSCSender, OSCReceiver
from lunar_tools.control_input import MidiInput
from PIL import Image
import requests
import numpy as np
import cv2
import torch.nn.functional as F
import time

from prompt_blender import PromptBlender
import u_deepacid

# import sys, os
# dp_git = os.path.join(os.path.dirname(os.path.realpath(__file__)).split("git")[0]+"git")
# sys.path.append(os.path.join(dp_git,'garden4'))
# import general as gs

akai_midimix = MidiInput(device_name="akai_midimix")
#mainOSCReceiver = OSCReceiver('10.20.17.122', port_receiver=8011)
# mainOSCReceiver = OSCReceiver('10.20.17.122', port_receiver=8100)

acid_profile = 'a01'
acidman = u_deepacid.AcidMan(0, akai_midimix, None)
acidman.init(acid_profile)

torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

use_maxperf = False

pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
    
gpu_id = 0
    
pipe.to(f"cuda:{gpu_id}")
pipe.vae = AutoencoderTiny.from_pretrained('madebyollin/taesdxl', torch_device=f'cuda:{gpu_id}', torch_dtype=torch.float16)
pipe.vae = pipe.vae.cuda(gpu_id)
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
def get_ctrl_img(cam_img, ctrlnet_type, low_threshold=100, high_threshold=200):
    
    cam_img = np.array(cam_img)
    ctrl_image = cv2.Canny(cam_img, low_threshold, high_threshold)
    ctrl_image = ctrl_image[:, :, None]
    ctrl_image = np.concatenate([ctrl_image, ctrl_image, ctrl_image], axis=2)
    ctrl_image = Image.fromarray(ctrl_image)

    return ctrl_image 

def zoom_image(image, zoom_factor):
    if zoom_factor == 1.0:
        return image  # No change if zoom_factor is 1.0

    # Calculate the new dimensions
    height, width, _ = image.shape
    new_height = int(height / zoom_factor)
    new_width = int(width / zoom_factor)

    # Calculate the cropping coordinates to keep the center
    crop_x = (width - new_width) // 2
    crop_y = (height - new_height) // 2

    # Crop the image around its center
    cropped_image = image[crop_y:crop_y + new_height, crop_x:crop_x + new_width]

    # Resize the cropped image to the original size
    zoomed_image = cv2.resize(cropped_image, (width, height), interpolation=cv2.INTER_LINEAR)

    return zoomed_image



torch.manual_seed(1)
noise_level = 0

#%%

from u_unet_modulated import forward_modulated
pipe.unet.forward = lambda *args, **kwargs: forward_modulated(pipe.unet, *args, **kwargs)

# from sfast.compilers.diffusion_pipeline_compiler import (compile, CompilationConfig)
# pipe.enable_xformers_memory_efficient_attention()
# config = CompilationConfig.Default()
# config.enable_xformers = True
# config.enable_triton = True
# config.enable_cuda_graph = True
# pipe = compile(pipe, config)

do_record_movie = False
if do_record_movie:
    fp_movie = "/tmp/oceania3.mp4"
    fps = 24
    ms = lt.MovieSaver(fp_movie, fps=fps)
    
r = [1,2]

# draw
sz = (512*2*r[0], 512*2*r[1])
renderer = lt.Renderer(width=sz[1], height=sz[0], gpu_id=gpu_id)
latents = torch.randn((1,4,64*r[0],64*r[1])).half().cuda(gpu_id) # 64 is the fastest
latents_additive = torch.randn((1,4,64*r[0],64*r[1])).half().cuda(gpu_id) # 64 is the fastest

speech_detector = lt.Speech2Text()

prompts = []
#prompts.append('ocean wave approaching the shore')
#prompts.append('ocean breaking wave top view')
#prompts.append('painting of a super beautiful picturesque underwater ocean')
prompts.append('painting of a strange cat with fur full of electric sparks')
prompts.append('painting of a tree')
prompts.append('painting of a dog')
prompts.append('painting of a winter')
prompts.append('painting of a car')
prompts.append('painting of a hieronymus bosch')
prompts.append('painting of a wild fox')

negative_prompt = "text, frame, photorealistic, photo"

blender = PromptBlender(pipe, gpu_id=gpu_id)

blender.get_all_embeddings(prompts)
all_prompt_embeds = blender.prompts_embeds
tensor_all_prompt_embeds = torch.cat([all_prompt_embeds[i][0] for i in range(len(all_prompt_embeds))], dim=0)

blender.set_init_position(0)
blender.set_target(1)
velocity = 10

blender_step_debug = 0

embeds1 = all_prompt_embeds[0]
embeds2 = all_prompt_embeds[1]

#
m = 0
fract = 0

def noise_mod_func(sample):
    #noise = (torch.rand(sample.shape, device=sample.device) - 0.5)
    noise =  torch.randn(sample.shape, device=sample.device, generator=torch.Generator(device=sample.device).manual_seed(1))
    
    # freq = akai_midimix.get("H0", val_min=0, val_max=10, val_default=0)
    # ramp = torch.linspace(0,1,sample.shape[2], device=sample.device) * 2*np.pi * freq
    # sin_mod = torch.sin(ramp)
    # noise = sin_mod.reshape([1,1,1,sample.shape[2]])
    
    # noise = 1
    
    return noise

def acid_func(sample):
    amp = 1e-1
    resample_grid = acidman.do_acid(sample[0].float().permute([1,2,0]), amp)
    amp_mod = (resample_grid - acidman.identity_resample_grid)     
    return amp_mod[:,:,0][None][None], resample_grid

modulations = {}
modulations['noise_mod_func'] = noise_mod_func

prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = pb.get_prompt_embeds("blabla")

while True:
    torch.manual_seed(1)
    
    do_record_mic = akai_midimix.get("A3", button_mode="held_down")
    
    new_noise = akai_midimix.get("A4", button_mode="released_once")
    if new_noise:
        latents = torch.randn((1,4,64*r[0],64*r[1])).half().cuda(gpu_id) # 64 is the fastest

    if do_record_mic:
        if not speech_detector.audio_recorder.is_recording:
            speech_detector.start_recording()
    elif not do_record_mic:
        if speech_detector.audio_recorder.is_recording:
            try:
                prompt = speech_detector.stop_recording()
            except Exception as e:
                print(f"FAIL {e}")
            print(f"New prompt: {prompt}")
            if prompt is not None:
                embeds1 = embeds2
                embeds2 = blender.get_prompt_embeds(prompt, negative_prompt)
                print(f"norm: {torch.linalg.norm(embeds1[0] - embeds2[0]).item()}")
            stop_recording = False    
            
    fract = 0
    
    # prepare modulations
    
    # encoder mods
    for i in range(3):
        modulations[f'e{i}_samp'] = akai_midimix.get(f"A{i}", val_min=0, val_max=1, val_default=0) * (10**i) / 10
        modulations[f'e{i}_emb'] = akai_midimix.get(f"D{i}", val_min=0, val_max=1, val_default=1)
    
    # bottlneck mods
    modulations['b0_samp'] = akai_midimix.get("B0", val_min=0, val_max=1, val_default=0) * 10
    modulations['b0_emb'] = akai_midimix.get("E0", val_min=0, val_max=1, val_default=1)
    
    # acid
    # modulations['b0_acid'] = acid_func

    # decoder mods
    for i in range(3):
        modulations[f'd{i}_samp'] = akai_midimix.get(f"C{i}", val_min=0, val_max=1, val_default=0) * 10
        modulations[f'd{i}_emb'] = akai_midimix.get(f"F{i}", val_min=0, val_max=1, val_default=1)
        
    # acid
    # modulations['d0_acid'] = acid_func        
    modulations['d2_acid'] = acid_func
    
    prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = blender.blend_prompts(embeds1, embeds2, fract)
    
    image = pipe(guidance_scale=0.0, num_inference_steps=1, latents=latents, 
                  prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds, 
                  pooled_prompt_embeds=pooled_prompt_embeds, negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                  modulations=modulations, device=latents.device).images[0]
    
    m += 1
        
    # Render the image
    image = np.asanyarray(image)
    image = np.uint8(image)
    renderer.render(image)
    
    if do_record_movie:
        ms.write_frame(image)
        
if do_record_movie:
    ms.finalize()
        
 