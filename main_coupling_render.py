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


from prompt_blender import PromptBlender

akai_lpd8 = MidiInput(device_name="akai_midimix")
#mainOSCReceiver = OSCReceiver('10.20.17.122', port_receiver=8011)
mainOSCReceiver = OSCReceiver('10.20.17.122', port_receiver=8100)

torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

use_maxperf = False
engine_mode = 'ctrl'    # puretext, image2image, ctrl


if engine_mode == 'image2image':
    print('Using image2image node')
    pipe = AutoPipelineForImage2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
elif engine_mode == 'ctrl':
    print('Using control net node')
    ctrlnet_type = "diffusers/controlnet-canny-sdxl-1.0-mid"
    
    controlnet = ControlNetModel.from_pretrained(
        ctrlnet_type,
        variant="fp16",
        use_safetensors=True,
        torch_dtype=torch.float16,
    ).to("cuda")
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained("stabilityai/sdxl-turbo", controlnet=controlnet, torch_dtype=torch.float16, variant="fp16")    
else:
    print('Using pure text')
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
# Image generation pipeline
sz = (512*2, 512*2)
renderer = lt.Renderer(width=sz[1], height=sz[0])
latents = torch.randn((1,4,64,64)).half().cuda() # 64 is the fastest
latents_additive = torch.randn((1,4,64,64)).half().cuda() # 64 is the fastest

prompts = ['cat', 'cat and dog']

# Iterate over blended promptsblender = PromptBlender(pipe)

prompts = []
prompts.append('painting of a rodent')
prompts.append('painting of a tree')
prompts.append('painting of a house')
prompts.append('painting of a car')
prompts.append('painting of a face')
prompts.append('painting of a cat')
prompts.append('painting of a water pump')
prompts.append('painting of a freaky child')

blender = PromptBlender(pipe)

prompt_embeds = blender.get_all_embeddings(prompts)

blender.set_init_position(0)
blender.set_target(1)
velocity = 10
#%%

url = 'https://img.freepik.com/free-photo/close-up-white-rat_23-2150760678.jpg?size=626&ext=jpg&ga=GA1.1.1880011253.1698883200&semt=ais'
image_from_web = Image.open(requests.get(url, stream=True).raw)
initial_image = np.uint8(image_from_web)
sz = initial_image.shape
h = (sz[1] - sz[0])//2
initial_image = initial_image[:,h:-h,:]
initial_image = cv2.resize(initial_image, (latents_additive.shape[2]*8,)*2)

#
#n_steps = 2500
#blended_prompts = blender.blend_sequence_prompts(prompts, n_steps)


blender_step_debug = 0
initial_image_crop = initial_image[100:120, 100:120, :]

while True:
    
    # if data is not None:
    #     print(f'data {len(data)}')    
    
    # latents_modulated = latents + osc_data * latents_additive * 5*1e-1
    # latents_modulated = latents + torch.roll(latents_additive, int(osc_data*100)) * 0.2
    # latents_modulated = torch.roll(latents_additive, int(osc_data*20))
    
    osc_data_env1 = mainOSCReceiver.get_last_value("/env1")
    osc_data_env2 = mainOSCReceiver.get_last_value("/env2")
    osc_data_env3 = mainOSCReceiver.get_last_value("/tracker")
    
    # print(f'data {osc_data_env3[-1]}')        button
    
    # fract = float(i) / (len(blended_prompts) - 1)
    fract = 0
    
    # modulation gain
    a0 = akai_lpd8.get("A0", val_min=0, val_max=1, val_default=0)
    a1 = akai_lpd8.get("A1", val_min=0, val_max=1, val_default=0)
    a2 = akai_lpd8.get("A2", val_min=0, val_max=1e-1, val_default=0.001)
    
    # image2image strength
    image2image_strength = akai_lpd8.get("B0", val_min=0.5, val_max=1, val_default=0.75)
    image2image_guidance = akai_lpd8.get("B1", val_min=0.5, val_max=1, val_default=0.75)
    # print(f'image2image_strength {image2image_strength}')
    
    # image2image modulation gain
    c0 = akai_lpd8.get("C0", val_min=0, val_max=1, val_default=0)
    c1 = akai_lpd8.get("C1", val_min=0.1, val_max=1, val_default=0.5)  # zoom the canny
    
    low_threshold = akai_lpd8.get("F0", val_default=0, val_min=0, val_max=255)
    high_threshold = akai_lpd8.get("F1", val_default=70, val_min=0, val_max=255)        
    num_inference_steps = int(akai_lpd8.get("H1", val_min=2, val_max=5, val_default=2))

    controlnet_conditioning_scale = akai_lpd8.get("E0", val_min=0, val_max=1, val_default=0.5)        
    guidance_scale = 0.0
    
    do_canny_debug = not akai_lpd8.get("A3", button_mode='toggle')
    
    for bi in range(8):
        button = akai_lpd8.get(f"{chr(65+bi)}4", button_mode='pressed_once')
        if button:
            blender.set_target(bi)
            blender_step_debug = 0
    
    # here we modulate prompts transitions
    # i_b = i
    # i_b += (int(osc_data_env1*50000*a0)-0.5)
    # i_b = int(np.clip(i_b, 0, len(blended_prompts)-2))

    # blended = blender.blend_prompts(blended_prompts[i_b], blended_prompts[i_b+1], fract)
    # prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = blended
    
    blender_step_debug += a2
    
    if blender_step_debug > 1:
        blender_step_debug = 1
    blender.step(blender_step_debug)
    # blender.step(a2)
    
    # if blender.first_fract == 1:
    #     blender.first_fract = 0
    #     blender.set_target(np.random.randint(len(prompts)))
    
    prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = blender.current        
    
    # print(f'fract {blender.fract}')
    # print(f'a2 {a2}')
    
    # here we modulate noise
    height = latents.shape[2]
    vert_ramp = torch.linspace(0,1,height).to(latents.device)
    latents_modulated = latents + latents_additive * vert_ramp.reshape([1,1,height,1]) * a1
    latents_modulated = latents_modulated.half()
    
    # here we modulate image2image inputs
    initial_image_modulated = initial_image.copy()
    # initial_image_modulated[100:200,100:200,:] = initial_image_modulated[100:200,100:200,:] * c0
    
    # initial_image_modulated = np.roll(initial_image_modulated, int((c0-0.5)*1000))
    # initial_image_modulated = np.roll(initial_image_modulated, int((osc_data_env1)*1000*c0), axis=0)
    if engine_mode == 'image2image':
        c = initial_image_crop.shape[0]
        
        i_x = int(c0 * (initial_image_modulated.shape[1]-c))
        i_y = int(c1 * (initial_image_modulated.shape[0]-c))
        
        # initial_image_modulated[:, i_x, :] = 255
        # initial_image_modulated[i_y, :, :] = 255
        
        initial_image_modulated[i_y:i_y+c, i_x:i_x+c, :] = initial_image_crop
        
        if do_canny_debug:
            image = initial_image_modulated
        else:
            torch.manual_seed(0)
            image = pipe(image=Image.fromarray(initial_image_modulated), latents=latents, num_inference_steps=2, strength=image2image_strength, 
                         guidance_scale=image2image_guidance, prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds, 
                         pooled_prompt_embeds=pooled_prompt_embeds, negative_pooled_prompt_embeds=negative_pooled_prompt_embeds).images[0]
    
    
    
    elif engine_mode == 'ctrl':        
        torch.manual_seed(0)
        ctrl_img = get_ctrl_img(initial_image, ctrlnet_type, low_threshold=low_threshold, high_threshold=high_threshold)
        #ctrl_img = np.roll(np.array(ctrl_img), int(1000*c0), axis=1)
        ctrl_img = np.array(ctrl_img)
        i_x = int(c0 * (ctrl_img.shape[1]-1))
        i_y = int(c1 * (ctrl_img.shape[0]-1))
        
        ctrl_img[ctrl_img.shape[0]//2, i_x, :] = 255
        ctrl_img[i_y, ctrl_img.shape[1]//2, :] = 255
        
        # ctrl_img = zoom_image(np.array(ctrl_img), c1)
        if do_canny_debug:
            image = ctrl_img
        else:
            image = pipe(image=Image.fromarray(ctrl_img), latents=latents, controlnet_conditioning_scale=controlnet_conditioning_scale, guidance_scale=guidance_scale, num_inference_steps=num_inference_steps, prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds, pooled_prompt_embeds=pooled_prompt_embeds, negative_pooled_prompt_embeds=negative_pooled_prompt_embeds).images[0]            
    else:
        image = pipe(guidance_scale=0.0, num_inference_steps=1, latents=latents_modulated, 
                     prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds, 
                     pooled_prompt_embeds=pooled_prompt_embeds, negative_pooled_prompt_embeds=negative_pooled_prompt_embeds).images[0]
        
    # Render the image
    image = np.asanyarray(image)
    image = np.uint8(image)
    renderer.render(image)
        
        
    
#%%
        
