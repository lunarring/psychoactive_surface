import numpy as np
import lunar_tools as lt
import random
from datasets import load_dataset
import random
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image, StableDiffusionXLControlNetPipeline
from diffusers import AutoencoderTiny
import torch
from prompt_blender import PromptBlender
from tqdm import tqdm
from sfast.compilers.diffusion_pipeline_compiler import (compile, CompilationConfig)

from u_unet_modulated import forward_modulated
import u_deepacid


#%% aux func
def get_prompts_and_img():
    list_imgs = []
    list_prompts = []
    for i in tqdm(range(nmb_rows*nmb_cols)):
        pb.set_prompt1(list_prompts_all[i])
        pb.set_prompt2(list_prompts_all[i])
        img = pb.generate_blended_img(0.0, latents)
        img_tile = img.resize(shape_hw[::-1])
        list_imgs.append(np.asarray(img_tile))
        list_prompts.append(list_prompts_all[i])
    return list_prompts, list_imgs

def get_aug_prompt(prompt):
    mod = ""
    if akai_midimix.get("A4", button_mode="toggle"):
        mod += "psychedelic "
    if akai_midimix.get("B4", button_mode="toggle"):
        mod += "electric "
    if akai_midimix.get("C4", button_mode="toggle"):
        mod += "surreal "
    if akai_midimix.get("D4", button_mode="toggle"):
        mod += "fractal "
    if akai_midimix.get("E4", button_mode="toggle"):
        mod += "spiritual "
    if akai_midimix.get("F4", button_mode="toggle"):
        mod += "metallic "
    if akai_midimix.get("G4", button_mode="toggle"):
        mod += "wobbly "
    if akai_midimix.get("H4", button_mode="toggle"):
        mod += "robotic "
    
    # if len(mod) > 0:
    #     mod = f"{mod}"
    
    prompt = f"{mod}{prompt}"
    print(prompt)
    return prompt

#%% inits
akai_midimix = lt.MidiInput("akai_midimix")

use_compiled_model = False

pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
pipe.to("cuda")
pipe.vae = AutoencoderTiny.from_pretrained('madebyollin/taesdxl', torch_device='cuda', torch_dtype=torch.float16)
pipe.vae = pipe.vae.cuda()
pipe.set_progress_bar_config(disable=True)

if use_compiled_model:
    pipe.enable_xformers_memory_efficient_attention()
    config = CompilationConfig.Default()
    config.enable_xformers = True
    config.enable_triton = True
    config.enable_cuda_graph = True
    pipe = compile(pipe, config)
else:
    pipe.unet.forward = lambda *args, **kwargs: forward_modulated(pipe.unet, *args, **kwargs)
    
    acidman = u_deepacid.AcidMan(0, akai_midimix, None)
    acidman.init('a01')

pb = PromptBlender(pipe)
pb.w = 64
pb.h = 64
latents = pb.get_latents()
secondary_renderer = lt.Renderer(width=1024, height=512, backend='opencv')
#%% prepare prompt window
nmb_rows,nmb_cols = (4,8)       # number of tiles

list_prompts_all = []
with open("good_prompts.txt", "r", encoding="utf-8") as file: 
    list_prompts_all = file.read().split('\n')
    
# Convert to images
shape_hw = (128, 256)   # image size
    
list_prompts, list_imgs = get_prompts_and_img()
gridrenderer = lt.GridRenderer(nmb_rows,nmb_cols,shape_hw)
gridrenderer.update(list_imgs)

#%%#

negative_prompt = "blurry, lowres, disfigured"
space_prompt = "photo of the moon"

# Run space
idx_cycle = 0
pb.set_prompt1(get_aug_prompt(space_prompt), negative_prompt)
latents2 = pb.get_latents()

modulations = {}
if not use_compiled_model:
    def noise_mod_func(sample):
        noise =  torch.randn(sample.shape, device=sample.device, generator=torch.Generator(device=sample.device).manual_seed(1))
        return noise    
    
    def acid_func(sample):
        amp = 1e-1
        resample_grid = acidman.do_acid(sample[0].float().permute([1,2,0]), amp)
        amp_mod = (resample_grid - acidman.identity_resample_grid)     
        return amp_mod[:,:,0][None][None], resample_grid
    
    modulations['noise_mod_func'] = noise_mod_func

while True:
    # cycle back from target to source
    latents1 = latents2.clone()
    pb.embeds1 = pb.embeds2
    # get new target
    latents2 = pb.get_latents()
    pb.set_prompt2(get_aug_prompt(space_prompt), negative_prompt)

    fract = 0
    while fract < 1:
        if not use_compiled_model:
            #modulations['b0_samp'] = akai_midimix.get("H0", val_min=0, val_max=100)
            modulations['b0_emb'] = akai_midimix.get("H1", val_min=0, val_max=1, val_default=1)
            
            for i in range(3):
                modulations[f'e{i}_emb'] = akai_midimix.get("H0", val_min=0, val_max=1, val_default=1, variable_name="bobo")
                modulations[f'd{i}_emb'] = akai_midimix.get("H2", val_min=0, val_max=1, val_default=1, variable_name="kobo")
                
            modulations['d1_acid'] = acid_func
        
        d_fract = akai_midimix.get("A0", val_min=0.0, val_max=0.1)
        latents_mix = pb.interpolate_spherical(latents1, latents2, fract)
        img_mix = pb.generate_blended_img(fract, latents_mix, modulations=modulations)
        secondary_renderer.render(img_mix)
        
        # Inject new space
        m,n = gridrenderer.render()
        if m != -1 and n != -1:
            idx = m*nmb_cols + n
            print(f'tile index: m {m} n {n} prompt {list_prompts[idx]}')
            
            # recycle old current embeddings and latents
            pb.embeds1 = pb.blend_prompts(pb.embeds1, pb.embeds2, fract)
            latents1 = pb.interpolate_spherical(latents1, latents2, fract)
            space_prompt = list_prompts[idx]
            fract = 0
            pb.set_prompt2(get_aug_prompt(space_prompt), negative_prompt)
        else:
            fract += d_fract
            
        do_new_prompts = akai_midimix.get("A3", button_mode="pressed_once")
        
        if do_new_prompts:
            print("getting new prompts...")
            list_prompts, list_imgs = get_prompts_and_img()
            gridrenderer.update(list_imgs)
            print("done!")
            
    idx_cycle += 1






            
    
    
    
    
    
    
    
    
    
    
