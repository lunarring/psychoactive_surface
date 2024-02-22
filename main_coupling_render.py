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
import time
from u_unet_modulated import forward_modulated
import u_deepacid
import hashlib
from PIL import Image
import os


#%% VARS
use_compiled_model = False
width_latents = 128
height_latents = 64
shape_hw_prev = (128, 256)   # image size
ip_address_osc_receiver = '10.40.48.97'
dir_embds_imgs = "embds_imgs"

#%% aux func
def get_prompts_and_img():
    list_imgs = []
    list_prompts = []
    for i in tqdm(range(nmb_rows*nmb_cols)):
        prompt = random.choice(list_prompts_all)

        hash_object = hashlib.md5(prompt.encode())
        hash_code = hash_object.hexdigest()[:6].upper()

        fp_img = f"{dir_embds_imgs}/{hash_code}.jpg"

        if os.path.exists(fp_img):
            img_tile = Image.open(fp_img)
        else:
            latents = pb.get_latents()
            prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = pb.get_prompt_embeds(prompt, negative_prompt)
            img = pb.generate_img(latents, prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds)
            img_tile = img.resize(shape_hw_prev[::-1])
        list_imgs.append(np.asarray(img_tile))
        list_prompts.append(prompt)
    return list_prompts, list_imgs

def get_aug_prompt(prompt):
    # mod = ""
    # if akai_midimix.get("A4", button_mode="toggle"):
    #     mod = "psychedelic "
    # if akai_midimix.get("B4", button_mode="toggle"):
    #     mod = "dark "
    # if akai_midimix.get("C4", button_mode="toggle"):
    #     mod = "bright "
    # if akai_midimix.get("D4", button_mode="toggle"):
    #     mod = "fractal "
    # if akai_midimix.get("E4", button_mode="toggle"):
    #     mod = "organic "
    # if akai_midimix.get("F4", button_mode="toggle"):
    #     mod = "metallic "
    # if akai_midimix.get("G4", button_mode="toggle"):
    #     mod = "weird and strange "
    # if akai_midimix.get("H4", button_mode="toggle"):
    #     mod = "robotic "

    # if mod != "":
    #     prompt = f"very {mod}, {prompt} looking very {mod}"
    # print(prompt)
    return prompt

#%% inits
meta_input = lt.MetaInput()


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
    
    acidman = u_deepacid.AcidMan(0, meta_input, None)
    acidman.init('a01')

pb = PromptBlender(pipe)
pb.w = width_latents
pb.h = height_latents
latents = pb.get_latents()
secondary_renderer = lt.Renderer(width=1024*2, height=512*2, backend='opencv')
#%% prepare prompt window
nmb_rows,nmb_cols = (4,8)       # number of tiles



list_prompts_all = []
with open("good_prompts.txt", "r", encoding="utf-8") as file: 
    list_prompts_all = file.read().split('\n')
    
    
list_prompts, list_imgs = get_prompts_and_img()
gridrenderer = lt.GridRenderer(nmb_rows, nmb_cols, shape_hw_prev)
gridrenderer.update(list_imgs)

show_osc_visualization = False

receiver = lt.OSCReceiver(ip_address_osc_receiver)
if show_osc_visualization:
    receiver.start_visualization(shape_hw_vis=(300, 500), nmb_cols_vis=3, nmb_rows_vis=2,backend='opencv')

speech_detector = lt.Speech2Text()
#%%#
negative_prompt = "blurry, lowres, disfigured"
space_prompt = list_prompts[0]

# Run space
idx_cycle = 0
pb.set_prompt1(get_aug_prompt(space_prompt), negative_prompt)
pb.set_prompt2(get_aug_prompt(space_prompt), negative_prompt)
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
    modulations['d*_extra_embeds'] = pb.get_prompt_embeds("full of electric sparkles")[0]


embeds_mod_full = pb.get_prompt_embeds("full of electric sparkles")

pb.num_inference_steps = 1

t_last = time.time()

is_noise_trans = True

while True:
    # cycle back from target to source
    latents1 = latents2.clone()
    pb.embeds1 = pb.embeds2
    # get new target
    latents2 = pb.get_latents()
    pb.set_prompt2(get_aug_prompt(space_prompt), negative_prompt)
    fract = 0

    while fract < 1:
        dt = time.time() - t_last
        lt.dynamic_print(f"fps: {1/dt:.1f}")
        t_last = time.time()
        if show_osc_visualization:
            receiver.show_visualization()
        osc_low = receiver.get_last_value("/low")
        osc_mid = receiver.get_last_value("/mid")
        osc_high = receiver.get_last_value("/high")
        
        if not use_compiled_model:
            H0 = meta_input.get(akai_midimix="H0", val_min=0, val_max=10, val_default=1)
            modulations['b0_samp'] = H0 * osc_low
            # modulations['b0_emb'] = H0 * osc_low
            #acidman.osc_kumulator += H0 * osc_low
            
            H1 = meta_input.get(akai_midimix="H1", val_min=0, val_max=10, val_default=0, variable_name="bobo")
            H2 = meta_input.get(akai_midimix="H2", val_min=0, val_max=10, val_default=0, variable_name="kobo")
            
            for i in range(3):
                modulations[f'e{i}_emb'] = 1 - (H1*osc_mid)
                modulations[f'd{i}_emb'] = 1 - (H2*osc_high)
                
            modulations['d2_acid'] = acid_func
            
            # EXPERIMENTAL WHISPER
            do_record_mic = meta_input.get(akai_midimix="A3", button_mode="held_down")
            # do_record_mic = akai_lpd8.get('s', button_mode='pressed_once')
            
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
                        embeds_mod_full = pb.get_prompt_embeds(prompt)
                    stop_recording = False
            
            fract_mod = meta_input.get(akai_midimix="G0", val_default=0, val_max=2, val_min=0)
            embeds_mod = pb.blend_prompts(pb.embeds1, embeds_mod_full, fract_mod)
            modulations['d*_extra_embeds'] = embeds_mod[0]
        
        # d_fract = akai_midimix.get("A0", val_min=0.0, val_max=0.1, val_default=0)
        d_fract_noise = meta_input.get(akai_lpd8="E0", akai_midimix="A0", val_min=0.0, val_max=0.1, val_default=0)
        d_fract_embed = meta_input.get(akai_lpd8="E1", akai_midimix="A1", val_min=0.0, val_max=0.1, val_default=0)
        
        #d_fract *= osc_low
        
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
            is_noise_trans = False
        else:
            if is_noise_trans:
                fract += d_fract_noise
            else:
                fract += d_fract_embed
            
        do_new_prompts = meta_input.get(akai_midimix="A4", button_mode="pressed_once")
        
        if do_new_prompts:
            print("getting new prompts...")
            list_prompts, list_imgs = get_prompts_and_img()
            gridrenderer.update(list_imgs)
            print("done!")
            
    idx_cycle += 1
    is_noise_trans = True






            
    
    
    
    
    
    
    
    
    
    
