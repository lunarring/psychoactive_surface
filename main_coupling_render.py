import sys
sys.path.append('../')

import numpy as np
import lunar_tools as lt
import random
from datasets import load_dataset
import random
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image, StableDiffusionXLControlNetPipeline
from diffusers.models import AutoencoderKL, ImageProjection, UNet2DConditionModel
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
use_compiled_model = True
width_latents = 128
height_latents = 64
shape_hw_prev = (128, 256)   # image size
# ip_address_osc_receiver = '10.40.48.82'
ip_address_osc_receiver = '192.168.50.13'
dir_embds_imgs = "embds_imgs"
show_osc_visualization = True

#%% aux func

class AudioVisualRouter():
    def __init__(self, meta_input):
        self.meta_input = meta_input
        self.sound_features = {}
        self.visual2sound = {}
        
    def map_av(self, sound_feature_name, visual_effect_name):
        self.visual2sound[visual_effect_name] = sound_feature_name
        
    def update_sound(self, sound_feature_name, value):
        self.sound_features[sound_feature_name] = value
        
    def get_modulation(self, visual_effect_name):
        return self.sound_features[self.visual2sound[visual_effect_name]]
        


def get_prompts_and_img():
    negative_prompt = "blurry, lowres, disfigured"
    
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


#%% inits
meta_input = lt.MetaInput()

pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
pipe.to("cuda")
pipe.vae = AutoencoderTiny.from_pretrained('madebyollin/taesdxl', torch_device='cuda', torch_dtype=torch.float16)
pipe.vae = pipe.vae.cuda()
pipe.set_progress_bar_config(disable=True)

pipe.unet.forward = forward_modulated.__get__(pipe.unet, UNet2DConditionModel)

if use_compiled_model:
    pipe.enable_xformers_memory_efficient_attention()
    config = CompilationConfig.Default()
    config.enable_xformers = True
    config.enable_triton = True
    config.enable_cuda_graph = True
    pipe = compile(pipe, config)

acidman = u_deepacid.AcidMan(0, meta_input, None)
acidman.init('a01')

pb = PromptBlender(pipe)
pb.w = width_latents
pb.h = height_latents
latents = pb.get_latents()
secondary_renderer = lt.Renderer(width=1024*2, height=512*2, backend='opencv')

def get_sample_shape_unet(coord):
    if coord[0] == 'e':
        coef = float(2**int(coord[1]))
        shape = [int(np.ceil(height_latents/coef)), int(np.ceil(width_latents/coef))]
    elif coord[0] == 'b':
        shape = [int(np.ceil(height_latents/4)), int(np.ceil(width_latents/4))]
    else:
        coef = float(2**(2-int(coord[1])))
        shape = [int(np.ceil(height_latents/coef)), int(np.ceil(width_latents/coef))]
        
    return shape


#%% prepare prompt window
nmb_rows,nmb_cols = (4,8)       # number of tiles

fn_prompts = 'good_prompts'
fn_prompts = 'gonsalo_prompts'

list_prompts_all = []
with open(f"{fn_prompts}.txt", "r", encoding="utf-8") as file: 
    list_prompts_all = file.read().split('\n')
    
list_prompts_all = [line for line in list_prompts_all if len(line) > 5]
    
list_prompts, list_imgs = get_prompts_and_img()
gridrenderer = lt.GridRenderer(nmb_rows, nmb_cols, shape_hw_prev)
gridrenderer.update(list_imgs)



receiver = lt.OSCReceiver(ip_address_osc_receiver)
if show_osc_visualization:
    receiver.start_visualization(shape_hw_vis=(300, 500), nmb_cols_vis=3, nmb_rows_vis=2,backend='opencv')

speech_detector = lt.Speech2Text()
#%%#
av_router = AudioVisualRouter(meta_input)

negative_prompt = "blurry, lowres, disfigured"
space_prompt = list_prompts[0]

# Run space
idx_cycle = 0
pb.set_prompt1(space_prompt, negative_prompt)
pb.set_prompt2(space_prompt, negative_prompt)
latents2 = pb.get_latents()

def get_noise_for_modulations(shape):
    return torch.randn(shape, device=pipe.device, generator=torch.Generator(device=pipe.device).manual_seed(1)).half()

modulations = {}
modulations_noise = {}
for i in range(3):
    modulations_noise[f'e{i}'] = get_noise_for_modulations(get_sample_shape_unet(f'e{i}'))
    modulations_noise[f'd{i}'] = get_noise_for_modulations(get_sample_shape_unet(f'd{i}'))
    
modulations_noise['b0'] = get_noise_for_modulations(get_sample_shape_unet('b0'))
    
# if not use_compiled_model or True:
#     def noise_mod_func(sample):
#         noise =  torch.randn(sample.shape, device=sample.device, generator=torch.Generator(device=sample.device).manual_seed(1))
#         return noise    
    
#     def acid_func(sample):
#         amp = 1e-1
#         resample_grid = acidman.do_acid(sample[0].float().permute([1,2,0]), amp)
#         amp_mod = (resample_grid - acidman.identity_resample_grid)     
#         return amp_mod[:,:,0][None][None], resample_grid
#     modulations['noise_mod_func'] = noise_mod_func
#     modulations['d*_extra_embeds'] = pb.get_prompt_embeds("full of electric sparkles")[0]


embeds_mod_full = pb.get_prompt_embeds("full of electric sparkles")

pb.num_inference_steps = 1

t_last = time.time()

sound_feature_names = ['low', 'mid', 'high']

# av_router.map_av('low', 'b0_samp')
av_router.map_av('mid', 'e*_emb')
av_router.map_av('high', 'd*_emb')
av_router.map_av('low', 'fract_decoder_emb')

is_noise_trans = True
while True:
    # cycle back from target to source
    latents1 = latents2.clone()
    pb.embeds1 = pb.embeds2
    # get new target
    latents2 = pb.get_latents()
    pb.set_prompt2(space_prompt, negative_prompt)
    fract = 0

    while fract < 1:
        dt = time.time() - t_last
        lt.dynamic_print(f"fps: {1/dt:.1f}")
        t_last = time.time()
        show_osc_visualization = meta_input.get(akai_lpd8="B0", button_mode="toggle")
        if show_osc_visualization:
            receiver.show_visualization()
        
        # update oscs
        for name in sound_feature_names:
            av_router.update_sound(f'{name}', receiver.get_last_value(f"/{name}"))
            # print(f'{name} {receiver.get_last_value(f"/{name}")}')
        
        # modulate osc with akai
        H0 = meta_input.get(akai_midimix="H0", akai_lpd8="G0", val_min=0, val_max=10, val_default=1)
        av_router.sound_features['low'] *= H0
        
        H1 = meta_input.get(akai_midimix="H1", akai_lpd8="G1",val_min=0, val_max=10, val_default=1)
        av_router.sound_features['mid'] *= H1
        
        H2 = meta_input.get(akai_midimix="H2", akai_lpd8="H0", val_min=0, val_max=10, val_default=1)
        av_router.sound_features['high'] *= H2
        
        #modulations['b0_samp'] = av_router.sound_features[av_router.visual2sound['b0_samp']]
        
        def get_modulation(self, visual_effect_name):
            return self.sound_features[self.visual2sound[visual_effect_name]]
        
        for i in range(3):
            modulations[f'e{i}_emb'] = torch.tensor(1 - av_router.get_modulation('e*_emb'), device=latents1.device)
            modulations[f'd{i}_emb'] = torch.tensor(1 - av_router.get_modulation('d*_emb'), device=latents1.device)
            
        amp = 1e-1
        resample_grid = acidman.do_acid(modulations_noise['d2'][None].float().permute([1,2,0]), amp)
        amp_mod = (resample_grid - acidman.identity_resample_grid)     
      
        acid_fields = amp_mod[:,:,0][None][None], resample_grid
        modulations['d2_acid'] = acid_fields
        
        # EXPERIMENTAL WHISPER
        do_record_mic = meta_input.get(akai_midimix="A3", akai_lpd8="A0", button_mode="held_down")
        # do_record_mic = akai_lpd8.get('s', button_mode='pressed_once')
        
        try:
            if do_record_mic:
                if not speech_detector.audio_recorder.is_recording:
                    speech_detector.start_recording()
            elif not do_record_mic:
                if speech_detector.audio_recorder.is_recording:
                    prompt = speech_detector.stop_recording()
                    print(f"New prompt: {prompt}")
                    if prompt is not None:
                        embeds_mod_full = pb.get_prompt_embeds(prompt)
                    stop_recording = False
        except Exception as e:
            print(f"FAIL {e}")
        
        # fract_mod = meta_input.get(akai_midimix="G0", akai_lpd8="F0", val_default=0, val_max=2, val_min=0)
        #fract_mod = av_router.sound_features[av_router.visual2sound['fract_decoder_emb']]
        # fract_mod = meta_input.get(akai_midimix="G0", val_default=0, val_max=2, val_min=0)
        fract_mod = av_router.get_modulation('fract_decoder_emb')
        embeds_mod = pb.blend_prompts(pb.embeds1, embeds_mod_full, fract_mod)
        modulations['d*_extra_embeds'] = embeds_mod[0]
        
        # d_fract = akai_midimix.get("A0", val_min=0.0, val_max=0.1, val_default=0)
        d_fract_noise = meta_input.get(akai_lpd8="E0", akai_midimix="A0", val_min=0.0, val_max=0.1, val_default=0)
        d_fract_embed = meta_input.get(akai_lpd8="E1", akai_midimix="A1", val_min=0.0, val_max=0.1, val_default=0)
        
        #d_fract *= osc_low
        
        cross_attention_kwargs ={}
        cross_attention_kwargs['modulations'] = modulations        
        
        latents_mix = pb.interpolate_spherical(latents1, latents2, fract)
        img_mix = pb.generate_blended_img(fract, latents_mix, cross_attention_kwargs=cross_attention_kwargs)
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
            pb.set_prompt2(space_prompt, negative_prompt)
            is_noise_trans = False
        else:
            if is_noise_trans:
                fract += d_fract_noise
            else:
                fract += d_fract_embed
            
        do_new_prompts = meta_input.get(akai_midimix="A4", akai_lpd8="A1", button_mode="pressed_once")
        
        if do_new_prompts:
            print("getting new prompts...")
            list_prompts, list_imgs = get_prompts_and_img()
            gridrenderer.update(list_imgs)
            print("done!")
            
    idx_cycle += 1
    is_noise_trans = True






            
    
    
    
    
    
    
    
    
    
    
