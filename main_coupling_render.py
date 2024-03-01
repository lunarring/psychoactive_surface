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
import cv2


#%% VARS
use_compiled_model = True
use_image2image = False
res_fact = 1
width_latents = int(96*res_fact)
height_latents = int(64*res_fact)
width_renderer = int(1024*1.5)
height_renderer = 512*2

shape_hw_prev = (2*height_latents, 2*width_latents)   # image size
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

if use_image2image:
    pipe = AutoPipelineForImage2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16", local_files_only=True)
else:
    pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16", local_files_only=True)
    
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
secondary_renderer = lt.Renderer(width=width_renderer, height=height_renderer, backend='opencv')

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
nmb_rows,nmb_cols = (6,6)       # number of tiles

fn_prompts = 'good_prompts'
fn_prompts = 'gonsalo_prompts'
# fn_prompts = 'prompts/underwater'
# fn_prompts = 'prompts/robot'
# fn_prompts = 'prompts/water.txt'

list_prompts_all = []
if not fn_prompts.endswith(".txt"):
    fn_prompts += ".txt"
with open(f"{fn_prompts}", "r", encoding="utf-8") as file: 
    list_prompts_all = file.read().split('\n')
    
list_prompts_all = [line for line in list_prompts_all if len(line) > 5]

gridrenderer = lt.GridRenderer(nmb_rows, nmb_cols, shape_hw_prev)

if not use_image2image:    
    list_prompts, list_imgs = get_prompts_and_img()
    gridrenderer.update(list_imgs)
else:
    list_prompts = ['water surface ripples 4K high res']
    fp_movie = '/home/lugo/Downloads/20240301_150735.mp4'
    vidcap = cv2.VideoCapture(fp_movie)



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
modulations['modulations_noise'] = modulations_noise

    
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

prev_diffusion_output = None
pb.num_inference_steps = 1

t_last = time.time()

sound_feature_names = ['DJ', 'SCRATCH', 'GLASS']

# av_router.map_av('SUB', 'b0_samp')
av_router.map_av('SCRATCH', 'e*_emb')
# av_router.map_av('SUB', 'd*_emb')
av_router.map_av('DJ', 'progress')
# av_router.map_av('SUB', 'fract_decoder_emb')
av_router.map_av('GLASS', 'd0_samp')

if use_image2image:
    noise_img2img = torch.randn((1,4,pb.h,pb.w)).half().cuda() * 0
    fp_image_init = '/home/lugo/Downloads/forest.png'
    image_init = cv2.imread(fp_image_init)[:,:,::-1]
    image_init = cv2.resize(image_init, (pb.w*4, pb.h*4))

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
        # lt.dynamic_print(f"fps: {1/dt:.1f}")
        t_last = time.time()
        show_osc_visualization = meta_input.get(akai_lpd8="B0", button_mode="toggle")
        if show_osc_visualization:
            receiver.show_visualization()
        
        # update oscs
        show_osc_vals = meta_input.get(akai_midimix="C4", button_mode="toggle")
        for name in sound_feature_names:
            av_router.update_sound(f'{name}', receiver.get_last_value(f"/{name}"))
            if show_osc_vals:
                print(f'{name} {receiver.get_last_value(f"/{name}")}')
        
        # modulate osc with akai
        H0 = meta_input.get(akai_midimix="H0", akai_lpd8="G0", val_min=0, val_max=1, val_default=1)
        av_router.sound_features['DJ'] *= H0
        
        H1 = meta_input.get(akai_midimix="H1", akai_lpd8="G1",val_min=0, val_max=10, val_default=1)
        av_router.sound_features['SCRATCH'] *= H1
        
        H2 = meta_input.get(akai_midimix="H2", akai_lpd8="H0", val_min=0, val_max=10, val_default=1)
        av_router.sound_features['GLASS'] *= H2
        
        for i in range(3):
            modulations[f'e{i}_emb'] = torch.tensor(1 - av_router.get_modulation('e*_emb'), device=latents1.device)
            # modulations[f'd{i}_emb'] = torch.tensor(1 - av_router.get_modulation('d*_emb'), device=latents1.device)
            
        # modulations['d0_samp'] = torch.tensor(av_router.get_modulation('d0_samp'), device=latents1.device)
            
            
        amp = 1e-1
        
        # the line below causes memory leak
        # resample_grid = acidman.do_acid(modulations_noise['d2'][None].float().permute([1,2,0]), amp)
        # amp_mod = (resample_grid - acidman.identity_resample_grid)     
      
        # acid_fields = amp_mod[:,:,0][None][None], resample_grid
        # modulations['d2_acid'] = acid_fields
        
        
        
        
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
        
        # fract_mod = av_router.get_modulation('fract_decoder_emb')
        # embeds_mod = pb.blend_prompts(pb.embeds_current, embeds_mod_full, fract_mod)
        # modulations['d0_extra_embeds'] = embeds_mod[0]
        
        # d_fract = akai_midimix.get("A0", val_min=0.0, val_max=0.1, val_default=0)
        d_fract_noise = meta_input.get(akai_lpd8="E0", akai_midimix="A0", val_min=0.0, val_max=0.02, val_default=0)
        d_fract_embed = meta_input.get(akai_lpd8="E1", akai_midimix="A1", val_min=0.0, val_max=0.02, val_default=0)
        
        cross_attention_kwargs ={}
        cross_attention_kwargs['modulations'] = modulations        
        
        latents_mix = pb.interpolate_spherical(latents1, latents2, fract)
        pb.generate_blended_img(fract, latents_mix, cross_attention_kwargs=cross_attention_kwargs)
        
        kwargs = {}
        kwargs['guidance_scale'] = 0
        kwargs['num_inference_steps'] = pb.num_inference_steps
        kwargs['latents'] = latents_mix
        kwargs['prompt_embeds'] = pb.prompt_embeds
        kwargs['negative_prompt_embeds'] = pb.negative_prompt_embeds
        kwargs['pooled_prompt_embeds'] = pb.pooled_prompt_embeds
        kwargs['negative_pooled_prompt_embeds'] = pb.negative_pooled_prompt_embeds
        
        if len(cross_attention_kwargs) > 0:
            kwargs['cross_attention_kwargs'] = cross_attention_kwargs
            
        if use_image2image:
            success, image_init = vidcap.read()
            image_init = cv2.resize(image_init, (pb.w*4, pb.h*4))
            
            alpha_acid = meta_input.get(akai_lpd8="E0", akai_midimix="G2", val_min=0.0, val_max=1, val_default=0)

            if prev_diffusion_output is not None:
                prev_diffusion_output = np.array(prev_diffusion_output)
                prev_diffusion_output = np.roll(prev_diffusion_output,1,axis=1)
                image_init = image_init.astype(np.float32) * (1-alpha_acid) + alpha_acid*prev_diffusion_output.astype(np.float32)
                image_init = image_init.astype(np.uint8)
            
            kwargs['image'] = Image.fromarray(image_init)
            kwargs['num_inference_steps'] = 2
            kwargs['strength'] = 0.5
            kwargs['guidance_scale'] = 0.5
            kwargs['noise_img2img'] = noise_img2img
            
        img_mix = pb.pipe(**kwargs).images[0]     
        secondary_renderer.render(img_mix)
        
        # save the previous diffusion output
        prev_diffusion_output = img_mix
        
        # Inject new space
        if not use_image2image:
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
                # regular movement
                # fract_osc = 0
                fract_osc = av_router.get_modulation('progress')
                if is_noise_trans:
                    fract += d_fract_noise + fract_osc
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






            
    
    
    
    
    
    
    
    
    
    
