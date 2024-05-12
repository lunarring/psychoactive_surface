#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 15:26:28 2024

@author: lugo
"""

import sys
sys.path.append('../')
sys.path.append("../garden4")
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
import matplotlib.pyplot as plt
import colorsys
import torch.nn.functional as F
from image_processing import multi_match_gpu
from util_motive_receiver import MarkerTracker
#%% VARS
use_compiled_model = True
res_fact = 1.5
width_latents = int(96*res_fact)
height_latents = int(64*res_fact)
width_renderer = int(1024*2)
height_renderer = 512*2

size_img_tiles_hw = (120, 260)   # tile image size
nmb_rows, nmb_cols = (7,7)       # number of tiles
ip_address_osc_receiver = '10.40.49.28'
dir_embds_imgs = "embds_imgs"
show_osc_visualization = False
use_cam = False


# key keys: G3 -> F3 -> F0 -> C5 -> G1 -> G2

#%% aux func
width_latents = int(np.round(width_latents/16)*16)
height_latents = int(np.round(height_latents/16)*16)
shape_cam=(600,800) 

class MovieReaderCustom():
    r"""
    Class to read in a movie.
    """

    def __init__(self, fp_movie):
        self.load_movie(fp_movie)

    def load_movie(self, fp_movie):
        self.video_player_object = cv2.VideoCapture(fp_movie)
        self.nmb_frames = int(self.video_player_object.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps_movie = int(30)
        self.shape = [shape_cam[0], shape_cam[1], 3]
        self.shape_is_set = False        

    def get_next_frame(self, speed=2):
        success = False
        for it in range(speed):
            success, image = self.video_player_object.read()
            
        if success:
            if not self.shape_is_set:
                self.shape_is_set = True
                self.shape = image.shape
            return image
        else:
            print('MovieReaderCustom: move cycle finished, resetting to first frame')
            self.video_player_object.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return np.random.randint(0,20,self.shape).astype(np.uint8)

#%% Image processing functions. These should live somewhere else
def zoom_image_torch(input_tensor, zoom_factor):
    # Ensure the input is a 4D tensor [batch_size, channels, height, width]
    if len(input_tensor.shape) == 3:
        input_tensor = input_tensor.unsqueeze(0)

    do_permute = False
    if input_tensor.shape[-1] <= 3:
        do_permute = True
        input_tensor = input_tensor.permute(0,3,1,2)
        
    
    # Original size
    original_height, original_width = input_tensor.shape[2], input_tensor.shape[3]
    
    # Calculate new size
    new_height = int(original_height * zoom_factor)
    new_width = int(original_width * zoom_factor)
    
    # Interpolate
    zoomed_tensor = F.interpolate(input_tensor, size=(new_height, new_width), mode='bilinear', align_corners=False)
    # zoomed_tensor = F.interpolate(input_tensor, size=(new_width, new_height), mode='bilinear', align_corners=False).permute(1,0,2)
    
    # Calculate padding to match original size
    pad_height = (original_height - new_height) // 2
    pad_width = (original_width - new_width) // 2
    
    # Adjust for even dimensions to avoid negative padding
    pad_height_extra = original_height - new_height - 2*pad_height
    pad_width_extra = original_width - new_width - 2*pad_width
    
    # Pad to original size
    if zoom_factor < 1:
        zoomed_tensor = F.pad(zoomed_tensor, (pad_width, pad_width + pad_width_extra, pad_height, pad_height + pad_height_extra), 'reflect', 0)
    else:
        # For zoom_factor > 1, center crop to original dimensions
        start_row = (zoomed_tensor.shape[2] - original_height) // 2
        start_col = (zoomed_tensor.shape[3] - original_width) // 2
        zoomed_tensor = zoomed_tensor[:, :, start_row:start_row + original_height, start_col:start_col + original_width]
    
    zoomed_tensor = zoomed_tensor.squeeze(0) # Remove batch dimension before returning
    if do_permute:
        zoomed_tensor = zoomed_tensor.permute(1,2,0)  
    return zoomed_tensor

def multi_match_gpu(list_images, weights=None, simple=False, clip_max='auto', gpu=0,  is_input_tensor=False):
    """
    Match colors of images according to weights.
    """
    from scipy import linalg
    if is_input_tensor:
        list_images_gpu = [img.clone() for img in list_images]
    else:
        list_images_gpu = [torch.from_numpy(img.copy()).float().cuda(gpu) for img in list_images]
    
    if clip_max == 'auto':
        clip_max = 255 if list_images[0].max() > 16 else 1  
    
    if weights is None:
        weights = [1]*len(list_images_gpu)
    weights = np.array(weights, dtype=np.float32)/sum(weights) 
    assert len(weights) == len(list_images_gpu)
    # try:
    assert simple == False    
    def cov_colors(img):
        a, b, c = img.size()
        img_reshaped = img.view(a*b,c)
        mu = torch.mean(img_reshaped, 0, keepdim=True)
        img_reshaped -= mu
        cov = torch.mm(img_reshaped.t(), img_reshaped) / img_reshaped.shape[0]
        return cov, mu
    
    covs = np.zeros((len(list_images_gpu),3,3), dtype=np.float32)
    mus = torch.zeros((len(list_images_gpu),3)).float().cuda(gpu)
    mu_target = torch.zeros((1,1,3)).float().cuda(gpu)
    #cov_target = np.zeros((3,3), dtype=np.float32)
    for i, img in enumerate(list_images_gpu):
        cov, mu = cov_colors(img)
        mus[i,:] = mu
        covs[i,:,:]= cov.cpu().numpy()
        mu_target += mu * weights[i]
            
    cov_target = np.sum(weights.reshape(-1,1,1)*covs, 0)
    covs += np.eye(3, dtype=np.float32)*1
    
    # inversion_fail = False
    try:
        sqrtK = linalg.sqrtm(cov_target)
        assert np.isnan(sqrtK.mean()) == False
    except Exception as e:
        print(e)
        # inversion_fail = True
        sqrtK = linalg.sqrtm(cov_target + np.random.rand(3,3)*0.01)
    list_images_new = []
    for i, img in enumerate(list_images_gpu):
        
        Ms = np.real(np.matmul(sqrtK, linalg.inv(linalg.sqrtm(covs[i]))))
        Ms = torch.from_numpy(Ms).float().cuda(gpu)
        #img_new = img - mus[i]
        img_new = torch.mm(img.view([img.shape[0]*img.shape[1],3]), Ms.t())
        img_new = img_new.view([img.shape[0],img.shape[1],3]) + mu_target
        
        img_new = torch.clamp(img_new, 0, clip_max)

        assert torch.isnan(img_new).max().item() == False
        if is_input_tensor:
            list_images_new.append(img_new)
        else:
            list_images_new.append(img_new.cpu().numpy())
    return list_images_new


def rotate_hue(image, angle):
    """
    Rotate the hue of an image by a specified angle.
    
    Parameters:
    - image: An image in RGB color space.
    - angle: The angle by which to rotate the hue. Can be positive or negative.
    
    Returns:
    - The image with rotated hue in RGB color space.
    """
    # Convert the image from BGR to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Rotate the hue
    # Hue is represented in OpenCV as a value from 0 to 180 instead of 0 to 360
    # Therefore, we need to scale the angle accordingly
    hsv_image[:, :, 0] = (hsv_image[:, :, 0] + (angle / 2)) % 180
    
    # Convert back to BGR from HSV
    rotated_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    
    return rotated_image

#%%

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

def get_noise_for_modulations(shape):
    return torch.randn(shape, device=pipe_text2img.device, generator=torch.Generator(device=pipe_text2img.device).manual_seed(1)).half()


class NoodleMachine():
    
    def __init__(self):
        self.causes= {}
        self.effect2causes = {}
        self.effect2functs = {}
    
    def create_noodle(self, cause_names, effect_name, func=np.prod):
        if type(cause_names) == str:
            cause_names = [cause_names]
        assert hasattr(cause_names, '__getitem__')
        assert type(cause_names[0])==str
        if effect_name in self.effect2causes.keys():
            raise ValueError(f'effect {effect_name} already noodled!')
        self.effect2causes[effect_name] = cause_names
        self.effect2functs[effect_name] = func      
        for cause_name in cause_names:
            if cause_name not in self.causes.keys():
                self.causes[cause_name] = None
            
    def set_cause(self, cause_name, cause_value, must_exist=False):
        if cause_name in self.causes.keys():
            self.causes[cause_name] = cause_value
        elif must_exist:
            raise ValueError(f'cause {cause_name} not known')

    def get_effect(self, effect_name):
        if effect_name not in self.effect2causes.keys():
            raise ValueError(f'effect {effect_name} not known')
        cause_names = self.effect2causes[effect_name]
        cause_values = []
        for cause_name in cause_names:
            if cause_name not in self.causes.keys():
                raise ValueError(f'cause {cause_name} not known')
            elif self.causes[cause_name] is None:
                raise ValueError(f'cause {cause_name} not set')
            cause_values.append(self.causes[cause_name])
        return self.effect2functs[effect_name](cause_values)
                


class PromptHolder():
    def __init__(self, prompt_blender, size_img_tiles_hw, use_image2image=False):
        self.use_image2image = use_image2image
        self.pb = prompt_blender
        self.width_images = size_img_tiles_hw[1]
        self.height_images = size_img_tiles_hw[0]
        self.img_spaces = {}
        self.prompt_spaces = {}
        self.images = {}
        self.init_buttons()
        self.active_space_idx = 0
        self.dir_embds_imgs = "embds_imgs"
        self.negative_prompt = "blurry, lowres, disfigured"
        if not os.path.exists(dir_embds_imgs):
            os.makedirs(dir_embds_imgs)
        self.init_prompts()
        self.set_next_space()
        self.active_space = list(self.prompt_spaces.keys())[self.active_space_idx]
        self.set_random_space()
        self.show_all_spaces = False
        self.list_spaces = list(self.prompt_spaces.keys())
        self.list_spaces.sort()
        
        
    def init_buttons(self):
        img_black = Image.new('RGB', (self.width_images, self.height_images), (0, 0, 0))
        self.button_redraw = lt.add_text_to_image(img_black, "redraw", font_color=(255, 255, 255))
        img_black = Image.new('RGB', (self.width_images, self.height_images), (0, 0, 0))
        self.button_space = lt.add_text_to_image(img_black, "show spaces", font_color=(255, 255, 255))
        
        
    def set_next_space(self):
        self.active_space_idx += 1
        if self.active_space_idx >= len(self.prompt_spaces.keys()):
            self.active_space_idx = 0
        self.active_space = list(self.prompt_spaces.keys())[self.active_space_idx]
        
    def set_random_space(self):
        self.active_space_idx = np.random.randint(len(self.prompt_spaces.keys()))
        self.active_space = list(self.prompt_spaces.keys())[self.active_space_idx]
    
    def prompt2hash(self, prompt):
        hash_object = hashlib.md5(prompt.encode())
        hash_code = hash_object.hexdigest()[:6].upper()
        return hash_code
    
    def prompt2img(self, prompt):
        hash_code = self.prompt2hash(prompt)
        if hash_code in self.images.keys():
            return self.images[hash_code]
        else:
            return self.get_black_img()
        
    def get_black_img(self):
        img = Image.new('RGB', (self.width_images, self.height_images), (0, 0, 0))
        return img
        
        
    def init_prompts(self):
        print("prompt holder: init prompts and images!")

        list_prompt_txts = os.listdir("prompts/")
        list_prompt_txts = [l for l in list_prompt_txts if l.endswith(".txt")]
        for fn_prompts in list_prompt_txts:
            name_space = fn_prompts.split(".txt")[0]
            list_prompts_all = []
            try:
                with open(f"prompts/{fn_prompts}", "r", encoding="utf-8") as file: 
                    list_prompts_all = file.read().split('\n')
                list_prompts_all = [l for l in list_prompts_all if len(l) > 8]
                self.prompt_spaces[name_space] = list_prompts_all
                for prompt in tqdm(list_prompts_all, desc=f'loading space: {name_space}'):
                    img, hash_code = self.load_or_gen_image(prompt)
                    self.images[hash_code] = img
                # Init space images, just taking the last image!
                img_space= lt.add_text_to_image(img.copy(), name_space, font_color=(255, 255, 255))
                self.img_spaces[name_space] = img_space # we always just take the last one
                    
            except Exception as e:
                print(f"failed: {e}")
        
    def load_or_gen_image(self, prompt):
        hash_code = self.prompt2hash(prompt)
    
        fp_img = f"{dir_embds_imgs}/{hash_code}.jpg"
        fp_embed = f"{dir_embds_imgs}/{hash_code}.pkl"
        fp_prompt = f"{dir_embds_imgs}/{hash_code}.txt"
    
        if os.path.exists(fp_img) and os.path.exists(fp_embed) and os.path.exists(fp_prompt) :
            image = Image.open(fp_img)
            image = image.resize((self.width_images, self.height_images))
            return image, hash_code
        
        if self.use_image2image:
            return self.get_black_img(), "XXXXXX"
    
        latents = self.pb.get_latents()
        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = pb.get_prompt_embeds(prompt, self.negative_prompt)
        image = pb.generate_img(latents, prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds)
    
        embeddings = {
        "prompt_embeds": prompt_embeds.cpu(),
        "negative_prompt_embeds": negative_prompt_embeds.cpu(),
        "pooled_prompt_embeds": pooled_prompt_embeds.cpu(),
        "negative_pooled_prompt_embeds": negative_pooled_prompt_embeds.cpu()
        }
        torch.save(embeddings, fp_embed)
        image = image.resize((self.width_images, self.height_images))
        image.save(fp_img)
        with open(fp_prompt, "w", encoding="utf-8") as f:
            f.write(prompt)
        return image, hash_code
            
    
    def get_prompts_imgs_within_space(self, nmb_imgs):
        list_imgs = []
        list_prompts = []
        
        # decide if we take subsequent or random
        nmb_imgs_space = len(self.prompt_spaces[self.active_space])
        if nmb_imgs_space < nmb_imgs:
            idx_imgs = np.arange(nmb_imgs_space)
        else:
            idx_imgs = np.random.choice(nmb_imgs_space, nmb_imgs, replace=False)
            
        list_prompts.append("XXX")
        list_prompts.append("XXX")
        list_imgs.append(self.button_space)
        list_imgs.append(self.button_redraw)
        
        for j in idx_imgs:
            prompt = self.prompt_spaces[self.active_space][j]
            image =  self.prompt2img(prompt)
            
            list_prompts.append(prompt)
            list_imgs.append(image)

        return list_prompts, list_imgs 
            
    
    def get_imgs_all_spaces(self, nmb_imgs):
        list_imgs = []
        for name_space in self.list_spaces:
            list_imgs.append(self.img_spaces[name_space])
        return list_imgs 
            

def remap_fract(x, c):
    if x <= 0.5:
        return 0.5 * np.power(2 * x, 1 / (1 + c))
    else:
        return 1 - 0.5 * np.power(2 * (1 - x), 1 / (1 + c))



#%% inits
midi_input = lt.MidiInput(device_name="akai_midimix")

pipe_img2img = AutoPipelineForImage2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16", local_files_only=True)
pipe_text2img = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16", local_files_only=True)
    
pipe_text2img.to("cuda")
pipe_text2img.vae = AutoencoderTiny.from_pretrained('madebyollin/taesdxl', torch_device='cuda', torch_dtype=torch.float16)
pipe_text2img.vae = pipe_text2img.vae.cuda()
pipe_text2img.set_progress_bar_config(disable=True)

pipe_text2img.unet.forward = forward_modulated.__get__(pipe_text2img.unet, UNet2DConditionModel)
    
if use_compiled_model:
    pipe_text2img.enable_xformers_memory_efficient_attention()
    config = CompilationConfig.Default()
    config.enable_xformers = True
    config.enable_triton = True
    config.enable_cuda_graph = True
    pipe_text2img = compile(pipe_text2img, config)
    
pipe_img2img.to("cuda")
pipe_img2img.vae = AutoencoderTiny.from_pretrained('madebyollin/taesdxl', torch_device='cuda', torch_dtype=torch.float16)
pipe_img2img.vae = pipe_img2img.vae.cuda()
pipe_img2img.set_progress_bar_config(disable=True)

pipe_img2img.unet.forward = forward_modulated.__get__(pipe_img2img.unet, UNet2DConditionModel)
    
if use_compiled_model:    
    pipe_img2img.enable_xformers_memory_efficient_attention()
    config = CompilationConfig.Default()
    config.enable_xformers = True
    config.enable_triton = True
    config.enable_cuda_graph = True
    pipe_img2img = compile(pipe_img2img, config)

# acidman = u_deepacid.AcidMan(0, meta_input, None)
# acidman.init('a01')

pb = PromptBlender(pipe_text2img)
pb.w = width_latents
pb.h = height_latents
latents = pb.get_latents()
secondary_renderer = lt.Renderer(width=width_renderer, height=height_renderer, backend='opencv')


prompt_holder = PromptHolder(pb, size_img_tiles_hw)


#%% prepare prompt window
gridrenderer = lt.GridRenderer(nmb_rows, nmb_cols, size_img_tiles_hw)

list_prompts, list_imgs = prompt_holder.get_prompts_imgs_within_space(nmb_cols*nmb_rows)
gridrenderer.update(list_imgs)
   
if use_cam: 
    cam = lt.WebCam(cam_id=0, shape_hw=shape_cam)
    cam.cam.set(cv2.CAP_PROP_AUTOFOCUS, 1)

receiver = lt.OSCReceiver(ip_address_osc_receiver, port_receiver = 8003)
if show_osc_visualization:
    receiver.start_visualization(shape_hw_vis=(300, 500), nmb_cols_vis=3, nmb_rows_vis=2,backend='opencv')

speech_detector = lt.Speech2Text()
#%%# img2img moving shapes

# Initialize a reference time
reference_time = time.time()

dn_movie = 'psurf_vids'        
list_fp_movies = ['bangbang_dance_interp_mflow','blue_dancer_interp_mflow',
                  'multiskeleton_dance_interp_mflow','skeleton_dance_interp_mflow',
                  'betta_fish', 'complex_ink', 'lava_lamp', 'liquid1_slow',
                  'liquid12_cropped_slow','neon_dancer']

fp_movie = os.path.join(dn_movie, np.random.choice(list_fp_movies) + '.mp4')
movie_reader = MovieReaderCustom(fp_movie)

#%% decoder embedding mixing

list_embed_modifiers_prompts = ["dramatic", "black and white", "metallic", "color explosion", "wood", 
    "stone", "abstract", "rusty", "bright", "high contrast", "neon", "surreal",
    "minimalistic", "vintage", "futuristic",  "glossy",
    "matte finish", "psychedelic", "gritty", "ethereal", "soft focus", "glowing",
    "shadowy", "muted colors", "saturated", "dusty", "crystalline", "film noir",
    "steampunk", "monochrome",  "holographic", "textured",
    "velvet", "mirror", "blurry", "geometric", "mosaic", "oil-painted",
    "watercolor", "charcoal sketch", "pen and ink", "silhouetted",
    "thermal imaging", "acidic", "noir",
    "zen", "chaotic", "floral", "urban decay", "oceanic", "space", "cyberpunk",
    "tropical", "antique", "radiant", "ghostly", "disco",
    "medieval",  "glitch", "pop art", "frosted",
    "chiaroscuro", "apocalyptic", "heavenly", "infernal", "submerged",
    "jewel-toned", "bioluminescent", "lace", "bejeweled",
    "enamel", "tattooed", "cobwebbed", "granular", "rippled", "pixelated",
    "collage", "marbled", "fluffy", "frozen"
]

nmb_embed_modifiers = 5
selected_modifiers = random.sample(list_embed_modifiers_prompts, nmb_embed_modifiers)
selected_embeds = [pb.get_prompt_embeds(modifier) for modifier in selected_modifiers]


def compute_mixed_embed(selected_embeds, weights):
    embeds_mod_full = []
    for i in range(4):
        for j in range(len(weights)):
            if j==0:
                emb = selected_embeds[j][i] * weights[j]
            else:
                emb += selected_embeds[j][i] * weights[j]
        embeds_mod_full.append(emb)
    return embeds_mod_full

#%%#
noodle_machine = NoodleMachine()

negative_prompt = "blurry, lowres, disfigured"
space_prompt = prompt_holder.prompt_spaces[prompt_holder.active_space][0]

space_prompt = 'person outline made of of sparking and trippy stars and nebula' 
space_prompt = 'person body drawn with trippy outlines and tracer colorful lines' 

# Run space
idx_cycle = 0
pb.set_prompt1(space_prompt, negative_prompt)
pb.set_prompt2(space_prompt, negative_prompt)
latents2 = pb.get_latents()
latents1 = pb.get_latents()


modulations = {}
modulations_noise = {}
for i in range(3):
    modulations_noise[f'e{i}'] = get_noise_for_modulations(get_sample_shape_unet(f'e{i}'))
    modulations_noise[f'd{i}'] = get_noise_for_modulations(get_sample_shape_unet(f'd{i}'))
    
modulations_noise['b0'] = get_noise_for_modulations(get_sample_shape_unet('b0'))
modulations['modulations_noise'] = modulations_noise

noise_cam = np.random.randn(pb.h*8, pb.w*8, 3).astype(np.float32)*100

embeds_mod_full = pb.get_prompt_embeds("dramatic")

prev_diffusion_output = None
prev_camera_output = None
pb.num_inference_steps = 1

noise_img2img = torch.randn((1,4,pb.h,pb.w)).half().cuda() * 0

t_last = time.time()

sound_feature_names = ['DJLOW', 'DJMID', 'DJHIGH']
effect_names = ['diffusion_noise', 'mem_acid', 'hue_rot']

noodle_machine.create_noodle(['DJMID'], 'diffusion_noise_mod')
noodle_machine.create_noodle(['DJLOW'], 'mem_acid_mod')
noodle_machine.create_noodle(['DJHIGH'], 'hue_rot_mod')
# noodle_machine.create_noodle(['DJMID', 'H1', 'G1'], 'diffusion_noise', lambda x: 255*x[0]*x[1]+x[2])
# noodle_machine.create_noodle(['DJLOW','H2','G2'], 'alpha_acid', lambda x: 10*x[0]*x[1]+x[2])
# noodle_machine.create_noodle(['DJHIGH','H0'], 'hue_rot_mix', lambda x: 100*x[0]*x[1])
# noodle_machine.create_noodle(['/test'], 'osc_zoom')

motive = MarkerTracker('192.168.50.64')
time.sleep(0.5)
init_drawing = True


xm = motive.get_last()['labeled_markers']
coord_array = np.array(list(xm.values()))

color_vec = torch.rand(coord_array.shape[0],3).cuda()*100



idx_embed_mod = 0
embeds_mod_full = pb.get_prompt_embeds(list_embed_modifiers_prompts[idx_embed_mod])

# embeds_mod_full = embeds_mod_full1
fract_noise = 0
fract_prompt = 0

latents1 = pb.get_latents()
latents2 = pb.get_latents()
coords = np.zeros(3)

while True:
    if fract_noise >= 1:
        # cycle back from target to source
        fract_noise = 0
        latents1 = latents2.clone()
        latents2 = pb.get_latents() 
        
    if fract_prompt >= 1:
        if do_auto_change:
            # go to random img
            space_prompt = random.choice(list_prompts[2:]) #because of two buttons 
            print(f"auto change to: {space_prompt}")
        
        fract_prompt = 0
        pb.embeds1 = pb.embeds2
        pb.set_prompt2(space_prompt, negative_prompt)
    
    # fract = 0

    # while fract < 1:
    dt = time.time() - t_last
    t_last = time.time()
    # show_osc_visualization = meta_input.get(akai_lpd8="B0", button_mode="toggle")
    # if show_osc_visualization:
    #     receiver.show_visualization()
        
    use_image2image = midi_input.get("G3", button_mode="toggle")
    
    # print(f'receiver messages: {receiver.dict_messages}')
    # update oscs
    show_osc_vals = midi_input.get("C4", button_mode="toggle")
    for name in sound_feature_names:
        noodle_machine.set_cause(f'{name}', receiver.get_last_value(f"/{name}"))
        if show_osc_vals:
            print(f'{name} {receiver.get_last_value(f"/{name}")}')
            lt.dynamic_print(f"fps: {1/dt:.1f}")
    
    fract_emb = midi_input.get("B0", val_min=0, val_max=1, val_default=0)
    if fract_emb > 0:
        modulations['b0_emb'] = torch.tensor(1 - fract_emb, device=latents1.device)        
        for i in range(3):
            modulations[f'd{i}_emb'] = torch.tensor(1 - fract_emb, device=latents1.device)        
    
    
    enable_embed_mod = midi_input.get("A3", button_mode="toggle")
    if enable_embed_mod:
        weights_emb = []
        for k in range(nmb_embed_modifiers):
            weights_emb.append(midi_input.get(f"{chr(65 + k)}5", val_min=0, val_max=1, val_default=0, variable_name = f"embed {k}"))
        
        embeds_mod_full = compute_mixed_embed(selected_embeds, weights_emb)
        total_weight = np.sum(np.asarray(weights_emb))
        embeds_mod = pb.blend_prompts(pb.embeds_current, embeds_mod_full, total_weight)
        if total_weight > 0:
            modulations['d0_extra_embeds'] = embeds_mod[0]
    else:
        if 'd0_extra_embeds' in modulations:
            del modulations['d0_extra_embeds']
        
    
    # d_fract = akai_midimix.get("A0", val_min=0.0, val_max=0.1, val_default=0)
    d_fract_noise = midi_input.get("A0", val_min=0.0, val_max=0.1, val_default=0)
    d_fract_prompt = midi_input.get("A1", val_min=0.0, val_max=0.1, val_default=0)
    
    cross_attention_kwargs ={}
    cross_attention_kwargs['modulations'] = modulations        
    
    latents_mix = pb.interpolate_spherical(latents1, latents2, fract_noise)
    fract_prompt_nonlinearity = midi_input.get("A2", val_min=0.0, val_max=3, val_default=0)
    pb.blend_stored_embeddings(remap_fract(fract_prompt, fract_prompt_nonlinearity))
    
    kwargs = {}
    kwargs['guidance_scale'] = 0.0
    kwargs['latents'] = latents_mix
    kwargs['prompt_embeds'] = pb.prompt_embeds
    kwargs['negative_prompt_embeds'] = pb.negative_prompt_embeds
    kwargs['pooled_prompt_embeds'] = pb.pooled_prompt_embeds
    kwargs['negative_pooled_prompt_embeds'] = pb.negative_pooled_prompt_embeds
    kwargs['strength'] = 0.5
    kwargs['noise_img2img'] = noise_img2img
    
    if len(cross_attention_kwargs) > 0:
        kwargs['cross_attention_kwargs'] = cross_attention_kwargs
        
    # img2img controls
    do_new_movie = midi_input.get("F3", button_mode="released_once")
    use_capture_dev = midi_input.get("G4", button_mode="toggle")
    # use_cam = midi_input.get("H4", button_mode="toggle")
    # if use_cam:
    #     try:
    #         cam = cam
    #     except:
    #         cam = lt.WebCam(cam_id=-1, shape_hw=shape_cam)
    #         cam.cam.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            
    do_color_matching = midi_input.get("F4", button_mode="toggle")
    speed_movie = midi_input.get("B1", val_min=1, val_max=16, val_default=1)
    hue_rot_drive = int(midi_input.get("G0", val_min=0.0, val_max=255, val_default=0))
    
    image_inlay_gain = midi_input.get("F0", val_min=0.0, val_max=1, val_default=0.5)
    color_matching = midi_input.get("F2", val_min=0.0, val_max=1., val_default=0)
    zoom_factor = midi_input.get("F1", val_min=0.8, val_max=1.2, val_default=1)
    do_debug_verlay = midi_input.get("H3", button_mode="toggle")
    do_drawing = midi_input.get("E3", button_mode="toggle")
    
    diffusion_noise_base = midi_input.get("G1", val_min=0.0, val_max=1, val_default=0)
    diffusion_noise_gain = midi_input.get("H1", val_min=0.0, val_max=1, val_default=0)
    
    mem_acid_base = midi_input.get("G2", val_min=0.0, val_max=1, val_default=0)
    mem_acid_gain = midi_input.get("H2", val_min=0.0, val_max=1, val_default=0)
    
    get_new_embed_modifier = midi_input.get("B4", button_mode="released_once")
    
    if get_new_embed_modifier:
        selected_modifiers = random.sample(list_embed_modifiers_prompts, nmb_embed_modifiers)
        selected_embeds = [pb.get_prompt_embeds(modifier) for modifier in selected_modifiers]
        # idx_embed_mod += 1
        # if idx_embed_mod == len(list_embed_modifiers_prompts):
        #     idx_embed_mod = 0
        # prompt_embed_modifier = list_embed_modifiers_prompts[idx_embed_mod]
        # embeds_mod_full = pb.get_prompt_embeds(prompt_embed_modifier)
        # print(f"new embed modifier: {prompt_embed_modifier} idx {idx_embed_mod}")
    
    mask_radius = midi_input.get("E2", val_min=0, val_max=10, val_default=2)
    decay_rate = midi_input.get("E0", val_min=0.0, val_max=1, val_default=0.9)
    coord_scale = midi_input.get("E1", val_min=0.0, val_max=100, val_default=75)
    drawing_intensity = midi_input.get("D2", val_min=1, val_max=15, val_default=10)
    
    if use_image2image:
        kwargs['num_inference_steps'] = 2
        
        if do_new_movie:
            fp_movie = os.path.join(dn_movie, np.random.choice(list_fp_movies) + '.mp4')
            print(f'switching movie to {fp_movie}')
            movie_reader.load_movie(fp_movie)
        
        if use_capture_dev:
            try:
                img_drive = cam.get_img()
            except Exception as e:
                print("capture card fail!")
        elif do_drawing:
            if init_drawing:
                sz_drawing = [300,600,3]
                
                canvas = torch.zeros(sz_drawing, device=latents.device)
                noise_patch = torch.rand(sz_drawing, device=latents.device)
                # draw_pos_y = canvas.shape[0]//2
                # draw_pos_x = canvas.shape[1]//2
                init_drawing = False
    
            def draw_circular_patch(Y,X,y,x, brush_size):
                # Calculate the distance from the center (x, y)
                distance = ((X - x) ** 2 + (Y - y) ** 2).float().sqrt()
                
                mask = distance > brush_size
                patch = brush_size - distance
                patch[mask] = 0
                # distance = 1 / (distance + 1e-3)
                # distance[distance < brush_size] = 0
                
                return patch                

            height, width = canvas.shape[:2]
            
            # 64 ms for 39 markers
            # coordinate test run
            
            
            xm = motive.get_last()['labeled_markers']
            coord_array = np.array(list(xm.values()))

            # rigid body positions
            # xm = motive.get_last()['rigid_bodies']
            # coord_array = np.array([v['position'] for v in xm.values()])
            
            
            # coord_scale = midi_input.get("E1", val_min=0.0, val_max=100, val_default=75)
            coord_offset = np.array([0,3,3])[None]
            
            if len(coord_array) > 0:
            # invert Y axis
                coord_array[:,1] = -coord_array[:,1]
                coord_array += coord_offset
                coord_array[coord_array < 0] = 0
                
                coord_array *= coord_scale
                coord_array = coord_array.astype(np.int32)
                
                coord_array[:,1][coord_array[:,1] >= sz_drawing[0]] = sz_drawing[0] - 1
                coord_array[:,2][coord_array[:,2] >= sz_drawing[1]] = sz_drawing[1] - 1
                
                # Mask parameters
                # mask_radius = 5      # Radius of the circle is 25 pixels for a 50 pixels diameter
                # constant_hue = midi_input.get("E0", val_min=0.0, val_max=1, val_default=0)
                
                # mask_radius = midi_input.get("E2", val_min=0, val_max=10, val_default=2)
    
                # erase
                # canvas[:] = 0
                
                # decay canvas
                # decay_rate = midi_input.get("E0", val_min=0.0, val_max=1, val_default=0.9)
                canvas = canvas * decay_rate
                # canvas[:] = 0
                
                # Create a grid of coordinates
                Y, X = torch.meshgrid(torch.arange(height, device='cuda'), torch.arange(width, device='cuda'), indexing='ij')                
                X = X.float()
                Y = Y.float()
                
                # color_vec = torch.zeros((1,1,3), device=latents.device)
                # color_vec[0,0,:] = torch.rand(3).cuda()*10
                # print(f'color_vec {color_vec}')
                
                for idx, coord in enumerate(coord_array):
                    # marker drop
                    if idx < color_vec.shape[0]:
                        colors = draw_circular_patch(Y,X,coord[1], coord[2],mask_radius).unsqueeze(2)*noise_patch*color_vec[idx][None][None]
        
                    # Add the color gradient to the image
                    canvas += colors * drawing_intensity
                    
                canvas = torch.roll(canvas, 2, dims=[0]1)
            else:
                print('cant see markers')

            # Ensure values remain within the 0-255 range after addition
            img_drive = canvas.cpu().numpy()
            img_drive = np.clip(img_drive, 0, 255)
            img_drive = img_drive.astype(np.uint8)
                
        else:
            # speed_movie += int(av_router.get_modulation('acid'))
            # print(f'speedmovie {speed_movie} {av_router.get_modulation("acid")}')
            
            img_drive = movie_reader.get_next_frame(speed=int(speed_movie))
            img_drive = np.flip(img_drive, axis=2)
            
        
        if hue_rot_drive > 0:
            img_drive = rotate_hue(img_drive, hue_rot_drive)

        image_init = cv2.resize(img_drive, (pb.w*8, pb.h*8))
        
        # print(av_router.get_modulation('diffusion_noise'), noodle_machine.get_effect('diffusion_noise'))
        # cam_noise_coef += av_router.get_modulation('diffusion_noise') * 255 XXX
        # diffusion_noise = noodle_machine.get_effect('diffusion_noise')
        diffusion_noise_mod = noodle_machine.get_effect('diffusion_noise_mod')
        # diffusion_noise_base = midi_input.get("G1", val_min=0.0, val_max=1, val_default=0)
        # diffusion_noise_gain = midi_input.get("H1", val_min=0.0, val_max=1, val_default=0)
        
        diffusion_noise = 1 * (diffusion_noise_base + diffusion_noise_gain*diffusion_noise_mod)
        
        image_init = image_init.astype(np.float32)
        image_init *= image_inlay_gain
        
        image_init = image_init + diffusion_noise*noise_cam
        image_init = np.clip(image_init, 0, 255)
        image_init = image_init.astype(np.uint8)
        
        mem_acid_mod = noodle_machine.get_effect('mem_acid_mod')
        
        mem_acid = mem_acid_base + mem_acid_gain * mem_acid_mod

        if prev_diffusion_output is not None:
            prev_diffusion_output = np.array(prev_diffusion_output)
            prev_diffusion_output = np.roll(prev_diffusion_output, 2, axis=0)
            if zoom_factor != 1:
                prev_diffusion_output = torch.from_numpy(prev_diffusion_output).to(pipe_img2img.device)
                prev_diffusion_output = zoom_image_torch(prev_diffusion_output, zoom_factor)
                prev_diffusion_output = prev_diffusion_output.cpu().numpy()
            
            image_init = image_init.astype(np.float32) * (1-mem_acid) + mem_acid*prev_diffusion_output.astype(np.float32)
            image_init = image_init.astype(np.uint8)
            
            if do_color_matching:
                image_init_torch = torch.Tensor(image_init).cuda()
                prev_diffusion_output_torch = torch.Tensor(prev_diffusion_output).cuda()
                image_init_torch_matched, _ = multi_match_gpu([image_init_torch, prev_diffusion_output_torch], weights=[1-color_matching, color_matching], simple=False, clip_max=255, gpu=0,  is_input_tensor=True)
                image_init = image_init_torch_matched.cpu().numpy().astype(np.uint8)
        
        kwargs['image'] = Image.fromarray(image_init)
        
        img_mix = pipe_img2img(**kwargs).images[0]
    else:
        kwargs['num_inference_steps'] = 1
        img_mix = pipe_text2img(**kwargs).images[0]
        img_noise_drive = np.asarray(img_mix.copy())
        

    # save the previous diffusion output
    img_mix = np.array(img_mix)
    prev_diffusion_output = img_mix.astype(np.float32)
    
    hue_rot_mod = noodle_machine.get_effect('hue_rot_mod')
    hue_rot_gain = midi_input.get("H0", val_min=0, val_max=1, val_default=0)
    
    hue_rot = 100 * hue_rot_gain * hue_rot_mod
    
    # img_mix = rotate_hue(img_mix, hue_rot)
    
    if do_debug_verlay and use_image2image:
        secondary_renderer.render(img_drive)
    else:
        secondary_renderer.render(img_mix)
    
    # Handle clicks in gridrenderer
    m,n = gridrenderer.render()
    if m != -1 and n != -1:
        try:
            idx = m*nmb_cols + n
            print(f'tile index {idx}: m {m} n {n}')
            
            if not prompt_holder.show_all_spaces:
                if idx == 0:
                    # move into space view
                    print(f'SHOWING ALL SPACES')
                    list_imgs = prompt_holder.get_imgs_all_spaces(nmb_cols*nmb_rows)
                    gridrenderer.update(list_imgs)
                    prompt_holder.show_all_spaces = True
                elif idx == 1:
                    print(f'REDRAWING IMAGES FROM SPCACE')
                    list_prompts, list_imgs = prompt_holder.get_prompts_imgs_within_space(nmb_cols*nmb_rows)
                    gridrenderer.update(list_imgs)
                else:
                    # recycle old current embeddings and latents
                    pb.embeds1 = pb.blend_prompts(pb.embeds1, pb.embeds2, fract_prompt)
                    latents1 = pb.interpolate_spherical(latents1, latents2, fract_noise)
                    space_prompt = list_prompts[idx]
                    fract_noise = 0
                    fract_prompt = 0
                    pb.set_prompt2(space_prompt, negative_prompt)
            else:
                # space selection
                prompt_holder.active_space = prompt_holder.list_spaces[idx]
                print(f"new activate space: {prompt_holder.active_space}") 
                list_prompts, list_imgs = prompt_holder.get_prompts_imgs_within_space(nmb_cols*nmb_rows)
                gridrenderer.update(list_imgs)
                prompt_holder.show_all_spaces = False
                
        except Exception as e:
            print(f"fail of click event! {e}")
        
    fract_osc = 0
    # fract_osc = av_router.get_modulation('diffusion_noise') # XXX
    fract_noise += d_fract_noise + fract_osc
    fract_prompt += d_fract_prompt + fract_osc
            
    do_new_space = midi_input.get("B3", button_mode="released_once") 
    if do_new_space:     
        prompt_holder.set_next_space()
        list_prompts, list_imgs = prompt_holder.get_prompts_imgs_within_space(nmb_cols*nmb_rows)
        gridrenderer.update(list_imgs)
        
    do_auto_change = midi_input.get("A4", button_mode="toggle")
    





    
    
    
