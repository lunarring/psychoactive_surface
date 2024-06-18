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
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image
from diffusers.models import UNet2DConditionModel
from diffusers import AutoencoderTiny
import torch
import torchvision
from prompt_blender import PromptBlender
from sfast.compilers.diffusion_pipeline_compiler import (compile, CompilationConfig)
import time
from u_unet_modulated import forward_modulated
from PIL import Image
import os
import cv2
from image_processing import multi_match_gpu
from util_motive_receiver import MarkerTracker, RigidBody
import torchvision.transforms as T
import util
import util_noodle

"""
REFACTOR OPS
- banish aux functions away
- prompt folder handling
- load a scene (embed mods)
- embeds mod motchie!
"""
#%% VARS
do_compile = True
res_fact = 1.2
width_latents = int(96*res_fact)
height_latents = int(64*res_fact)
width_renderer = 1920
height_renderer = 1080
height_renderer = 1200

size_img_tiles_hw = (120, 260)   # tile image size
nmb_rows, nmb_cols = (7,7)       # number of tiles
ip_address_osc_receiver = '192.168.50.238'
ip_address_osc_sender = '192.168.50.42' # this name is a bit confusing
show_osc_visualization = False
use_cam = False
do_fullscreen = False
do_raw_kids_drawing = False
do_acid_plane_transforms_by_tracking = True

# key keys: G3 -> F3 -> F0 -> C5 -> G1 -> G2

# AUTO VARS

width_latents = int(np.round(width_latents/16)*16)
height_latents = int(np.round(height_latents/16)*16)
shape_cam=(600,800) 
dir_prompts = "prompts_all"


#%% aux func


# Generate some sample data
np.random.seed(42)
X = np.random.rand(100, 3)  # 100 points in 2D


#%% INITS
midi_input = lt.MidiInput(device_name="akai_midimix")

pipe_img2img = AutoPipelineForImage2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16", local_files_only=True)
pipe_text2img = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16", local_files_only=True)
    
pipe_text2img.to("cuda")
pipe_text2img.vae = AutoencoderTiny.from_pretrained('madebyollin/taesdxl', torch_device='cuda', torch_dtype=torch.float16)
pipe_text2img.vae = pipe_text2img.vae.cuda()
pipe_text2img.set_progress_bar_config(disable=True)

pipe_text2img.unet.forward = forward_modulated.__get__(pipe_text2img.unet, UNet2DConditionModel)
    
if do_compile:
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
    
if do_compile:    
    pipe_img2img.enable_xformers_memory_efficient_attention()
    config = CompilationConfig.Default()
    config.enable_xformers = True
    config.enable_triton = True
    config.enable_cuda_graph = True
    pipe_img2img = compile(pipe_img2img, config)

pb = PromptBlender(pipe_text2img)
pb.w = width_latents
pb.h = height_latents
latents = pb.get_latents()
secondary_renderer = lt.Renderer(width=width_renderer, height=height_renderer, 
                                 backend='opencv', do_fullscreen=True)

# motive = MarkerTracker('192.168.50.64', process_list=["unlabeled_markers", "velocities", "rigid_bodies", "labeled_markers"])
# motive = MarkerTracker('192.168.50.64', process_list=["unlabeled_markers", "rigid_bodies", "labeled_markers"])
motive = MarkerTracker('10.40.48.84', process_list=["unlabeled_markers", "velocities", "labeled_markers", "rigid_bodies"])


prompt_holder = util.PromptHolder(pb, size_img_tiles_hw, dir_prompts=dir_prompts)

underlay_image = cv2.imread('/home/lugo/Documents/forest.jpg')
underlay_image = np.flip(underlay_image, axis=2)
underlay_image = cv2.resize(underlay_image, (600,300))

blur_kernel = torchvision.transforms.GaussianBlur(11, sigma=(7, 7))
blur_kernel_noise = torchvision.transforms.GaussianBlur(5, sigma=(3, 3))

#% prepare prompt window
gridrenderer = lt.GridRenderer(nmb_rows, nmb_cols, size_img_tiles_hw)

list_prompts, list_imgs = prompt_holder.get_prompts_imgs_within_space(nmb_cols*nmb_rows)
gridrenderer.update(list_imgs)
   
if use_cam: 
    cam = lt.WebCam(cam_id=0, shape_hw=shape_cam)
    cam.cam.set(cv2.CAP_PROP_AUTOFOCUS, 1)

receiver = lt.OSCReceiver(ip_address_osc_receiver)
sender = lt.OSCSender(ip_address_osc_sender)
if show_osc_visualization:
    receiver.start_visualization(shape_hw_vis=(300, 500), nmb_cols_vis=3, nmb_rows_vis=2,backend='opencv')

speech_detector = lt.Speech2Text()
#%%# img2img moving shapes

# Initialize a reference time
reference_time = time.time()

dn_movie = 'psurf_vids'        
list_fn_movies_all = ['bangbang_dance_interp_mflow','blue_dancer_interp_mflow',
                  'multiskeleton_dance_interp_mflow','skeleton_dance_interp_mflow',
                  'betta_fish', 'complex_ink', 'lava_lamp', 'liquid1_slow',
                  'liquid12_cropped_slow','neon_dancer']
list_fn_movies = list_fn_movies_all[:]
# list_fn_movies = ['abstract_liquid', 'betta_fish']
fp_movie = os.path.join(dn_movie, np.random.choice(list_fn_movies) + '.mp4')
movie_reader = util.MovieReaderCustom(fp_movie, shape_cam)

#%% decoder embedding mixing

# list_embed_modifiers_prompts = ["dramatic", "black and white", "metallic", "color explosion", "wood", 
#     "stone", "abstract", "rusty", "bright", "high contrast", "neon", "surreal",
#     "minimalistic", "vintage", "futuristic",  "glossy",
#     "matte finish", "psychedelic", "gritty", "ethereal", "soft focus", "glowing",
#     "shadowy", "muted colors", "saturated", "dusty", "crystalline", "film noir",
#     "steampunk", "monochrome",  "holographic", "textured",
#     "velvet", "mirror", "blurry", "geometric", "mosaic", "oil-painted",
#     "watercolor", "charcoal sketch", "pen and ink", "silhouetted",
#     "thermal imaging", "acidic", "noir",
#     "zen", "chaotic", "floral", "urban decay", "oceanic", "space", "cyberpunk",
#     "tropical", "antique", "radiant", "ghostly", "disco",
#     "medieval",  "glitch", "pop art", "frosted",
#     "chiaroscuro", "apocalyptic", "heavenly", "infernal", "submerged",
#     "jewel-toned", "bioluminescent", "lace", "bejeweled",
#     "enamel", "tattooed", "cobwebbed", "granular", "rippled", "pixelated",
#     "collage", "marbled", "fluffy", "frozen"
# ]

# list_embed_modifiers_prompts = ["rusty", "corroded", "dystopian", "cyberpunk", "eerie", "gothic", 
#     "sinister", "abandoned", "shadowy", "grim", "blood-stained", "mechanical", "grimy", "haunting",
#     "ominous", "distorted", "metallic", "dark", "gloomy", "vintage", "broken", "desolate",
#     "violent", "war-torn", "industrial", "decayed", "creepy", "haunted", "murky", "grungy",
#     "steampunk", "chaotic", "noir", "macabre", "bleak", "glitched", "disheveled", "barbed",
#     "rundown", "twisted", "gruesome", "cluttered", "derelict", "bio-mechanical", "ominous glow",
#     "alien", "jagged", "deformed", "demonic", "steely", "dark fantasy", "destructive",
#     "overgrown", "blood-soaked", "rusted chains", "gore", "punk", "menacing", "horrifying",
#     "monstrous", "nightmarish", "bleak future", "warped", "ironclad", "horror",
#     "cybernetic", "cold", "frightening", "dismal", "scorched", "devastated",
#     "clawed", "mutant", "explosive", "barbaric", "intimidating", "toxic",
#     "iron", "battered", "chained", "brutal", "worn-out", "piercing", "vengeful",
#     "charred", "corrosive", "warrior", "destructive", "spiked", "fearsome"
# ]

list_embed_modifiers_prompts = [
    "hazy", "diffuse", "surreal", "colorful", "gradients", "rich tones", "delicate shading",
    "halo-effect", "radiating", "dream-like", "background aura", "morphing periodontal",
    "gingiva", "charcoal water", "liquid", "smudge", "dramatic", "coal", "bronze metal",
    "evaporation", "steam", "shadows", "face", "transparent ropes", "glossy", "distorted body",
    "strange", "weird", "disfigured", "reflection", "crystal", "broken glass", "mirror-like",
    "neon entangled ropes", "fractured", "bright", "high-contrast", "sunrays", "otherworldly shapes",
    "organic", "tongue", "trans-lucid", "morning light", "double-exposure", "sharp colors", 
    "hazy", "glowing hue", "sunlight", "ethereal", "phantasmagoric", "psychedelic", "vivid",
    "kaleidoscopic", "luminous", "opalescent", "translucent", "glittering", "shimmering",
    "whimsical", "bizarre", "outlandish", "alien", "extraterrestrial", "chimeric", "hallucinatory",
    "phantasmal", "incandescent", "brilliant", "spectral", "phantom-like", "diaphanous",
    "gossamer", "lucent", "irradiated", "phantasm", "hypnotic", "mesmerizing", "saturated",
    "blazing", "coruscating", "radiant", "resplendent", "prismatic", "lurid", "nebulous",
    "misty", "opalescent", "dappled", "effulgent", "lambent", "refracted", "supernatural",
    "polychromatic", "incandescent", "bioluminescent", "chiaroscuro", "tenebrous", "obscure",
    "ethereal", "dystopian", "psychoactive", "transcendent", "extravagant", "baroque", "grotesque",
    "glittering", "holographic", "nebular", "chromatic", "unearthly", "celestial", "astral",
    "transdimensional", "cosmic", "stellar", "luminary", "blinding", "fiery", "opulent",
    "rich", "flamboyant", "ornate", "elaborate", "lavish", "grandiose", "luxuriant",
    "sumptuous", "extravagant", "rococo", "fantastical", "dreamscape", "nightmarish", "delirious",
    "over-saturated", "blinding", "glaring", "blazing", "blinding light", "corona", "aura",
    "wraith-like", "phantom-like", "gothic", "eldritch", "arcane", "mystical", "enchanting",
    "bewitching", "spellbinding", "entrancing", "surrealistic", "hallucination", "nightmarish",
    "eldritch horror", "supernatural glow", "luminescent", "eerie", "hauntingly beautiful",
    "shadowy", "lurid glow", "neon-lit", "otherworldly light", "sublime", "enigmatic",
    "fascinating", "mysterious", "uncanny", "cryptic", "arcane", "runic", "hieroglyphic",
    "symbolic", "esoteric", "hermetic", "translucent", "transmorphic", "metamorphic", 
    "shape-shifting", "mutative", "fluctuating", "amorphous", "polymorphic", "kaleidoscopic",
    "psychedelic", "lysergic", "acidy", "trippy", "morphing", "shifting", "protean",
    "multifarious", "complex", "labyrinthine", "serpentine", "convoluted", "twisting", 
    "curving", "spiraling", "winding", "coiling", "serpentine", "snaking", "undulating",
    "flowing", "liquid", "fluid", "molten", "viscous", "syrupy", "gelatinous", "slippery",
    "oily", "greasy", "slick", "glossy", "shiny", "polished", "sleek", "smooth", "satin",
    "silken", "velvety", "lush", "plush", "voluptuous", "richly textured", "opulently colored",
    "deep hues", "vibrant shades", "brilliant tones", "radiant colors", "luminous shades",
    "incandescent hues", "glowing tones", "brilliant lights", "shining beams", "glistening rays",
    "radiating warmth", "sizzling energy", "scintillating flashes", "flickering luminescence",
    "shimmering glow", "blazing brilliance", "flashing illuminations", "sparkling glints",
    "dazzling gleams", "brilliant scintillations", "twinkling stars", "shimmering starlight",
    "glistening dewdrops", "shining orbs", "glowing spheres", "radiant halos", "bright coronas",
    "dazzling auras", "blinding glares", "burning flares", "fiery sparks", "radiant bursts",
    "brilliant explosions", "blazing effulgence", "incandescent luminescence", "radiant brilliance",
    "resplendent flashes", "flaring gleams", "dazzling illuminations", "shimmering brilliance"
]



nmb_embed_modifiers = 8
selected_modifiers = random.sample(list_embed_modifiers_prompts, nmb_embed_modifiers)
selected_embeds = [pb.get_prompt_embeds(modifier) for modifier in selected_modifiers]


#%%#

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
    modulations_noise[f'e{i}'] = util.get_noise_for_modulations(util.get_sample_shape_unet(f'e{i}', height_latents, width_latents), pipe_text2img)
    modulations_noise[f'd{i}'] = util.get_noise_for_modulations(util.get_sample_shape_unet(f'd{i}', height_latents, width_latents), pipe_text2img)
    
modulations_noise['b0'] = util.get_noise_for_modulations(util.get_sample_shape_unet('b0', height_latents, width_latents), pipe_text2img)
modulations['modulations_noise'] = modulations_noise

noise_cam = np.random.randn(pb.h*8, pb.w*8, 3).astype(np.float32)*100

embeds_mod_full = pb.get_prompt_embeds("dramatic")

prev_diffusion_output = None
prev_camera_output = None
pb.num_inference_steps = 1

noise_img2img = torch.randn((1,4,pb.h,pb.w)).half().cuda() * 0

t_last = time.time()

#%% noodling zone

noodle_machine = util_noodle.NoodleMachine()
sound_feature_names = ['crash', 'hihat', 'kick', 'snare']
# sound_feature_names = ['osc1', 'osc2', 'osc3', 'osc4', 'osc5']


# effect_names = ['diffusion_noise', 'mem_acid', 'hue_rot', 'zoom_factor'] # UNUSED
# motion_feature_names = ['total_kinetic_energy', 'total_absolute_momentum', 'total_angular_momentum'] # UNUSED
# body_feature_names = ['left_hand_y', 'right_hand_y'] # UNUSED

## sound coupling
# noodle_machine.create_noodle('hihat', 'd_fract_noise_mod', init_value=1)
# noodle_machine.create_noodle('crash', 'sound_embed_mod_1', init_value=1)
# noodle_machine.create_noodle('xxxxhihat', 'sound_embed_mod_2', init_value=1)
# noodle_machine.create_noodle('kick', 'sound_embed_mod_3', init_value=1)
# noodle_machine.create_noodle('snare', 'sound_embed_mod_4', init_value=1)
# noodle_machine.create_noodle('osc5', 'sound_embed_mod_5', init_value=1)
# noodle_machine.create_noodle('osc6', 'sound_embed_mod_6', init_value=1)

# noodle_machine.create_noodle('v_left_hand', 'sound_embed_mod_left_hand', init_value=1, do_auto_scale=False)
# noodle_machine.create_noodle('v_right_hand', 'sound_embed_mod_right_hand', init_value=1, do_auto_scale=False)
# noodle_machine.create_noodle('v_mic', 'sound_embed_mod_mic', init_value=1, do_auto_scale=False)

# noodle_machine.create_noodle(['DJLOW'], 'mem_acid_mod')
# noodle_machine.create_noodle(['DJHIGH'], 'hue_rot_mod')


noodle_machine.create_noodle(['absolute_angular_momentum'], 'diffusion_noise_mod')
# noodle_machine.create_noodle(['right_hand_y'], 'd_fract_noise_mod')
# noodle_machine.create_noodle(['DJHIGH'], 'hue_rot_mod')

# noodle_machine.create_noodle('DJLOW', 'd_fract_noise_mod')
noodle_machine.create_noodle('total_kinetic_energy', 'd_fract_prompt_mod')
# noodle_machine.create_noodle('total_kinetic_energy', 'mem_acid_mod')
noodle_machine.create_noodle('total_spread', 'mem_acid_mod')




#% 
list_bodyparts = ["left_hand"]
rigid_bodies = {}
for bodypart in list_bodyparts:
    rigid_bodies[bodypart] = RigidBody(motive, bodypart)
    

# center = RigidBody(motive, "center")
# head = RigidBody(motive, "head")
# right_foot = RigidBody(motive, "right_foot")
# left_foot = RigidBody(motive, "left_foot")
# list_body_parts = [right_hand, left_hand, right_foot, left_foot, head, center]

time.sleep(0.5)
init_drawing = True

color_vec = torch.rand(2,3).cuda()*100

idx_embed_mod = 0
embeds_mod_full = pb.get_prompt_embeds(list_embed_modifiers_prompts[idx_embed_mod])

# embeds_mod_full = embeds_mod_full1
fract_noise = 0
fract_prompt = 0

latents1 = pb.get_latents()
latents2 = pb.get_latents()
coords = np.zeros(3)

do_kinematics_new = False
do_kinematics_ricardo = False

last_render_timestamp = 0
hue_rot = 0

frame_count = 0
dict_velocities = {}

for bodypart in rigid_bodies:
    noodle_machine.create_noodle(f'v_{bodypart}', f'embed_mod_{bodypart}', init_value=1, do_auto_scale=False)

while True:
    # Set noodles
    sender.send_message('\test', frame_count)
    
    if do_acid_plane_transforms_by_tracking:
        for bodypart in rigid_bodies:
            rigid_bodies[bodypart].update()
            
        # print(f"rigid_bodies['left_hand'].positions {rigid_bodies['left_hand'].positions}")
    # 
    if do_kinematics_new:    
        for bodypart in rigid_bodies:
            rigid_bodies[bodypart].update()
        
        smoothing_win = int(midi_input.get("H0", val_min=1, val_max=10, val_default=1))
        
        print(dict_velocities)
        # Process velocities
        for bodypart in rigid_bodies:
            dict_velocities[bodypart] = np.linalg.norm(np.mean(rigid_bodies[bodypart].velocities[-smoothing_win:]))
            noodle_machine.set_cause(f'v_{bodypart}', dict_velocities[bodypart])
            
    
    
    if do_kinematics_ricardo:
        right_hand.update()
        left_hand.update()
        mic.update()
        smoothing_win = int(midi_input.get("H0", val_min=1, val_max=10, val_default=1))
        try:
            
            v_right_hand = np.linalg.norm(np.mean(right_hand.velocities[-smoothing_win:]))
            v_left_hand = np.linalg.norm(np.mean(left_hand.velocities[-smoothing_win:]))
            v_mic = np.linalg.norm(np.mean(mic.velocities[-smoothing_win:]))
        except Exception as e:
            v_right_hand = 0
            v_left_hand = 0
            v_mic = 0
            print(e)
        
        # print(f"{v_right_hand} {v_left_hand} {v_mic}")
        noodle_machine.set_cause('v_right_hand', v_right_hand)
        noodle_machine.set_cause('v_left_hand', v_left_hand)
        noodle_machine.set_cause('v_mic', v_mic)

    positions = motive.positions[motive.pos_idx-1]
    # print(positions[:5])
    velocities = motive.velocities[motive.pos_idx-1]
    not_nan = ~np.isnan(positions.sum(axis=1))
    positions = positions[not_nan]
    velocities = velocities[not_nan]
    
    if False:
        # MOTIVE FOR PASTA tracking first
        # right_hand.update()
        # left_hand.update()
        # right_foot.update()
        # left_foot.update()
        # center.update()
        # head.update()
        
        # if len(right_hand.kinetic_energies) > 0:
        #     print(right_hand.kinetic_energies[-1])
        
        # pos_history_range = 10    
        # pos_idx = motive.pos_idx
        # if pos_idx < pos_history_range:
        #     position_history = np.zeros([pos_history_range,motive.positions.shape[1],3])
        #     times = np.ones(pos_history_range) / 1000
        #     # last_velocities = np.zeros([10,motive.positions.shape[1],3])
        # else:
        #     position_history = motive.positions[motive.pos_idx-pos_history_range:motive.pos_idx]
        #     not_nan = ~np.isnan(position_history.sum(axis=0).sum(axis=1))
        #     position_history = position_history[:,not_nan,:]
        #     times = np.array(motive.list_timestamps[motive.pos_idx-pos_history_range:motive.pos_idx]) / 1000
        # masses = np.ones(len(positions))
        
        if False:
            masses = 1
            # positions_l = position_history[pos_history_range//2]
            # positions_ll = position_history[0]
            positions = motive.positions[motive.pos_idx-1]
            velocities = motive.velocities[motive.pos_idx-1]
            not_nan = ~np.isnan(positions.sum(axis=1))
            positions = positions[not_nan]
            velocities = velocities[not_nan]
            
            #!! remove later
            # positions = positions[abs(velocities).sum(axis=1)>0.1]
            # velocities = velocities[abs(velocities).sum(axis=1)>0.1]
            
            # velocities = (positions - positions_l)/(times[-1] - times[pos_history_range//2])
            # velocities_l = (positions_l - positions_ll)/(times[pos_history_range//2]-times[0])
            # accelerations = (velocities - velocities_l)/(times[-1] - times[pos_history_range//2])
            # center_of_mass = np.average(positions, axis=0, weights=masses)
            center_of_mass = np.average(positions, axis=0)
            rel_positions = positions - center_of_mass
            # print(positions)
            momenta = masses * velocities
            angular_momenta = np.cross(rel_positions, momenta)
            total_angular_momentum = angular_momenta.sum(axis=0)
            absolute_angular_momentum = abs(total_angular_momentum).sum()
            # kinetic_energies = 0.5 * masses * np.linalg.norm(velocities)**2
            kinetic_energies = 0.5 * np.linalg.norm(velocities, axis=1)**2
            total_kinetic_energy = kinetic_energies.sum()
            total_kinetic_energy_sqrt = np.sqrt(total_kinetic_energy)
            total_absolute_momentum = abs(momenta).sum()
            noodle_machine.set_cause('total_absolute_momentum', total_absolute_momentum)
            # potential_energy = center_of_mass[1] * masses.sum()
            potential_energy = center_of_mass[1] #* masses.sum()
            total_spread = np.linalg.norm(rel_positions, axis=1).sum()
            noodle_machine.set_cause('total_spread', total_spread)
            # print(positions)
            # print(f'total angular momentum {total_angular_momentum}')
            
            if do_kinematics_ricardo:
                try:
                    for part in list_body_parts:
                        total_kinetic_energy += part.kinetic_energies[-1]
                    
                    right_hand_y = right_hand.positions[-1][1]
                    left_hand_y = left_hand.positions[-1][1]
                    right_hand_x = right_hand.positions[-1][0]
                    left_hand_x = left_hand.positions[-1][0]
                    right_hand_z = right_hand.positions[-1][2]
                    left_hand_z = left_hand.positions[-1][2]
                    center_velocity = np.linalg.norm(center.velocities[-1])
                except Exception as E:
                    # print(E)
                    right_hand_y = 0
                    left_hand_y = 0
                    right_hand_x = 0
                    left_hand_x = 0
                    right_hand_z = 0
                    left_hand_z = 0
                    center_velocity = 0
                    
                # print(f'total_kinetic_energy: {total_kinetic_energy}')
                noodle_machine.set_cause('right_hand_y', right_hand_y)
                noodle_machine.set_cause('total_kinetic_energy', total_kinetic_energy)
                # print(f'right_hand_y: {right_hand_y}')
                # print(f'total_kinetic_energy: {total_kinetic_energy}')
        # except:
            # print('nothing coming from tracking')
    
    
    # REST
    if fract_noise >= 1:
        # cycle back from target to source
        fract_noise = 0
        latents1 = latents2.clone()
        latents2 = pb.get_latents() 
        
    if fract_prompt >= 1:
        if do_auto_change_prompts:
            # go to random img
            space_prompt = random.choice(list_prompts[2:]) #because of two buttons 
            print(f"auto change to: {space_prompt}")
        
        fract_prompt = 0
        pb.embeds1 = pb.embeds2
        pb.set_prompt2(space_prompt, negative_prompt)
    
    # while fract < 1:
    dt = time.time() - t_last
    t_last = time.time()
    # show_osc_visualization = meta_input.get(akai_lpd8="B0", button_mode="toggle")
    
    if show_osc_visualization:
        receiver.show_visualization()
        
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
    kill_embed_weights = midi_input.get("C3", button_mode="toggle")
    max_embed_mods = midi_input.get("A2", val_min=0.3, val_max=2.5, val_default=0.8)
    if enable_embed_mod:
        
        list_amp_embed_mods = []
        for x in ["A", "B", "C", "D", "E"]:
            list_amp_embed_mods.append(midi_input.get(f"{x}5", val_min=0, val_max=max_embed_mods, val_default=0))
            # amp_embed_mod1 = midi_input.get(f"A5", val_min=0, val_max=max_embed_mods, val_default=0)
            # amp_embed_mod2 = midi_input.get(f"B5", val_min=0, val_max=max_embed_mods, val_default=0)
            # amp_embed_mod3 = midi_input.get(f"C5", val_min=0, val_max=max_embed_mods, val_default=0)
            # amp_embed_mod4 = midi_input.get(f"D5", val_min=0, val_max=max_embed_mods, val_default=0)
            # amp_embed_mod5 = midi_input.get(f"E5", val_min=0, val_max=max_embed_mods, val_default=0)
        
        # amp_embed_mod_left_hand = midi_input.get("F5", val_min=0, val_max=max_embed_mods, val_default=0)
        # amp_embed_mod_right_hand = midi_input.get("G5", val_min=0, val_max=max_embed_mods, val_default=0)
        # amp_embed_mod_mic = midi_input.get("H5", val_min=0, val_max=max_embed_mods, val_default=0)
        
        nmb_mods = min(len(list_amp_embed_mods), len(list_bodyparts))
        weights_emb = []
        ks = 10
        for i in range(nmb_mods):
            bodypart = list_bodyparts[i]
            weights_emb.append(ks*list_amp_embed_mods[i] * noodle_machine.get_effect(f'embed_mod_{bodypart}'))
            
            
            
        # weights_emb.append(amp_embed_mod1 * noodle_machine.get_effect("sound_embed_mod_1"))
        # weights_emb.append(amp_embed_mod2 * noodle_machine.get_effect("sound_embed_mod_2"))
        # weights_emb.append(amp_embed_mod3 * noodle_machine.get_effect("sound_embed_mod_3"))
        # weights_emb.append(amp_embed_mod4 * noodle_machine.get_effect("sound_embed_mod_4"))
        # weights_emb.append(amp_embed_mod5 * noodle_machine.get_effect("sound_embed_mod_5"))
        
        
        # for bodypart in rigid_bodies:
        #     noodle_machine.create_noodle(f'v_{bodypart}', 'embed_mod_{bodypart}', init_value=1, do_auto_scale=False)
            
        # weights_emb.append(ks*amp_embed_mod_left_hand * noodle_machine.get_effect("embed_mod_left_hand"))
        # weights_emb.append(ks*amp_embed_mod_right_hand * noodle_machine.get_effect("embed_mod_right_hand"))
        # weights_emb.append(ks*amp_embed_mod_mic * noodle_machine.get_effect("embed_mod_mic"))
        
        embeds_mod_full = util.compute_mixed_embed(selected_embeds, weights_emb)
        total_weight = np.sum(np.asarray(weights_emb))
        embeds_mod = pb.blend_prompts(pb.embeds_current, embeds_mod_full, total_weight)
        if total_weight > 0:
            modulations['d0_extra_embeds'] = embeds_mod[0]
        if kill_embed_weights and 'd0_extra_embeds' in modulations:
            del modulations['d0_extra_embeds']
    else:
        if 'd0_extra_embeds' in modulations:
            del modulations['d0_extra_embeds']
        
    
    d_fract_noise_gain = midi_input.get("A0", val_min=0.0, val_max=0.1, val_default=0.01)
    d_fract_noise = d_fract_noise_gain# * noodle_machine.get_effect('d_fract_noise_mod')
    
    d_fract_prompt = midi_input.get("A1", val_min=0.0, val_max=0.01, val_default=0)
    
    cross_attention_kwargs ={}
    cross_attention_kwargs['modulations'] = modulations        
    
    latents_mix = pb.interpolate_spherical(latents1, latents2, fract_noise)
    fract_prompt_nonlinearity = 1.7
    pb.blend_stored_embeddings(util.remap_fract(fract_prompt, fract_prompt_nonlinearity))
    
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
            
    do_color_matching = midi_input.get("F4", button_mode="toggle")
    speed_movie = midi_input.get("B1", val_min=2, val_max=16, val_default=2)
    hue_rot_drive = int(midi_input.get("G0", val_min=0.0, val_max=255, val_default=0))
    
    image_inlay_gain = midi_input.get("F0", val_min=0.0, val_max=1, val_default=0.95)
    color_matching = midi_input.get("F2", val_min=0.0, val_max=1., val_default=0.2)
    
    zoom_factor_base = midi_input.get("F1", val_min=0.8, val_max=1.2, val_default=0.992)
    zoom_factor = zoom_factor_base
    
    do_debug_verlay = midi_input.get("H3", button_mode="toggle")
    do_drawing = midi_input.get("E3", button_mode="toggle")
    disable_zoom = not midi_input.get("H4", button_mode="toggle")
    
    diffusion_noise_base = midi_input.get("G1", val_min=0.0, val_max=1, val_default=0.1)
    diffusion_noise_gain = midi_input.get("H1", val_min=0.0, val_max=1, val_default=0)
    
    mem_acid_base = midi_input.get("G2", val_min=0.0, val_max=1, val_default=0.15)
    mem_acid_gain = midi_input.get("H2", val_min=0.0, val_max=1, val_default=0)
    
    get_new_embed_modifier = midi_input.get("B3", button_mode="released_once")
    
    if get_new_embed_modifier:
        selected_modifiers = random.sample(list_embed_modifiers_prompts, nmb_embed_modifiers)
        print(f"get_new_embed_modifier: {selected_modifiers}")
        selected_embeds = [pb.get_prompt_embeds(modifier) for modifier in selected_modifiers]
    
    mask_radius = midi_input.get("E2", val_min=0, val_max=50, val_default=8)
    decay_rate = np.sqrt(midi_input.get("E0", val_min=0.0, val_max=1, val_default=0.62))
    coord_scale = midi_input.get("E1", val_min=0.0, val_max=600, val_default=150)
    drawing_intensity = midi_input.get("D2", val_min=0, val_max=1, val_default=0.9)
    use_underlay_image = midi_input.get("D3", button_mode="toggle")
    color_angle = midi_input.get("C2", val_min=0, val_max=7, val_default=0.55)
    if use_image2image:
        kwargs['num_inference_steps'] = 2
        
        if do_new_movie:
            if len(list_fn_movies) == 0:
                list_fn_movies = list_fn_movies_all[:]
            fn_movie = np.random.choice(list_fn_movies)
            list_fn_movies.remove(fn_movie)
            fp_movie = os.path.join(dn_movie, fn_movie + '.mp4')
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
                noise_patch = torch.rand(sz_drawing, device=latents.device) #- 0.5
                init_drawing = False
    
            height, width = canvas.shape[:2]
            
            if do_raw_kids_drawing:
                coord_array = positions.copy()            
                coord_array[:,1] = -coord_array[:,1]
                coord_array[:,0] = -coord_array[:,0]
                
                # print(f'positions {positions}')
                
                coord_offset = np.array([320,340,280])[None]
                if len(coord_array) > 0:
                    coord_array *= coord_scale
                    coord_array[:,1] *= 1.2
                    coord_array[:,0] *= 1.2
                    coord_array += coord_offset
                    
                    coord_array[coord_array < 0] = 0
                        
                    coord_array[:,0][coord_array[:,0] >= sz_drawing[1]] = sz_drawing[1] - 1
                    coord_array[:,1][coord_array[:,1] >= sz_drawing[0]] = sz_drawing[0] - 1
                    
                    # print(f'coord_array {coord_array}')
                    
                    coord_array = coord_array.astype(np.int32)
                    
                    # decay canvas
                    canvas = canvas * decay_rate
                    
                    # Create a grid of coordinates
                    Y, X = torch.meshgrid(torch.arange(height, device='cuda'), torch.arange(width, device='cuda'), indexing='ij')                
                    X = X.float()
                    Y = Y.float()
                    
                    for idx, coord in enumerate(coord_array):
                        color_angle = idx*10
                        color_vec = util.angle_to_rgb(color_angle)
                        color_vec = torch.from_numpy(np.array(color_vec)).float().cuda(canvas.device)
                        
                        mask_radius = coord[2] / 10
                        
                        patch = util.draw_circular_patch(Y,X,coord[1], coord[0], mask_radius)
                        if use_underlay_image:
                            colors = patch.unsqueeze(2)*color_vec[None][None]
                        else:
                            colors = patch.unsqueeze(2)*noise_patch*color_vec[None][None]
            
                        # Add the color gradient to the image
                        colors /= (colors.max() + 0.0001)
                        canvas += colors * drawing_intensity * 255
                        canvas = canvas.clamp(0, 255)

                    canvas_numpy = canvas.cpu().numpy()
                    canvas_numpy = np.clip(canvas_numpy, 0, 255)
                    
                    
                    if use_underlay_image:
                        canvas_numpy /= 255
                        img_drive = underlay_image * canvas_numpy
                        img_drive = img_drive.astype(np.uint8)
                    else:
                        img_drive = canvas_numpy                       
                else:
                    print('cant see markers')
                    
                 
            else:
                canvas_numpy = canvas.cpu().numpy()
                canvas_numpy = np.clip(canvas_numpy, 0, 255)
                
                
                if use_underlay_image:
                    canvas_numpy /= 255
                    img_drive = underlay_image * canvas_numpy
                    img_drive = img_drive.astype(np.uint8)
                else:
                    img_drive = canvas_numpy
                        
        else:
            img_drive = movie_reader.get_next_frame(speed=int(speed_movie))
            img_drive = np.flip(img_drive, axis=2)
            
        
        if hue_rot_drive > 0:
            img_drive = util.rotate_hue(img_drive, hue_rot_drive)

        image_init = cv2.resize(img_drive, (pb.w*8, pb.h*8))
        
        diffusion_noise_mod = noodle_machine.get_effect('diffusion_noise_mod')
        
        diffusion_noise = 1 * (diffusion_noise_base + diffusion_noise_gain*diffusion_noise_mod)
        
        image_init = image_init.astype(np.float32)
        image_init *= image_inlay_gain
        
        image_init = image_init + diffusion_noise*noise_cam
        image_init = np.clip(image_init, 0, 255)
        image_init = image_init.astype(np.uint8)
        
        mem_acid_mod = noodle_machine.get_effect('mem_acid_mod')
        
        mem_acid = mem_acid_base + mem_acid_gain * mem_acid_mod
        if mem_acid > 1:
            mem_acid = 1
            
        rotation_angle_left = midi_input.get("C0", val_min=0, val_max=90, val_default=0)
        rotation_angle_right = midi_input.get("D0", val_min=0, val_max=90, val_default=0)
        rotation_angle = rotation_angle_left - rotation_angle_right
        if rotation_angle < 0:
            rotation_angle = 360 + rotation_angle
        if prev_diffusion_output is not None:
            prev_diffusion_output = np.array(prev_diffusion_output)
            # prev_diffusion_output = np.roll(prev_diffusion_output, 1, axis=0)
            if zoom_factor != 1 and not disable_zoom:
                prev_diffusion_output = torch.from_numpy(prev_diffusion_output).to(pipe_img2img.device)
                prev_diffusion_output = util.zoom_image_torch(prev_diffusion_output, zoom_factor)
                prev_diffusion_output = prev_diffusion_output.cpu().numpy()
            
            
            if rotation_angle > 0:
                prev_diffusion_output = torch.from_numpy(prev_diffusion_output).to(pipe_img2img.device)
                padding = int(prev_diffusion_output.shape[1] // (2*np.sqrt(2)))
                padding = (padding, padding)
                prev_diffusion_output = T.Pad(padding=padding, padding_mode='reflect')(prev_diffusion_output.permute(2,0,1))
                prev_diffusion_output = T.functional.rotate(prev_diffusion_output, angle=rotation_angle, interpolation=T.functional.InterpolationMode.BILINEAR, expand=False).permute(1,2,0)
                prev_diffusion_output = prev_diffusion_output[padding[0]:prev_diffusion_output.shape[0]-padding[0],padding[1]:prev_diffusion_output.shape[1]-padding[1]]
                prev_diffusion_output = prev_diffusion_output.cpu().numpy()
                
            if do_acid_plane_transforms_by_tracking:
                # shift_pos = rigid_bodies['left_hand'].positions[-1]
                shift_pos = rigid_bodies['left_hand'].velocities[-1]
                orientation = rigid_bodies['left_hand'].orientations[-1]
                
                if type(shift_pos) != int:
                    # print(f'shift_pos {shift_pos}')
                    
                    # single rigid body mode
                    euler_angle = util.euler_from_quaternion(*orientation)
                    # print(f'euler_angle {euler_angle}')
                    
                    # amp_shift = 100 # positions
                    amp_shift = 400  # velocities
                    
                    # positions
                    # x_shift = -int(shift_pos[0]*amp_shift)
                    # y_shift = -int((shift_pos[1]-1.5)*amp_shift)
                    
                    # velocies
                    x_shift = -int(shift_pos[0]*amp_shift)
                    y_shift = -int((shift_pos[1])*amp_shift)
                    
                    rotation_angle = -2*euler_angle[2]
                    hue_rot = 50*euler_angle[1]
                    
                    zoom_factor = 1 + shift_pos[2]*2
                    if zoom_factor < 0.25:
                        zoom_factor = 0.25
                    # print(f'x_shift {x_shift} y_shift {y_shift} zoom_factor {zoom_factor}')
    
                    prev_diffusion_output = torch.from_numpy(prev_diffusion_output).to(pipe_img2img.device)
                    
                    # scale
                    try:
                        prev_diffusion_output = util.zoom_image_torch(prev_diffusion_output, zoom_factor)
                    except:
                        pass
                    
                    # rotate
                    padding = int(prev_diffusion_output.shape[1] // (2*np.sqrt(2)))
                    padding = (padding, padding)
                    prev_diffusion_output = T.Pad(padding=padding, padding_mode='reflect')(prev_diffusion_output.permute(2,0,1))
                    prev_diffusion_output = T.functional.rotate(prev_diffusion_output, angle=rotation_angle, interpolation=T.functional.InterpolationMode.BILINEAR, expand=False).permute(1,2,0)
                    prev_diffusion_output = prev_diffusion_output[padding[0]:prev_diffusion_output.shape[0]-padding[0],padding[1]:prev_diffusion_output.shape[1]-padding[1]]
                    
                    prev_diffusion_output = torch.roll(prev_diffusion_output, (y_shift, x_shift), (0,1))    
                    prev_diffusion_output = prev_diffusion_output.cpu().numpy()
            else:
                hue_rot = 0
            
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
    
    if do_acid_plane_transforms_by_tracking:
        img_mix = util.rotate_hue(img_mix, hue_rot)
    
    if do_fullscreen:
        cv2.namedWindow('lunar_render_window', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('lunar_render_window',cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    
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
    fract_noise += d_fract_noise + fract_osc
    fract_prompt += d_fract_prompt + fract_osc
            
    do_new_space = midi_input.get("B4", button_mode="released_once") 
    if do_new_space:     
        prompt_holder.set_next_space()
        list_prompts, list_imgs = prompt_holder.get_prompts_imgs_within_space(nmb_cols*nmb_rows)
        gridrenderer.update(list_imgs)
        
    do_auto_change_prompts = midi_input.get("A4", button_mode="toggle")
    frame_count += 1
    if frame_count % 1000 == 0:
        midi_input.show()

    time_difference = time.time() - last_render_timestamp
    last_render_timestamp = time.time()
    fps = np.round(1/time_difference)
    # lt.dynamic_print(f'fps: {fps}')
