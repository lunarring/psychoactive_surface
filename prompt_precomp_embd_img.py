#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
import hashlib
import os
from tqdm import tqdm

#%% VARS
use_compiled_model = False
width_latents = 128
height_latents = 64
negative_prompt = ""
dir_embds_imgs = "embds_imgs"
width_images = 256
height_images = 128



#%%
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


pb = PromptBlender(pipe)
pb.w = width_latents
pb.h = height_latents


# %%
if not os.path.exists(dir_embds_imgs):
    os.makedirs(dir_embds_imgs)

list_prompts_all = []
with open("good_prompts.txt", "r", encoding="utf-8") as file: 
    list_prompts_all = file.read().split('\n')

for prompt in tqdm(list_prompts_all):
    
    hash_object = hashlib.md5(prompt.encode())
    hash_code = hash_object.hexdigest()[:6].upper()

    fp_img = f"{dir_embds_imgs}/{hash_code}.jpg"
    fp_embed = f"{dir_embds_imgs}/{hash_code}.pkl"
    fp_prompt = f"{dir_embds_imgs}/{hash_code}.txt"

    # if os.path.exists(fp_img) and os.path.exists(fp_embed) and os.path.exists(fp_prompt) :
    #     continue

    latents = pb.get_latents()
    prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = pb.get_prompt_embeds(prompt, negative_prompt)
    image = pb.generate_img(latents, prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds)

    embeddings = {
    "prompt_embeds": prompt_embeds.cpu(),
    "negative_prompt_embeds": negative_prompt_embeds.cpu(),
    "pooled_prompt_embeds": pooled_prompt_embeds.cpu(),
    "negative_pooled_prompt_embeds": negative_pooled_prompt_embeds.cpu()
    }
    torch.save(embeddings, fp_embed)
    image = image.resize((width_images, height_images))
    image.save(fp_img)
    with open(fp_prompt, "w", encoding="utf-8") as f:
        f.write(prompt)



"""
LOADING EMBEDDINGS

with open(fp_embed, "rb") as f:
    embeddings = torch.load(f)
prompt_embeds = embeddings["prompt_embeds"].cuda()
negative_prompt_embeds = embeddings["negative_prompt_embeds"].cuda()
pooled_prompt_embeds = embeddings["pooled_prompt_embeds"].cuda()
negative_pooled_prompt_embeds = embeddings["negative_pooled_prompt_embeds"].cuda()


"""

# %%
