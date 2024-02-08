#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 22:30:17 2023

@author: lunar
"""

import torch
import numpy as np

@staticmethod
@torch.no_grad()

#%%
class PromptBlender:
    def __init__(self, pipe, gpu_id=0):
        self.pipe = pipe
        self.fract = 0
        self.first_fract = 0
        self.gpu_id = gpu_id
        self.embeds1 = None
        self.embeds2 = None
        self.num_inference_steps = 1
        self.guidance_scale = 0.0
        self.device = "cuda"
        self.tree_final_imgs = None
        self.tree_fracts = None
        self.tree_similarities = None
        self.tree_insertion_idx = None

    def load_lpips(self):
        import lpips
        self.lpips = lpips.LPIPS(net='alex').cuda(self.gpu_id)

    def get_lpips_similarity(self, imgA, imgB):
        r"""
        Computes the image similarity between two images imgA and imgB.
        Used to determine the optimal point of insertion to create smooth transitions.
        High values indicate low similarity.
        """
        tensorA = torch.from_numpy(np.asarray(imgA)).float().cuda(self.device)
        tensorA = 2 * tensorA / 255.0 - 1
        tensorA = tensorA.permute([2, 0, 1]).unsqueeze(0)
        tensorB = torch.from_numpy(np.asarray(imgB)).float().cuda(self.device)
        tensorB = 2 * tensorB / 255.0 - 1
        tensorB = tensorB.permute([2, 0, 1]).unsqueeze(0)
        lploss = self.lpips(tensorA, tensorB)
        lploss = float(lploss[0][0][0][0])
        return lploss
        
    def interpolate_spherical(self, p0, p1, fract_mixing: float):
        """
        Helper function to correctly mix two random variables using spherical interpolation.
        """
        if p0.dtype == torch.float16:
            recast_to = 'fp16'
        else:
            recast_to = 'fp32'

        p0 = p0.double()
        p1 = p1.double()
        norm = torch.linalg.norm(p0) * torch.linalg.norm(p1)
        epsilon = 1e-7
        dot = torch.sum(p0 * p1) / norm
        dot = dot.clamp(-1 + epsilon, 1 - epsilon)

        theta_0 = torch.arccos(dot)
        sin_theta_0 = torch.sin(theta_0)
        theta_t = theta_0 * fract_mixing
        s0 = torch.sin(theta_0 - theta_t) / sin_theta_0
        s1 = torch.sin(theta_t) / sin_theta_0
        interp = p0 * s0 + p1 * s1

        if recast_to == 'fp16':
            interp = interp.half()
        elif recast_to == 'fp32':
            interp = interp.float()

        return interp
        
    def set_prompt1(self, prompt, negative_prompt=""):
        self.embeds1 = self.get_prompt_embeds(prompt, negative_prompt)
    
    def set_prompt2(self, prompt, negative_prompt=""):
        self.embeds2 = self.get_prompt_embeds(prompt, negative_prompt)


    def get_prompt_embeds(self, prompt, negative_prompt=""):
        """
        Encodes a text prompt into embeddings using the model pipeline.
        """
        (
         prompt_embeds, 
         negative_prompt_embeds, 
         pooled_prompt_embeds, 
         negative_pooled_prompt_embeds
         ) = self.pipe.encode_prompt(
            prompt=prompt,
            prompt_2=prompt,
            device=f"cuda:{self.gpu_id}",
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            pooled_prompt_embeds=None,
            negative_pooled_prompt_embeds=None,
            lora_scale=0,
            clip_skip=False
        )

        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds
    
    
    def get_all_embeddings(self, prompts):
        
        prompts_embeds = []
        for prompt in prompts:
            prompts_embeds.append(self.get_prompt_embeds(prompt))
            
        self.prompts_embeds = prompts_embeds
    
        return prompts_embeds
    
    def set_init_position(self, index):
        self.current = [self.prompts_embeds[index][i] for i in range(4)]
    
    def set_target(self, index):
        self.target = [self.prompts_embeds[index][i] for i in range(4)]
    
    def step(self, pvelocity):
        for i in range(4):
            d = self.target[i] - self.current[i]
            
            d_norm = torch.linalg.norm(d)
            if d_norm > 0:
                # self.fract = pvelocity / d_norm
                # self.fract = torch.sqrt(self.fract)
                self.fract = pvelocity
                
                # self.fract = pvelocity
                if self.fract > 1:
                    self.fract = 1
            else:
                self.fract = 1
            
            self.current[i] = self.interpolate_spherical(self.current[i], self.target[i], self.fract)
            if i == 0:
                self.first_fract = self.fract

    def blend_stored_embeddings(self, fract):
        assert hasattr(self, 'embeds1'), "embeds1 not set. Please set embeds1 before blending."
        assert hasattr(self, 'embeds2'), "embeds2 not set. Please set embeds2 before blending."
        fract = max(0, min(fract, 1))
        self.prompt_embeds, self.negative_prompt_embeds, self.pooled_prompt_embeds, self.negative_pooled_prompt_embeds = self.blend_prompts(self.embeds1, self.embeds2, fract)
    


    def blend_prompts(self, embeds1, embeds2, fract):
        """
        Blends two sets of prompt embeddings based on a specified fraction.
        """
        prompt_embeds1, negative_prompt_embeds1, pooled_prompt_embeds1, negative_pooled_prompt_embeds1 = embeds1
        prompt_embeds2, negative_prompt_embeds2, pooled_prompt_embeds2, negative_pooled_prompt_embeds2 = embeds2

        blended_prompt_embeds = self.interpolate_spherical(prompt_embeds1, prompt_embeds2, fract)
        blended_negative_prompt_embeds = self.interpolate_spherical(negative_prompt_embeds1, negative_prompt_embeds2, fract)
        blended_pooled_prompt_embeds = self.interpolate_spherical(pooled_prompt_embeds1, pooled_prompt_embeds2, fract)
        blended_negative_pooled_prompt_embeds = self.interpolate_spherical(negative_pooled_prompt_embeds1, negative_pooled_prompt_embeds2, fract)

        return blended_prompt_embeds, blended_negative_prompt_embeds, blended_pooled_prompt_embeds, blended_negative_pooled_prompt_embeds

    def blend_sequence_prompts(self, prompts, n_steps):
        """
        Generates a sequence of blended prompt embeddings for a list of text prompts.
        """
        blended_prompts = []
        for i in range(len(prompts) - 1):
            prompt_embeds1 = self.get_prompt_embeds(prompts[i])
            prompt_embeds2 = self.get_prompt_embeds(prompts[i + 1])
            for step in range(n_steps):
                fract = step / float(n_steps - 1)
                blended = self.blend_prompts(prompt_embeds1, prompt_embeds2, fract)
                blended_prompts.append(blended)
        return blended_prompts

    def generate_blended_img(self, fract, latents=None):
        # Set the embeddings first with blend_stored_embeddings
        torch.manual_seed(420)
        self.blend_stored_embeddings(fract)
        # Then call the pipeline to generate the image using the embeddings set by blend_stored_embeddings
        image = self.pipe(guidance_scale=0.0, num_inference_steps=1, latents=latents, 
                        prompt_embeds=self.prompt_embeds, negative_prompt_embeds=self.negative_prompt_embeds, 
                        pooled_prompt_embeds=self.pooled_prompt_embeds, negative_pooled_prompt_embeds=self.negative_pooled_prompt_embeds).images[0]
        return image

    def init_tree(self, img_first=None, img_last=None, latents=None):
        if img_first is None:
            img_first = self.generate_blended_img(0.0, latents)
        if img_last is None:
            img_last = self.generate_blended_img(1.0, latents)
        self.tree_final_imgs = [img_first, img_last]
        self.tree_fracts = [0.0, 1.0]
        self.tree_similarities = [self.get_lpips_similarity(img_first, img_last)]
        self.tree_insertion_idx = [0, 0]


    def insert_into_tree(self, img_insert, fract_mixing):
        r"""
        Inserts all necessary parameters into the trajectory tree.
        Args:
            fract_mixing: float
                the fraction along the transition axis [0, 1]
        """
        
        b_parent1, b_parent2 = self.get_closest_idx(fract_mixing)
        left_sim = self.get_lpips_similarity(img_insert, self.tree_final_imgs[b_parent1])
        right_sim = self.get_lpips_similarity(img_insert, self.tree_final_imgs[b_parent2])
        idx_insert = b_parent1 + 1
        self.tree_final_imgs.insert(idx_insert, img_insert)
        self.tree_fracts.insert(idx_insert, fract_mixing)
        idx_max = np.max(self.tree_insertion_idx) + 1
        self.tree_insertion_idx.insert(idx_insert, idx_max)
        
        # update similarities
        self.tree_similarities[b_parent1] = left_sim
        self.tree_similarities.insert(idx_insert, right_sim)


    # Auxiliary functions
    def get_closest_idx(
            self,
            fract_mixing: float):
        r"""
        Helper function to retrieve the parents for any given mixing.
        Example: fract_mixing = 0.4 and self.tree_fracts = [0, 0.3, 0.6, 1.0]
        Will return the two closest values here, i.e. [1, 2]
        """

        pdist = fract_mixing - np.asarray(self.tree_fracts)
        pdist_pos = pdist.copy()
        pdist_pos[pdist_pos < 0] = np.inf
        b_parent1 = np.argmin(pdist_pos)
        pdist_neg = -pdist.copy()
        pdist_neg[pdist_neg <= 0] = np.inf
        b_parent2 = np.argmin(pdist_neg)

        if b_parent1 > b_parent2:
            tmp = b_parent2
            b_parent2 = b_parent1
            b_parent1 = tmp

        return b_parent1, b_parent2

    def get_mixing_parameters(self):
        r"""
        Computes which parental latents should be mixed together to achieve a smooth blend.
        As metric, we are using lpips image similarity. The insertion takes place
        where the metric is maximal.
        """
        # get_lpips_similarity
        similarities = self.tree_similarities
        # similarities = self.get_tree_similarities()
        b_closest1 = np.argmax(similarities)
        b_closest2 = b_closest1 + 1
        fract_closest1 = self.tree_fracts[b_closest1]
        fract_closest2 = self.tree_fracts[b_closest2]
        fract_mixing = (fract_closest1 + fract_closest2) / 2

        return fract_mixing, b_closest1, b_closest2
#%% FOR ASYNC INSERTION OF PROMPTS
    
class PromptBlenderAsync:
    def __init__(self, pipe, initial_prompts, n_steps):
        self.pipe = pipe
        self.prompts = initial_prompts
        self.n_steps = n_steps
        self.blended_prompts = self.blend_sequence_prompts(self.prompts, self.n_steps)
        
        
    def add_prompt(self, new_prompt):
        """
        Adds a new prompt to the sequence and updates the blended prompts.
        """
        self.prompts.append(new_prompt)
        self.blended_prompts = self.blend_sequence_prompts(self.prompts, self.n_steps)

    @staticmethod
    @torch.no_grad()
    def interpolate_spherical(p0, p1, fract_mixing: float):
        """
        Helper function to correctly mix two random variables using spherical interpolation.
        """
        if p0.dtype == torch.float16:
            recast_to = 'fp16'
        else:
            recast_to = 'fp32'

        p0 = p0.double()
        p1 = p1.double()
        norm = torch.linalg.norm(p0) * torch.linalg.norm(p1)
        epsilon = 1e-7
        dot = torch.sum(p0 * p1) / norm
        dot = dot.clamp(-1 + epsilon, 1 - epsilon)

        theta_0 = torch.arccos(dot)
        sin_theta_0 = torch.sin(theta_0)
        theta_t = theta_0 * fract_mixing
        s0 = torch.sin(theta_0 - theta_t) / sin_theta_0
        s1 = torch.sin(theta_t) / sin_theta_0
        interp = p0 * s0 + p1 * s1

        if recast_to == 'fp16':
            interp = interp.half()
        elif recast_to == 'fp32':
            interp = interp.float()

        return interp

    def get_prompt_embeds(self, prompt, negative_prompt=""):
        """
        Encodes a text prompt into embeddings using the model pipeline.
        """
        (
         prompt_embeds, 
         negative_prompt_embeds, 
         pooled_prompt_embeds, 
         negative_pooled_prompt_embeds
         ) = self.pipe.encode_prompt(
            prompt=prompt,
            prompt_2=prompt,
            device="cuda",
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            pooled_prompt_embeds=None,
            negative_pooled_prompt_embeds=None,
            lora_scale=0,
            clip_skip=False,
        )
        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

    def blend_prompts(self, embeds1, embeds2, fract):
        """
        Blends two sets of prompt embeddings based on a specified fraction.
        """
        prompt_embeds1, negative_prompt_embeds1, pooled_prompt_embeds1, negative_pooled_prompt_embeds1 = embeds1
        prompt_embeds2, negative_prompt_embeds2, pooled_prompt_embeds2, negative_pooled_prompt_embeds2 = embeds2

        blended_prompt_embeds = self.interpolate_spherical(prompt_embeds1, prompt_embeds2, fract)
        blended_negative_prompt_embeds = self.interpolate_spherical(negative_prompt_embeds1, negative_prompt_embeds2, fract)
        blended_pooled_prompt_embeds = self.interpolate_spherical(pooled_prompt_embeds1, pooled_prompt_embeds2, fract)
        blended_negative_pooled_prompt_embeds = self.interpolate_spherical(negative_pooled_prompt_embeds1, negative_pooled_prompt_embeds2, fract)

        return blended_prompt_embeds, blended_negative_prompt_embeds, blended_pooled_prompt_embeds, blended_negative_pooled_prompt_embeds

    def blend_sequence_prompts(self, prompts, n_steps):
        """
        Generates a sequence of blended prompt embeddings for a list of text prompts.
        """
        blended_prompts = []
        for i in range(len(prompts) - 1):
            prompt_embeds1 = self.get_prompt_embeds(prompts[i])
            prompt_embeds2 = self.get_prompt_embeds(prompts[i + 1])
            for step in range(n_steps):
                fract = step / float(n_steps - 1)
                blended = self.blend_prompts(prompt_embeds1, prompt_embeds2, fract)
                blended_prompts.append(blended)
        return blended_prompts
    
    
    
#%% LPIPS
if __name__ == "__main__":
    from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image, StableDiffusionXLControlNetPipeline
    from diffusers import AutoencoderTiny

    do_compile = False

    pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
    pipe.to("cuda")
    pipe.vae = AutoencoderTiny.from_pretrained('madebyollin/taesdxl', torch_device='cuda', torch_dtype=torch.float16)
    pipe.vae = pipe.vae.cuda()
    pipe.set_progress_bar_config(disable=True)
    
    if do_compile:
        from sfast.compilers.diffusion_pipeline_compiler import (compile, CompilationConfig)
        pipe.enable_xformers_memory_efficient_attention()
        config = CompilationConfig.Default()
        config.enable_xformers = True
        config.enable_triton = True
        config.enable_cuda_graph = True
        pipe = compile(pipe, config)

#%%
    self = PromptBlender(pipe)
    self.load_lpips()

    self.set_prompt1("photo of a house")
    self.set_prompt2("painting of a cat")
    
    latents = torch.randn((1,4,64,64)).half().cuda() # 64 is the fastest

    self.init_tree(latents=latents)
    
    
    #%%
    nmb_branches = 50
    for i in range(nmb_branches):
        fract, b1, b2 = self.get_mixing_parameters()
        img_mix = self.generate_blended_img(fract, latents)
        self.insert_into_tree(img_mix, fract)

#%%
    from lunar_tools import MovieSaver
    ms = MovieSaver("/tmp/test.mp4")
    for img in self.tree_final_imgs:
        ms.write_frame(img)
    ms.finalize()
    



    
