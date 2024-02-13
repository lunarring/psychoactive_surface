import numpy as np
import lunar_tools as lt


if __name__ == '__main__':
    # Get list of prompts
    nmb_rows,nmb_cols = (4,8)       # number of tiles
    from datasets import load_dataset
    import random
    dataset = load_dataset("FredZhang7/stable-diffusion-prompts-2.47M")
    
    list_prompts_all = [random.choice(dataset['train'])['text'] for i in range(nmb_rows*nmb_cols)]
    
    # Convert to images
    shape_hw = (128, 256)   # image size
    from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image, StableDiffusionXLControlNetPipeline
    from diffusers import AutoencoderTiny
    import torch
    from prompt_blender import PromptBlender
    from tqdm import tqdm


    pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
    pipe.to("cuda")
    pipe.vae = AutoencoderTiny.from_pretrained('madebyollin/taesdxl', torch_device='cuda', torch_dtype=torch.float16)
    pipe.vae = pipe.vae.cuda()
    pipe.set_progress_bar_config(disable=True)
    
    from sfast.compilers.diffusion_pipeline_compiler import (compile, CompilationConfig)
    pipe.enable_xformers_memory_efficient_attention()
    config = CompilationConfig.Default()
    config.enable_xformers = True
    config.enable_triton = True
    config.enable_cuda_graph = True
    pipe = compile(pipe, config)
    
    
    pb = PromptBlender(pipe)
    pb.w = 128
    latents = pb.get_latents()

    list_imgs = []
    list_prompts = []
    for i in tqdm(range(nmb_rows*nmb_cols)):
        pb.set_prompt1(list_prompts_all[i])
        pb.set_prompt2(list_prompts_all[i])
        img = pb.generate_blended_img(0.0, latents)
        img_tile = img.resize(shape_hw[::-1])
        list_imgs.append(np.asarray(img_tile))
        list_prompts.append(list_prompts_all[i])
    
    gridrenderer = lt.GridRenderer(nmb_rows,nmb_cols,shape_hw)
    secondary_renderer = lt.Renderer(width=1024, height=512, backend='opencv')
    
    gridrenderer.update(list_imgs)
    
    #%%#
    def get_aug_prompt(prompt):
        return prompt
    
    negative_prompt = "blurry, lowres, disfigured"
    space_prompt = "photo of the moon"
    
    # Run space
    idx_cycle = 0
    pb.set_prompt1(get_aug_prompt(space_prompt), negative_prompt)
    latents2 = pb.get_latents()
    
    while True:
        # cycle back from target to source
        latents1 = latents2.clone()
        pb.embeds1 = pb.embeds2
        # get new target
        latents2 = pb.get_latents()
        pb.set_prompt2(get_aug_prompt(space_prompt), negative_prompt)
    
        fract = 0
        while fract < 1:
            d_fract = 0.01#akai_lpd8.get("E0", val_min=0.005, val_max=0.1)
            latents_mix = pb.interpolate_spherical(latents1, latents2, fract)
            img_mix = pb.generate_blended_img(fract, latents_mix)
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
                
        idx_cycle += 1

    
    
    


            
    
    
    
    
    
    
    
    
    
    
