import numpy as np
import lunar_tools as lt

class GridRenderer():
    def __init__(self, M, N, sz):
        """
                       M: Number of tiles in vertical direction
                       N: Number of tiles in horizontal direction
                       sz: (H,W) = tuple (height,width)
        """
        
        self.H = sz[0]
        self.W = sz[1]
        self.M = M
        self.N = N
        self.canvas = np.zeros((M,N,sz[0],sz[1],3))
        
        self.renderer = lt.Renderer(width=sz[1]*N, height=sz[0]*M)
        
    def inject_tiles(self, tiles):
        """
        Concatenate image tiles into one large canvas.
    
        :param tiles: NumPy array of shape MxNxHxW*C
                       M: Number of tiles in vertical direction
                       N: Number of tiles in horizontal direction
                       H: Height of each tile
                       W: Width of each tile
                       C: Number of RGB channels
        :return: NumPy array representing the large canvas with shape (M*H)x(N*W)
        """
        M, N, H, W, C = tiles.shape
        fail_msg = 'GridRenderer->inject_tiles: tiles shape inconsistent with initialization'
        assert (M == self.M) and (N == self.N), print(fail_msg)
        assert (H == self.H) and (W == self.W), print(fail_msg)
        
        # Reshape and transpose to bring tiles next to each other
        self.canvas = tiles.transpose(0, 2, 1, 3, 4).reshape(M*H, N*W, C)
        
    def list_to_tensor(self, list_images):
        """
        Reshape image tiles from list to tensor.
    
        :param list_images: list of images of shape H*W*3
        :return: NumPy array of shape MxNxHxW*C
        """        
        
        grid_input = np.zeros((self.M, self.N, self.H, self.W, 3))
        for m in range(self.M):
            for n in range(self.N):
                if m*self.N + n < len(list_images):
                    grid_input[m,n,:,:,:] = list_images[m*self.N + n]
        return grid_input        

    def render(self):
        """
        Render canvas abd find the index of the tile given a mouse click pixel coordinate on the 2D canvas.
        :return: A tuple (m, n) representing the tile index in the range (0..M, 0..N).
        """
        
        peripheralEvent = self.renderer.render(self.canvas)
        
        if peripheralEvent.mouse_button_state > 0:
            x = peripheralEvent.mouse_posX
            y = peripheralEvent.mouse_posY
            
            m = y // self.H
            n = x // self.W
            return m, n
        else:
            return -1, -1
        

if __name__ == '__main__':
    # Get list of prompts
    M,N = (4,8)       # number of tiles
    # from datasets import load_dataset
    import random
    # dataset = load_dataset("FredZhang7/stable-diffusion-prompts-2.47M")
    
    list_prompts_all = []
    with open("good_prompts.txt", "r") as file: 
        list_prompts_all = file.read().split('\n')
        
    # Convert to images
    sz = (128, 256)   # image size
    from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image, StableDiffusionXLControlNetPipeline
    from diffusers import AutoencoderTiny
    import torch
    from prompt_blender import PromptBlender
    from tqdm import tqdm
    from sfast.compilers.diffusion_pipeline_compiler import (compile, CompilationConfig)


    pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
    pipe.to("cuda")
    pipe.vae = AutoencoderTiny.from_pretrained('madebyollin/taesdxl', torch_device='cuda', torch_dtype=torch.float16)
    pipe.vae = pipe.vae.cuda()
    pipe.set_progress_bar_config(disable=True)
    
    pipe.enable_xformers_memory_efficient_attention()
    config = CompilationConfig.Default()
    config.enable_xformers = True
    config.enable_triton = True
    config.enable_cuda_graph = True
    pipe = compile(pipe, config)
    
    akai_midimix = lt.MidiInput("akai_midimix")
    
    pb = PromptBlender(pipe)
    pb.w = 64
    pb.h = 64
    latents = pb.get_latents()
    
    def get_prompts_and_img():
        list_imgs = []
        list_prompts = random.sample(list_prompts_all, N*M)
        for prompt in tqdm(list_prompts):
            pb.set_prompt1(prompt)
            pb.set_prompt2(prompt)
            img = pb.generate_blended_img(0.0, latents)
            img_tile = img.resize(sz[::-1])
            list_imgs.append(np.asarray(img_tile))
            
        return list_prompts, list_imgs
    
    
    gridrenderer = GridRenderer(M,N,sz)
    secondary_renderer = lt.Renderer(width=2048, height=1024, backend='opencv')
    
    list_prompts, list_imgs = get_prompts_and_img()
    grid_input = gridrenderer.list_to_tensor(list_imgs)
    gridrenderer.inject_tiles(grid_input)
    
    #%%#
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
            d_fract = akai_midimix.get("A0", val_min=0.005, val_max=0.1)
            latents_mix = pb.interpolate_spherical(latents1, latents2, fract)
            img_mix = pb.generate_blended_img(fract, latents_mix)
            secondary_renderer.render(img_mix)
            
            # Inject new space
            m,n = gridrenderer.render()
            if m != -1 and n != -1:
                idx = m*N + n
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
                grid_input = gridrenderer.list_to_tensor(list_imgs)
                gridrenderer.inject_tiles(grid_input)
                print("done!")
                
        idx_cycle += 1

    
    
    


            
    
    
    
    
    
    
    
    
    
    
