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
import matplotlib.pyplot as plt

#%% VARS
use_compiled_model = False
use_image2image = False
res_fact = 1.5
width_latents = int(96*res_fact)
height_latents = int(64*res_fact)
width_renderer = int(1024*2)
height_renderer = 512*2

size_img_tiles_hw = (150, 300)   # tile image size
nmb_rows, nmb_cols = (6,6)       # number of tiles
ip_address_osc_receiver = '192.168.50.130'
dir_embds_imgs = "embds_imgs"
show_osc_visualization = True

#%% aux func
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
    return torch.randn(shape, device=pipe.device, generator=torch.Generator(device=pipe.device).manual_seed(1)).half()


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
        nmb_imgs_space = len(self.prompt_spaces[self.active_space]) - 2 # two buttons exist
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

prompt_holder = PromptHolder(pb, size_img_tiles_hw)


#%% prepare prompt window
gridrenderer = lt.GridRenderer(nmb_rows, nmb_cols, size_img_tiles_hw)

if use_image2image:
    list_prompts, list_imgs = prompt_holder.get_prompts_imgs_within_space(nmb_cols*nmb_rows)
    gridrenderer.update(list_imgs)
    # image_init = Image.open("/home/lugo/Downloads/copertina-1.jpg")
    
    shape_cam=(600,800) 
    cam = lt.WebCam(cam_id=-1, shape_hw=shape_cam)
    cam.cam.set(cv2.CAP_PROP_AUTOFOCUS, 1)
else:
    # fp_movie = '/home/lugo/Downloads/20240301_150735.mp4'
    # vidcap = cv2.VideoCapture(fp_movie)    
    list_prompts, list_imgs = prompt_holder.get_prompts_imgs_within_space(nmb_cols*nmb_rows)
    gridrenderer.update(list_imgs)


receiver = lt.OSCReceiver(ip_address_osc_receiver, port_receiver = 8004)
if show_osc_visualization:
    receiver.start_visualization(shape_hw_vis=(300, 500), nmb_cols_vis=3, nmb_rows_vis=2,backend='opencv')

speech_detector = lt.Speech2Text()
#%%# img2img moving shapes

# Initialize a reference time
reference_time = time.time()

#%%#
# shape = (pb.h*8, pb.w*8, 3)  # Shape of the numpy array
# base_rect_image_noise = np.random.randint(0, 64, shape, dtype=np.uint8)

# def initialize_triangles(num_triangles, shape):
#     """
#     Initializes triangles with random colors, speeds, and rotation rates.

#     Parameters:
#     - num_triangles: int, the number of triangles to initialize.
#     - shape: tuple, the shape of the numpy array (height, width, channels).

#     Returns:
#     - List of dictionaries, each containing the triangle's properties.
#     """
#     height, width, _ = shape
#     triangles = []
#     for _ in range(num_triangles):
#         center = np.array([np.random.randint(width), np.random.randint(height)])
#         size = np.random.randint(20, 50)
#         vertices = np.array([
#             [center[0] - size, center[1] - size],
#             [center[0] + size, center[1] - size],
#             [center[0], center[1] + size]
#         ])
#         color = tuple(np.random.randint(0, 256, size=3).tolist())
#         speed = np.random.uniform(0.5*0.01, 2.0*0.01)
#         rotation_rate = np.random.uniform(-5, 5)  # degrees per second
#         rotation_rate = np.random.uniform(-1, 1)  # degrees per second

#         triangles.append({
#             'vertices': vertices,
#             'color': color,
#             'speed': speed,
#             'rotation_rate': rotation_rate,
#             'center': center
#         })
#     return triangles

# def update_triangle_centers(triangles, elapsed_time, shape):
#     """
#     Updates the center of each triangle based on its speed and the elapsed time,
#     ensuring movement across the screen.
#     """
#     height, width = shape[:2]
#     for triangle in triangles:
#         # Calculate new displacement and update the center position
#         dx = (elapsed_time * triangle['speed']) % width
#         dy = (elapsed_time * triangle['speed'] / 2) % height  # Slower vertical movement
#         new_center_x = (triangle['center'][0] + dx) % width
#         new_center_y = (triangle['center'][1] + dy) % height
#         triangle['center'] = np.array([new_center_x, new_center_y])

# def render_moving_rotating_triangles(shape, triangles):
#     global reference_time
#     img = np.zeros(shape, dtype=np.uint8)
#     elapsed_time = time.time() - reference_time

#     # Update triangle centers for movement
#     update_triangle_centers(triangles, elapsed_time, shape)

#     for triangle in triangles:
#         angle = (elapsed_time * triangle['rotation_rate']) % 360

#         # Ensure center is a tuple of floats for rotation
#         center_as_floats = tuple(map(float, triangle['center']))

#         # Rotation matrix
#         M = cv2.getRotationMatrix2D(center_as_floats, angle, 1.0)

#         # Apply rotation to triangle vertices
#         rotated_vertices = cv2.transform(np.array([triangle['vertices']]), M)[0].astype(np.int32)

#         # Draw the triangle
#         cv2.fillConvexPoly(img, rotated_vertices, triangle['color'])

#     return img

# # Initialize triangles
# triangles = initialize_triangles(3, shape)


# # Example usage
# rect_size = (50*1, 100*1)  # Size of the rectangle2
# speed = 100  # Speed of the rectangle's movement in pixels per second

# # Call the function to get a single frame
# # frame = render_moving_rectangle(shape, rect_size, speed)
# # plt.imshow(frame)
# # To display this frame, further processing is required, depending on the environment.



#%%#
av_router = AudioVisualRouter(meta_input)

negative_prompt = "blurry, lowres, disfigured"
space_prompt = prompt_holder.prompt_spaces[prompt_holder.active_space][0]

# Run space
idx_cycle = 0
pb.set_prompt1(space_prompt, negative_prompt)
pb.set_prompt2(space_prompt, negative_prompt)
latents2 = pb.get_latents()


modulations = {}
modulations_noise = {}
for i in range(3):
    modulations_noise[f'e{i}'] = get_noise_for_modulations(get_sample_shape_unet(f'e{i}'))
    modulations_noise[f'd{i}'] = get_noise_for_modulations(get_sample_shape_unet(f'd{i}'))
    
modulations_noise['b0'] = get_noise_for_modulations(get_sample_shape_unet('b0'))
modulations['modulations_noise'] = modulations_noise

noise_cam = np.random.randn(pb.h*8, pb.w*8, 3).astype(np.float32)*100

embeds_mod_full = pb.get_prompt_embeds("full of electric sparkles")

prev_diffusion_output = None
pb.num_inference_steps = 1

t_last = time.time()

sound_feature_names = ['DJLOW', 'DJMID', 'DJHIGH']

# av_router.map_av('SUB', 'b0_samp')
av_router.map_av('DJMID', 'e*_emb')
# av_router.map_av('SUB', 'd*_emb')
av_router.map_av('DJLOW', 'progress')
# av_router.map_av('SUB', 'fract_decoder_emb')
av_router.map_av('DJHIGH', 'd2_samp')

if use_image2image:
    noise_img2img = torch.randn((1,4,pb.h,pb.w)).half().cuda() * 0
    fp_image_init = '/home/lugo/Downloads/copertina-1.jpg'
    image_init_input = cv2.imread(fp_image_init)[:,:,::-1]
    image_init_input = cv2.resize(image_init_input, (pb.w*8, pb.h*8))

is_noise_trans = True

t_prompt_injected = time.time()

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
                lt.dynamic_print(f"fps: {1/dt:.1f}")
        
        # modulate osc with akai
        H0 = meta_input.get(akai_midimix="H0", akai_lpd8="G0", val_min=0, val_max=1, val_default=0)
        av_router.sound_features['DJLOW'] *= H0
        
        H1 = meta_input.get(akai_midimix="H1", akai_lpd8="G1",val_min=0, val_max=1, val_default=0)
        av_router.sound_features['DJMID'] *= H1
        
        H2 = meta_input.get(akai_midimix="H2", akai_lpd8="H0", val_min=0, val_max=100, val_default=0)
        av_router.sound_features['DJHIGH'] *= H2
        
        for i in range(3):
            modulations[f'e{i}_emb'] = torch.tensor(1 - av_router.get_modulation('e*_emb'), device=latents1.device)
            # modulations[f'd{i}_emb'] = torch.tensor(1 - av_router.get_modulation('d*_emb'), device=latents1.device)
            
        modulations['d2_samp'] = torch.tensor(av_router.get_modulation('d2_samp'), device=latents1.device)
        
        # the line below (resample_grid...) causes memory leak
        # amp = 1e-1
        # resample_grid = acidman.do_acid(modulations_noise['d2'][None].float().permute([1,2,0]), amp)
        # amp_mod = (resample_grid - acidman.identity_resample_grid)     
      
        # acid_fields = amp_mod[:,:,0][None][None], resample_grid
        # modulations['d2_acid'] = acid_fields
            
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
            # success, image_init = vidcap.read()
            # image_init = cv2.resize(image_init, (pb.w*4, pb.h*4))
            # image_init_input = render_moving_rotating_triangles(shape, triangles)
            # plt.imshow(image_init_input); plt.show(); plt.ion()
            # image_init = image_init_input.copy()
            # image_init_input = np.roll(image_init_input, 2,axis=0)
            
            cam_img = cam.get_img()
            cam_img = np.flip(cam_img, axis=1)
            image_init = cv2.resize(cam_img, (pb.w*8, pb.h*8))
            
            cam_noise_coef = meta_input.get(akai_lpd8="E0", akai_midimix="G1", val_min=0.0, val_max=1, val_default=0)
            
            image_init = image_init.astype(np.float32) + cam_noise_coef*noise_cam
            image_init = np.clip(image_init, 0, 255)
            image_init = image_init.astype(np.uint8)
            
            alpha_acid = meta_input.get(akai_lpd8="E0", akai_midimix="G2", val_min=0.0, val_max=1, val_default=0)

            if prev_diffusion_output is not None:
                prev_diffusion_output = np.array(prev_diffusion_output)
                prev_diffusion_output = np.roll(prev_diffusion_output, 2, axis=0)
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
                        pb.embeds1 = pb.blend_prompts(pb.embeds1, pb.embeds2, fract)
                        latents1 = pb.interpolate_spherical(latents1, latents2, fract)
                        space_prompt = list_prompts[idx]
                        fract = 0
                        pb.set_prompt2(space_prompt, negative_prompt)
                        is_noise_trans = False
                        t_prompt_injected = time.time()
                else:
                    # space selection
                    prompt_holder.active_space = prompt_holder.list_spaces[idx]
                    print(f"new activate space: {prompt_holder.active_space}")
                    list_prompts, list_imgs = prompt_holder.get_prompts_imgs_within_space(nmb_cols*nmb_rows)
                    gridrenderer.update(list_imgs)
                    prompt_holder.show_all_spaces = False
                    
                    
                    
                    
            except Exception as e:
                print(f"fail of click event! {e}")
        else:
            # regular movement
            # fract_osc = 0
            fract_osc = av_router.get_modulation('progress')
            if is_noise_trans:
                fract += d_fract_noise + fract_osc
            else:
                fract += d_fract_embed + fract_osc
            
        do_new_space = meta_input.get(akai_midimix="A3", akai_lpd8="A0", button_mode="released_once")
        if do_new_space:     
            prompt_holder.set_next_space()
            list_prompts, list_imgs = prompt_holder.get_prompts_imgs_within_space(nmb_cols*nmb_rows)
            gridrenderer.update(list_imgs)
            
        do_auto_change = meta_input.get(akai_midimix="A4", akai_lpd8="A1", button_mode="toggle")
        t_auto_change = meta_input.get(akai_midimix="A2", val_min=1, val_max=10)
        
        if do_auto_change and is_noise_trans and time.time() - t_prompt_injected > t_auto_change:
            # go to random img
            
            space_prompt = random.choice(list_prompts)
            fract = 0
            pb.set_prompt2(space_prompt, negative_prompt)
            is_noise_trans = False
            t_prompt_injected = time.time()
            print(f"auto change to: {space_prompt}")
        
        # do_new_prompts = meta_input.get(akai_midimix="A4", akai_lpd8="A1", button_mode="pressed_once")
        # if do_new_prompts:
        #     list_prompts, list_imgs = prompt_holder.get_prompts_imgs_within_space(nmb_cols*nmb_rows)
        #     gridrenderer.update(list_imgs)
                
    idx_cycle += 1
    is_noise_trans = True






            
    
"""
CEMENTARY

        # EXPERIMENTAL WHISPER
        # do_record_mic = meta_input.get(akai_midimix="A3", akai_lpd8="A0", button_mode="held_down")
        # do_record_mic = akai_lpd8.get('s', button_mode='pressed_once')
        
        # try:
        #     if do_record_mic:
        #         if not speech_detector.audio_recorder.is_recording:
        #             speech_detector.start_recording()
        #     elif not do_record_mic:
        #         if speech_detector.audio_recorder.is_recording:
        #             prompt = speech_detector.stop_recording()
        #             print(f"New prompt: {prompt}")
        #             if prompt is not None:
        #                 embeds_mod_full = pb.get_prompt_embeds(prompt)
        #             stop_recording = False
        # except Exception as e:
        #     print(f"FAIL {e}")
        

    
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




"""
    
    
    
    
    
    
    
