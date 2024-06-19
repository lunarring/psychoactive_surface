import sys
import numpy as np
import lunar_tools as lt
import torch
from tqdm import tqdm
from PIL import Image
import hashlib
import os
import kornia
import cv2
import colorsys
import torch.nn.functional as F
import math

from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest


class MovieReaderCustom():
    r"""
    Class to read in a movie.
    """

    def __init__(self, fp_movie, shape_cam):
        self.shape = [shape_cam[0], shape_cam[1], 3]
        self.load_movie(fp_movie)

    def load_movie(self, fp_movie):
        self.video_player_object = cv2.VideoCapture(fp_movie)
        self.nmb_frames = int(self.video_player_object.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps_movie = int(30)
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

#% Image processing functions. These should live somewhere else
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

def angle_to_rgb(angle):
    """
    Convert an angle in radians (0 to 2*pi) to an RGB color vector.
    
    Parameters:
        angle (float): Angle in radians, where 0 to 2*pi maps to 0 to 1 in the hue.

    Returns:
        tuple: RGB color as a 3-element tuple, each component in the range 0 to 1.
    """
    # Normalize the angle to a range from 0 to 1
    hue = angle / (2 * 3.141592653589793)
    # Set saturation and value to 1 for maximum intensity and brightness
    saturation = 0.9
    value = 1
    # Convert HSV to RGB
    return colorsys.hsv_to_rgb(hue, saturation, value)

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


def detect_outliers(data, contamination=0.1):
    """
    Detect outliers in the data using Isolation Forest.

    Parameters:
    - data (array-like): The input data, shape (n_samples, n_features)
    - contamination (float): The proportion of outliers in the data set, default is 0.1

    Returns:
    - numpy array: Boolean array indicating outliers (True for outliers, False for inliers)
    """
    # Ensure data is a numpy array
    data = np.array(data)
    
    # Initialize the IsolationForest
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    
    # Fit the model and predict outliers
    outliers = iso_forest.fit_predict(data)
    
    # Convert predictions (-1 for outliers, 1 for inliers) to boolean
    return outliers == -1

# Set the number of clusters
def kmeans (X, num_clusters = 2):

    # Create the k-means clustering model
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    
    # Fit the model to the data
    kmeans.fit(X)
    
    # Get the cluster centroids and labels
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_
    return labels, centroids

def rotate_hue_torch(image, angle):
    # Convert angle from degrees to radians

    image = image.permute(2,0,1).unsqueeze(0)
    angle = angle * np.pi / 180
    
    # Normalize the image tensor to [0, 1]
    if image.max() > 1:
        image = image / 255.0
    
    # Convert RGB to HSV
    image_hsv = kornia.color.rgb_to_hsv(image)
    image_hsv[:, 0, :, :] = (image_hsv[:, 0, :, :] + angle / (2 * np.pi)) % 1.0
    
    # Convert HSV back to RGB
    image_rgb = kornia.color.hsv_to_rgb(image_hsv)
    
    # Convert the image back to [0, 255] range
    image_rgb = image_rgb * 255.0
    image_rgb = image_rgb.squeeze(0).permute(1,2,0)
    return image_rgb

#%

def get_sample_shape_unet(coord, height_latents, width_latents):
    if coord[0] == 'e':
        coef = float(2**int(coord[1]))
        shape = [int(np.ceil(height_latents/coef)), int(np.ceil(width_latents/coef))]
    elif coord[0] == 'b':
        shape = [int(np.ceil(height_latents/4)), int(np.ceil(width_latents/4))]
    else:
        coef = float(2**(2-int(coord[1])))
        shape = [int(np.ceil(height_latents/coef)), int(np.ceil(width_latents/coef))]
        
    return shape

def get_noise_for_modulations(shape, pipe_text2img):
    return torch.randn(shape, device=pipe_text2img.device, generator=torch.Generator(device=pipe_text2img.device).manual_seed(1)).half()

class PromptHolder():
    def __init__(self, prompt_blender, size_img_tiles_hw, use_image2image=False, dir_prompts="prompts", dir_embds_imgs = "embds_imgs"):
        self.dir_prompts = dir_prompts
        self.dir_embds_imgs = dir_embds_imgs
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
        self.negative_prompt = "blurry, lowres, disfigured, thin lines"
        self.negative_prompt = "blurry, lowres, thin lines, text"
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

        list_prompt_txts = os.listdir(f"{self.dir_prompts}/")
        list_prompt_txts = [l for l in list_prompt_txts if l.endswith(".txt")]
        for fn_prompts in list_prompt_txts:
            name_space = fn_prompts.split(".txt")[0]
            list_prompts_all = []
            try:
                with open(f"{self.dir_prompts}/{fn_prompts}", "r", encoding="utf-8") as file: 
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
    
        fp_img = f"{self.dir_embds_imgs}/{hash_code}.jpg"
        fp_embed = f"{self.dir_embds_imgs}/{hash_code}.pkl"
        fp_prompt = f"{self.dir_embds_imgs}/{hash_code}.txt"
    
        if os.path.exists(fp_img) and os.path.exists(fp_embed) and os.path.exists(fp_prompt) :
            image = Image.open(fp_img)
            image = image.resize((self.width_images, self.height_images))
            return image, hash_code
        
        if self.use_image2image:
            return self.get_black_img(), "XXXXXX"
    
        latents = self.pb.get_latents()
        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = self.pb.get_prompt_embeds(prompt, self.negative_prompt)
        image = self.pb.generate_img(latents, prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds)
    
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

def draw_circular_patch(Y,X,y,x, brush_size):
    # Calculate the distance from the center (x, y)
    distance = ((X - x) ** 2 + (Y - y) ** 2).float().sqrt()
    
    mask = distance > brush_size
    patch = brush_size - distance
    patch[mask] = 0
    # distance = 1 / (distance + 1e-3)
    # distance[distance < brush_size] = 0
    
    return patch       

def euler_from_quaternion(x, y, z, w):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)
 
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)
 
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)
    
    euler_angles = np.array([roll_x, pitch_y, yaw_z])
 
    return euler_angles # in radians
    
    
def desaturate(img, percent):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(hsv)
    
    # desaturate
    s_desat = cv2.multiply(s, percent).astype(np.uint8)
    hsv_new = cv2.merge([h,s_desat,v])
    bgr_desat = cv2.cvtColor(hsv_new, cv2.COLOR_HSV2BGR)
    
    # create 1D LUT for green
    # (120 out of 360) = (60 out of 180)  +- 25
    lut = np.zeros((1,256), dtype=np.uint8)
    white = np.full((1,50), 255, dtype=np.uint8)
    lut[0:1, 35:85] = white
    
    # apply lut to hue channel as mask
    mask = cv2.LUT(h, lut)
    mask = mask.astype(np.float32) / 255
    mask = cv2.merge([mask,mask,mask])
    
    # mask bgr_desat and img
    result = mask * bgr_desat + (1 - mask)*img
    result = result.clip(0,255).astype(np.uint8)
    
    return result