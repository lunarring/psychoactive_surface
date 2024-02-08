#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 22:13:24 2023

@author: lunar
"""

#%%
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image, StableDiffusionXLControlNetPipeline
from diffusers.models.unet_2d_condition import UNet2DConditionOutput
import torch

from diffusers import AutoencoderTiny
from sfast.compilers.stable_diffusion_pipeline_compiler import (compile, CompilationConfig)
from diffusers import ControlNetModel
from diffusers.utils import load_image
import lunar_tools as lt
import numpy as np
from PIL import Image
from lunar_tools.comms import OSCSender, OSCReceiver
from lunar_tools.control_input import MidiInput
from PIL import Image
import requests
import numpy as np
import cv2

# from unet
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import UNet2DConditionLoadersMixin
from diffusers.utils import USE_PEFT_BACKEND, BaseOutput, deprecate, logging, scale_lora_layers, unscale_lora_layers
from diffusers.models.activations import get_activation
from diffusers.models.attention_processor import (
    ADDED_KV_ATTENTION_PROCESSORS,
    CROSS_ATTENTION_PROCESSORS,
    Attention,
    AttentionProcessor,
    AttnAddedKVProcessor,
    AttnProcessor,
)
from diffusers.models.embeddings import (
    GaussianFourierProjection,
    ImageHintTimeEmbedding,
    ImageProjection,
    ImageTimeEmbedding,
    PositionNet,
    TextImageProjection,
    TextImageTimeEmbedding,
    TextTimeEmbedding,
    TimestepEmbedding,
    Timesteps,
)
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.unet_2d_blocks import (
    UNetMidBlock2D,
    UNetMidBlock2DCrossAttn,
    UNetMidBlock2DSimpleCrossAttn,
    get_down_block,
    get_up_block,
)

from prompt_blender import PromptBlender

akai_lpd8 = MidiInput(device_name="akai_midimix")
#mainOSCReceiver = OSCReceiver('10.20.17.122', port_receiver=8011)
mainOSCReceiver = OSCReceiver('10.20.17.122', port_receiver=8100)

torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

use_maxperf = False
engine_mode = 'puretext'    # puretext, image2image, ctrl


if engine_mode == 'image2image':
    print('Using image2image node')
    pipe = AutoPipelineForImage2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16", local_files_only=True)
elif engine_mode == 'ctrl':
    print('Using control net node')
    ctrlnet_type = "diffusers/controlnet-canny-sdxl-1.0-mid"
    
    controlnet = ControlNetModel.from_pretrained(
        ctrlnet_type,
        variant="fp16",
        use_safetensors=True,
        torch_dtype=torch.float16,
    ).to("cuda")
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained("stabilityai/sdxl-turbo", controlnet=controlnet, torch_dtype=torch.float16, variant="fp16")    
else:
    print('Using pure text')
    pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
    
    
pipe.to("cuda")
pipe.vae = AutoencoderTiny.from_pretrained('madebyollin/taesdxl', torch_device='cuda', torch_dtype=torch.float16)
pipe.vae = pipe.vae.cuda()
pipe.set_progress_bar_config(disable=True)

if use_maxperf:
    config = CompilationConfig.Default()
    config.enable_xformers = True
    config.enable_triton = True
    config.enable_cuda_graph = True
    config.enable_jit = True
    config.enable_jit_freeze = True
    config.trace_scheduler = True
    config.enable_cnn_optimization = True
    config.preserve_parameters = False
    config.prefer_lowp_gemm = True
    
    pipe = compile(pipe, config)

#%%
def get_ctrl_img(cam_img, ctrlnet_type, low_threshold=100, high_threshold=200):
    
    cam_img = np.array(cam_img)
    ctrl_image = cv2.Canny(cam_img, low_threshold, high_threshold)
    ctrl_image = ctrl_image[:, :, None]
    ctrl_image = np.concatenate([ctrl_image, ctrl_image, ctrl_image], axis=2)
    ctrl_image = Image.fromarray(ctrl_image)

    return ctrl_image 

def zoom_image(image, zoom_factor):
    if zoom_factor == 1.0:
        return image  # No change if zoom_factor is 1.0

    # Calculate the new dimensions
    height, width, _ = image.shape
    new_height = int(height / zoom_factor)
    new_width = int(width / zoom_factor)

    # Calculate the cropping coordinates to keep the center
    crop_x = (width - new_width) // 2
    crop_y = (height - new_height) // 2

    # Crop the image around its center
    cropped_image = image[crop_y:crop_y + new_height, crop_x:crop_x + new_width]

    # Resize the cropped image to the original size
    zoomed_image = cv2.resize(cropped_image, (width, height), interpolation=cv2.INTER_LINEAR)

    return zoomed_image



torch.manual_seed(1)
noise_level = 0

#%%
# Image generation pipeline
sz = (512*2, 512*2)
renderer = lt.Renderer(width=sz[1], height=sz[0])
latents = torch.randn((1,4,64,64)).half().cuda() # 64 is the fastest
latents_additive = torch.randn((1,4,64,64)).half().cuda() # 64 is the fastest

prompts = ['cat', 'cat and dog']

# Iterate over blended promptsblender = PromptBlender(pipe)

prompts = []
prompts.append('painting of a rodent')
prompts.append('painting of a tree')
prompts.append('painting of a house')
prompts.append('painting of a car')
prompts.append('painting of a face')
prompts.append('painting of a cat')
prompts.append('painting of a water pump')
prompts.append('painting of a freaky child')

blender = PromptBlender(pipe)

prompt_embeds = blender.get_all_embeddings(prompts)

blender.set_init_position(0)
blender.set_target(1)
velocity = 10

#%%# ovreride unet forward

def forward(
    self,
    sample: torch.FloatTensor,
    timestep: Union[torch.Tensor, float, int],
    encoder_hidden_states: torch.Tensor,
    class_labels: Optional[torch.Tensor] = None,
    timestep_cond: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
    down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
    mid_block_additional_residual: Optional[torch.Tensor] = None,
    down_intrablock_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
    encoder_attention_mask: Optional[torch.Tensor] = None,
    return_dict: bool = True,
) -> Union[UNet2DConditionOutput, Tuple]:
    r"""
    The [`UNet2DConditionModel`] forward method.

    Args:
        sample (`torch.FloatTensor`):
            The noisy input tensor with the following shape `(batch, channel, height, width)`.
        timestep (`torch.FloatTensor` or `float` or `int`): The number of timesteps to denoise an input.
        encoder_hidden_states (`torch.FloatTensor`):
            The encoder hidden states with shape `(batch, sequence_length, feature_dim)`.
        class_labels (`torch.Tensor`, *optional*, defaults to `None`):
            Optional class labels for conditioning. Their embeddings will be summed with the timestep embeddings.
        timestep_cond: (`torch.Tensor`, *optional*, defaults to `None`):
            Conditional embeddings for timestep. If provided, the embeddings will be summed with the samples passed
            through the `self.time_embedding` layer to obtain the timestep embeddings.
        attention_mask (`torch.Tensor`, *optional*, defaults to `None`):
            An attention mask of shape `(batch, key_tokens)` is applied to `encoder_hidden_states`. If `1` the mask
            is kept, otherwise if `0` it is discarded. Mask will be converted into a bias, which adds large
            negative values to the attention scores corresponding to "discard" tokens.
        cross_attention_kwargs (`dict`, *optional*):
            A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
            `self.processor` in
            [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
        added_cond_kwargs: (`dict`, *optional*):
            A kwargs dictionary containing additional embeddings that if specified are added to the embeddings that
            are passed along to the UNet blocks.
        down_block_additional_residuals: (`tuple` of `torch.Tensor`, *optional*):
            A tuple of tensors that if specified are added to the residuals of down unet blocks.
        mid_block_additional_residual: (`torch.Tensor`, *optional*):
            A tensor that if specified is added to the residual of the middle unet block.
        encoder_attention_mask (`torch.Tensor`):
            A cross-attention mask of shape `(batch, sequence_length)` is applied to `encoder_hidden_states`. If
            `True` the mask is kept, otherwise if `False` it is discarded. Mask will be converted into a bias,
            which adds large negative values to the attention scores corresponding to "discard" tokens.
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether or not to return a [`~models.unet_2d_condition.UNet2DConditionOutput`] instead of a plain
            tuple.
        cross_attention_kwargs (`dict`, *optional*):
            A kwargs dictionary that if specified is passed along to the [`AttnProcessor`].
        added_cond_kwargs: (`dict`, *optional*):
            A kwargs dictionary containin additional embeddings that if specified are added to the embeddings that
            are passed along to the UNet blocks.
        down_block_additional_residuals (`tuple` of `torch.Tensor`, *optional*):
            additional residuals to be added to UNet long skip connections from down blocks to up blocks for
            example from ControlNet side model(s)
        mid_block_additional_residual (`torch.Tensor`, *optional*):
            additional residual to be added to UNet mid block output, for example from ControlNet side model
        down_intrablock_additional_residuals (`tuple` of `torch.Tensor`, *optional*):
            additional residuals to be added within UNet down blocks, for example from T2I-Adapter side model(s)

    Returns:
        [`~models.unet_2d_condition.UNet2DConditionOutput`] or `tuple`:
            If `return_dict` is True, an [`~models.unet_2d_condition.UNet2DConditionOutput`] is returned, otherwise
            a `tuple` is returned where the first element is the sample tensor.
    """
    # By default samples have to be AT least a multiple of the overall upsampling factor.
    # The overall upsampling factor is equal to 2 ** (# num of upsampling layers).
    # However, the upsampling interpolation output size can be forced to fit any upsampling size
    # on the fly if necessary.
    default_overall_up_factor = 2**self.num_upsamplers

    # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
    forward_upsample_size = False
    upsample_size = None

    for dim in sample.shape[-2:]:
        if dim % default_overall_up_factor != 0:
            # Forward upsample size to force interpolation output size.
            forward_upsample_size = True
            break

    # ensure attention_mask is a bias, and give it a singleton query_tokens dimension
    # expects mask of shape:
    #   [batch, key_tokens]
    # adds singleton query_tokens dimension:
    #   [batch,                    1, key_tokens]
    # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
    #   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
    #   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
    if attention_mask is not None:
        # assume that mask is expressed as:
        #   (1 = keep,      0 = discard)
        # convert mask into a bias that can be added to attention scores:
        #       (keep = +0,     discard = -10000.0)
        attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
        attention_mask = attention_mask.unsqueeze(1)

    # convert encoder_attention_mask to a bias the same way we do for attention_mask
    if encoder_attention_mask is not None:
        encoder_attention_mask = (1 - encoder_attention_mask.to(sample.dtype)) * -10000.0
        encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

    # 0. center input if necessary
    if self.config.center_input_sample:
        sample = 2 * sample - 1.0

    # 1. time
    timesteps = timestep
    if not torch.is_tensor(timesteps):
        # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
        # This would be a good case for the `match` statement (Python 3.10+)
        is_mps = sample.device.type == "mps"
        if isinstance(timestep, float):
            dtype = torch.float32 if is_mps else torch.float64
        else:
            dtype = torch.int32 if is_mps else torch.int64
        timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
    elif len(timesteps.shape) == 0:
        timesteps = timesteps[None].to(sample.device)

    # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
    timesteps = timesteps.expand(sample.shape[0])

    t_emb = self.time_proj(timesteps)

    # `Timesteps` does not contain any weights and will always return f32 tensors
    # but time_embedding might actually be running in fp16. so we need to cast here.
    # there might be better ways to encapsulate this.
    t_emb = t_emb.to(dtype=sample.dtype)

    emb = self.time_embedding(t_emb, timestep_cond)
    aug_emb = None

    if self.class_embedding is not None:
        if class_labels is None:
            raise ValueError("class_labels should be provided when num_class_embeds > 0")

        if self.config.class_embed_type == "timestep":
            class_labels = self.time_proj(class_labels)

            # `Timesteps` does not contain any weights and will always return f32 tensors
            # there might be better ways to encapsulate this.
            class_labels = class_labels.to(dtype=sample.dtype)

        class_emb = self.class_embedding(class_labels).to(dtype=sample.dtype)

        if self.config.class_embeddings_concat:
            emb = torch.cat([emb, class_emb], dim=-1)
        else:
            emb = emb + class_emb

    if self.config.addition_embed_type == "text":
        aug_emb = self.add_embedding(encoder_hidden_states)
    elif self.config.addition_embed_type == "text_image":
        # Kandinsky 2.1 - style
        if "image_embeds" not in added_cond_kwargs:
            raise ValueError(
                f"{self.__class__} has the config param `addition_embed_type` set to 'text_image' which requires the keyword argument `image_embeds` to be passed in `added_cond_kwargs`"
            )

        image_embs = added_cond_kwargs.get("image_embeds")
        text_embs = added_cond_kwargs.get("text_embeds", encoder_hidden_states)
        aug_emb = self.add_embedding(text_embs, image_embs)
    elif self.config.addition_embed_type == "text_time":
        # SDXL - style
        if "text_embeds" not in added_cond_kwargs:
            raise ValueError(
                f"{self.__class__} has the config param `addition_embed_type` set to 'text_time' which requires the keyword argument `text_embeds` to be passed in `added_cond_kwargs`"
            )
        text_embeds = added_cond_kwargs.get("text_embeds")
        if "time_ids" not in added_cond_kwargs:
            raise ValueError(
                f"{self.__class__} has the config param `addition_embed_type` set to 'text_time' which requires the keyword argument `time_ids` to be passed in `added_cond_kwargs`"
            )
        time_ids = added_cond_kwargs.get("time_ids")
        time_embeds = self.add_time_proj(time_ids.flatten())
        time_embeds = time_embeds.reshape((text_embeds.shape[0], -1))
        add_embeds = torch.concat([text_embeds, time_embeds], dim=-1)
        add_embeds = add_embeds.to(emb.dtype)
        aug_emb = self.add_embedding(add_embeds)
    elif self.config.addition_embed_type == "image":
        # Kandinsky 2.2 - style
        if "image_embeds" not in added_cond_kwargs:
            raise ValueError(
                f"{self.__class__} has the config param `addition_embed_type` set to 'image' which requires the keyword argument `image_embeds` to be passed in `added_cond_kwargs`"
            )
        image_embs = added_cond_kwargs.get("image_embeds")
        aug_emb = self.add_embedding(image_embs)
    elif self.config.addition_embed_type == "image_hint":
        # Kandinsky 2.2 - style
        if "image_embeds" not in added_cond_kwargs or "hint" not in added_cond_kwargs:
            raise ValueError(
                f"{self.__class__} has the config param `addition_embed_type` set to 'image_hint' which requires the keyword arguments `image_embeds` and `hint` to be passed in `added_cond_kwargs`"
            )
        image_embs = added_cond_kwargs.get("image_embeds")
        hint = added_cond_kwargs.get("hint")
        aug_emb, hint = self.add_embedding(image_embs, hint)
        sample = torch.cat([sample, hint], dim=1)

    emb = emb + aug_emb if aug_emb is not None else emb

    if self.time_embed_act is not None:
        emb = self.time_embed_act(emb)

    if self.encoder_hid_proj is not None and self.config.encoder_hid_dim_type == "text_proj":
        encoder_hidden_states = self.encoder_hid_proj(encoder_hidden_states)
    elif self.encoder_hid_proj is not None and self.config.encoder_hid_dim_type == "text_image_proj":
        # Kadinsky 2.1 - style
        if "image_embeds" not in added_cond_kwargs:
            raise ValueError(
                f"{self.__class__} has the config param `encoder_hid_dim_type` set to 'text_image_proj' which requires the keyword argument `image_embeds` to be passed in  `added_conditions`"
            )

        image_embeds = added_cond_kwargs.get("image_embeds")
        encoder_hidden_states = self.encoder_hid_proj(encoder_hidden_states, image_embeds)
    elif self.encoder_hid_proj is not None and self.config.encoder_hid_dim_type == "image_proj":
        # Kandinsky 2.2 - style
        if "image_embeds" not in added_cond_kwargs:
            raise ValueError(
                f"{self.__class__} has the config param `encoder_hid_dim_type` set to 'image_proj' which requires the keyword argument `image_embeds` to be passed in  `added_conditions`"
            )
        image_embeds = added_cond_kwargs.get("image_embeds")
        encoder_hidden_states = self.encoder_hid_proj(image_embeds)
    elif self.encoder_hid_proj is not None and self.config.encoder_hid_dim_type == "ip_image_proj":
        if "image_embeds" not in added_cond_kwargs:
            raise ValueError(
                f"{self.__class__} has the config param `encoder_hid_dim_type` set to 'ip_image_proj' which requires the keyword argument `image_embeds` to be passed in  `added_conditions`"
            )
        image_embeds = added_cond_kwargs.get("image_embeds")
        image_embeds = self.encoder_hid_proj(image_embeds).to(encoder_hidden_states.dtype)
        encoder_hidden_states = torch.cat([encoder_hidden_states, image_embeds], dim=1)

    # 2. pre-process
    sample = self.conv_in(sample)

    # 2.5 GLIGEN position net
    if cross_attention_kwargs is not None and cross_attention_kwargs.get("gligen", None) is not None:
        cross_attention_kwargs = cross_attention_kwargs.copy()
        gligen_args = cross_attention_kwargs.pop("gligen")
        cross_attention_kwargs["gligen"] = {"objs": self.position_net(**gligen_args)}

    # 3. down
    lora_scale = cross_attention_kwargs.get("scale", 1.0) if cross_attention_kwargs is not None else 1.0
    if USE_PEFT_BACKEND:
        # weight the lora layers by setting `lora_scale` for each PEFT layer
        scale_lora_layers(self, lora_scale)

    is_controlnet = mid_block_additional_residual is not None and down_block_additional_residuals is not None
    # using new arg down_intrablock_additional_residuals for T2I-Adapters, to distinguish from controlnets
    is_adapter = down_intrablock_additional_residuals is not None
    # maintain backward compatibility for legacy usage, where
    #       T2I-Adapter and ControlNet both use down_block_additional_residuals arg
    #       but can only use one or the other
    if not is_adapter and mid_block_additional_residual is None and down_block_additional_residuals is not None:
        deprecate(
            "T2I should not use down_block_additional_residuals",
            "1.3.0",
            "Passing intrablock residual connections with `down_block_additional_residuals` is deprecated \
                   and will be removed in diffusers 1.3.0.  `down_block_additional_residuals` should only be used \
                   for ControlNet. Please make sure use `down_intrablock_additional_residuals` instead. ",
            standard_warn=False,
        )
        down_intrablock_additional_residuals = down_block_additional_residuals
        is_adapter = True

    down_block_res_samples = (sample,)
    for i, downsample_block in enumerate(self.down_blocks):
        
        noise_coef = akai_lpd8.get(f"A{i}", val_min=0, val_max=1, val_default=0)
        noise = torch.randn(sample.shape, device=sample.device) * noise_coef * (10**i) / 10
        sample += noise   
        
        if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
            # For t2i-adapter CrossAttnDownBlock2D
            additional_residuals = {}
            if is_adapter and len(down_intrablock_additional_residuals) > 0:
                additional_residuals["additional_residuals"] = down_intrablock_additional_residuals.pop(0)

            sample, res_samples = downsample_block(
                hidden_states=sample,
                temb=emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
                encoder_attention_mask=encoder_attention_mask,
                **additional_residuals,
            )
        else:
            sample, res_samples = downsample_block(hidden_states=sample, temb=emb, scale=lora_scale)
            if is_adapter and len(down_intrablock_additional_residuals) > 0:
                sample += down_intrablock_additional_residuals.pop(0)

        down_block_res_samples += res_samples
        
    if is_controlnet:
        new_down_block_res_samples = ()

        for down_block_res_sample, down_block_additional_residual in zip(
            down_block_res_samples, down_block_additional_residuals
        ):
            down_block_res_sample = down_block_res_sample + down_block_additional_residual
            new_down_block_res_samples = new_down_block_res_samples + (down_block_res_sample,)

        down_block_res_samples = new_down_block_res_samples

    # 4. mid
    if self.mid_block is not None:
        if hasattr(self.mid_block, "has_cross_attention") and self.mid_block.has_cross_attention:
            sample = self.mid_block(
                sample,
                emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
                encoder_attention_mask=encoder_attention_mask,
            )
        else:
            sample = self.mid_block(sample, emb)
            
        noise_coef = akai_lpd8.get("B0", val_min=0, val_max=100, val_default=0)
        noise = torch.randn(sample.shape, device=sample.device) * noise_coef
        sample += noise            
            
            
        # To support T2I-Adapter-XL
        if (
            is_adapter
            and len(down_intrablock_additional_residuals) > 0
            and sample.shape == down_intrablock_additional_residuals[0].shape
        ):
            sample += down_intrablock_additional_residuals.pop(0)

    if is_controlnet:
        sample = sample + mid_block_additional_residual

    # 5. up
    for i, upsample_block in enumerate(self.up_blocks):
        is_final_block = i == len(self.up_blocks) - 1

        res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
        down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

        # if we have not reached the final block and need to forward the
        # upsample size, we do it here
        if not is_final_block and forward_upsample_size:
            upsample_size = down_block_res_samples[-1].shape[2:]
            
        noise_coef = akai_lpd8.get(f"C{i}", val_min=0, val_max=10, val_default=0)
        noise = torch.randn(sample.shape, device=sample.device) * noise_coef
        sample += noise                  
            
        if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
            sample = upsample_block(
                hidden_states=sample,
                temb=emb,
                res_hidden_states_tuple=res_samples,
                encoder_hidden_states=encoder_hidden_states,
                cross_attention_kwargs=cross_attention_kwargs,
                upsample_size=upsample_size,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
            )
            
            
        else:
            sample = upsample_block(
                hidden_states=sample,
                temb=emb,
                res_hidden_states_tuple=res_samples,
                upsample_size=upsample_size,
                scale=lora_scale,
            )
            
    # 6. post-process
    if self.conv_norm_out:
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
    sample = self.conv_out(sample)

    if USE_PEFT_BACKEND:
        # remove `lora_scale` from each PEFT layer
        unscale_lora_layers(self, lora_scale)

    if not return_dict:
        return (sample,)

    return UNet2DConditionOutput(sample=sample)

pipe.unet.forward = lambda *args, **kwargs: forward(pipe.unet, *args, **kwargs)


#%%

url = 'https://img.freepik.com/free-photo/close-up-white-rat_23-2150760678.jpg?size=626&ext=jpg&ga=GA1.1.1880011253.1698883200&semt=ais'
image_from_web = Image.open(requests.get(url, stream=True).raw)
initial_image = np.uint8(image_from_web)
sz = initial_image.shape
h = (sz[1] - sz[0])//2
initial_image = initial_image[:,h:-h,:]
initial_image = cv2.resize(initial_image, (latents_additive.shape[2]*8,)*2)

#
#n_steps = 2500
#blended_prompts = blender.blend_sequence_prompts(prompts, n_steps)


blender_step_debug = 0
initial_image_crop = initial_image[100:120, 100:120, :]

while True:
    
    # if data is not None:
    #     print(f'data {len(data)}')    
    
    # latents_modulated = latents + osc_data * latents_additive * 5*1e-1
    # latents_modulated = latents + torch.roll(latents_additive, int(osc_data*100)) * 0.2
    # latents_modulated = torch.roll(latents_additive, int(osc_data*20))
    
    osc_data_env1 = mainOSCReceiver.get_last_value("/env1")
    osc_data_env2 = mainOSCReceiver.get_last_value("/env2")
    osc_data_env3 = mainOSCReceiver.get_last_value("/tracker")
    
    # print(f'data {osc_data_env3[-1]}')        button
    
    # fract = float(i) / (len(blended_prompts) - 1)
    fract = 0
    
    # modulation gain
    a0 = akai_lpd8.get("A0", val_min=0, val_max=1, val_default=0)
    a1 = akai_lpd8.get("A1", val_min=0, val_max=1, val_default=0)
    a2 = akai_lpd8.get("A2", val_min=0, val_max=1e-1, val_default=0.001)
    
    # image2image strength
    image2image_strength = akai_lpd8.get("B0", val_min=0.5, val_max=1, val_default=0.75)
    image2image_guidance = akai_lpd8.get("B1", val_min=0.5, val_max=1, val_default=0.75)
    # print(f'image2image_strength {image2image_strength}')
    
    # image2image modulation gain
    c0 = akai_lpd8.get("C0", val_min=0, val_max=1, val_default=0)
    c1 = akai_lpd8.get("C1", val_min=0.1, val_max=1, val_default=0.5)  # zoom the canny
    
    low_threshold = akai_lpd8.get("D0", val_default=0, val_min=0, val_max=255)
    high_threshold = akai_lpd8.get("D1", val_default=70, val_min=0, val_max=255)        
    num_inference_steps = 2

    controlnet_conditioning_scale = akai_lpd8.get("D2", val_min=0, val_max=1, val_default=0.5)        
    guidance_scale = 0.0
    
    do_canny_debug = not akai_lpd8.get("A3", button_mode='toggle')
    
    for bi in range(8):
        button = akai_lpd8.get(f"{chr(65+bi)}4", button_mode='pressed_once')
        if button:
            blender.set_target(bi)
            blender_step_debug = 0
    
    # here we modulate prompts transitions
    # i_b = i
    # i_b += (int(osc_data_env1*50000*a0)-0.5)
    # i_b = int(np.clip(i_b, 0, len(blended_prompts)-2))

    # blended = blender.blend_prompts(blended_prompts[i_b], blended_prompts[i_b+1], fract)
    # prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = blended
    
    blender_step_debug += a2
    
    if blender_step_debug > 1:
        blender_step_debug = 1
    blender.step(blender_step_debug)
    # blender.step(a2)
    
    # if blender.first_fract == 1:
    #     blender.first_fract = 0
    #     blender.set_target(np.random.randint(len(prompts)))
    
    prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = blender.current        
    
    # print(f'fract {blender.fract}')
    # print(f'a2 {a2}')
    
    # here we modulate noise
    height = latents.shape[2]
    vert_ramp = torch.linspace(0,1,height).to(latents.device)
    latents_modulated = latents + latents_additive * vert_ramp.reshape([1,1,height,1]) * a1
    latents_modulated = latents_modulated.half()
    
    # here we modulate image2image inputs
    initial_image_modulated = initial_image.copy()
    # initial_image_modulated[100:200,100:200,:] = initial_image_modulated[100:200,100:200,:] * c0
    
    # initial_image_modulated = np.roll(initial_image_modulated, int((c0-0.5)*1000))
    # initial_image_modulated = np.roll(initial_image_modulated, int((osc_data_env1)*1000*c0), axis=0)
    if engine_mode == 'image2image':
        c = initial_image_crop.shape[0]
        
        i_x = int(c0 * (initial_image_modulated.shape[1]-c))
        i_y = int(c1 * (initial_image_modulated.shape[0]-c))
        
        # initial_image_modulated[:, i_x, :] = 255
        # initial_image_modulated[i_y, :, :] = 255
        
        initial_image_modulated[i_y:i_y+c, i_x:i_x+c, :] = initial_image_crop
        
        if do_canny_debug:
            image = initial_image_modulated
        else:
            torch.manual_seed(0)
            image = pipe(image=Image.fromarray(initial_image_modulated), latents=latents, num_inference_steps=2, strength=image2image_strength, 
                         guidance_scale=image2image_guidance, prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds, 
                         pooled_prompt_embeds=pooled_prompt_embeds, negative_pooled_prompt_embeds=negative_pooled_prompt_embeds).images[0]
    
    
    
    elif engine_mode == 'ctrl':        
        torch.manual_seed(0)
        ctrl_img = get_ctrl_img(initial_image, ctrlnet_type, low_threshold=low_threshold, high_threshold=high_threshold)
        #ctrl_img = np.roll(np.array(ctrl_img), int(1000*c0), axis=1)
        ctrl_img = np.array(ctrl_img)
        i_x = int(c0 * (ctrl_img.shape[1]-1))
        i_y = int(c1 * (ctrl_img.shape[0]-1))
        
        ctrl_img[ctrl_img.shape[0]//2, i_x, :] = 255
        ctrl_img[i_y, ctrl_img.shape[1]//2, :] = 255
        
        # ctrl_img = zoom_image(np.array(ctrl_img), c1)
        if do_canny_debug:
            image = ctrl_img
        else:
            image = pipe(image=Image.fromarray(ctrl_img), latents=latents, controlnet_conditioning_scale=controlnet_conditioning_scale, guidance_scale=guidance_scale, num_inference_steps=num_inference_steps, prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds, pooled_prompt_embeds=pooled_prompt_embeds, negative_pooled_prompt_embeds=negative_pooled_prompt_embeds).images[0]            
    else:
        image = pipe(guidance_scale=0.0, num_inference_steps=1, latents=latents_modulated, 
                     prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds, 
                     pooled_prompt_embeds=pooled_prompt_embeds, negative_pooled_prompt_embeds=negative_pooled_prompt_embeds).images[0]
        
    # Render the image
    image = np.asanyarray(image)
    image = np.uint8(image)
    renderer.render(image)
        
        
    
#%%
        
