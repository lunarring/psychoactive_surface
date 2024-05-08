import os
import torch
torch.backends.cudnn.benchmark = False
torch.set_grad_enabled(False)
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from tqdm.auto import tqdm
from PIL import Image
import gradio as gr
import shutil
import uuid
from diffusers import AutoPipelineForText2Image
import datetime

warnings.filterwarnings('ignore')
torch.set_grad_enabled(False)
torch.backends.cudnn.benchmark = False
import json
import lunar_tools as lt
from datetime import datetime
import argparse


class GradioHolder():
    def __init__(
            self,
            pipe,
            width=512,
            height=512,
            share=False):
        r"""
        Args:
            pipe:
                pipe
            share: bool
                Set true to get a shareable gradio link (e.g. for running a remote server)
        """
        self.pipe = pipe
        self.share = share
        self.current_prompt = ""
        self.start_time = datetime.now()
        self.filename = "prompt_generator_" + self.start_time.strftime("%y%m%d_%H%M" + ".txt")
        self.idx_prompt = 0 
        self.width = width
        self.height = height
        self.num_inference_steps = 2
        self.good_prompts = ""


    def generate_img(self, prompt):
        return self.pipe(guidance_scale=0.0, num_inference_steps=self.num_inference_steps, prompt=prompt, width=self.width, height=self.height).images[0]
    

    def save(self, prompt):
        self.good_prompts += f"\n{prompt}"
        with open(self.filename, "w") as file:
            file.write(self.good_prompts)
    
        



if __name__ == "__main__":
    
    # Creating the parser
    parser = argparse.ArgumentParser(description="Process the server IP.")
    
    # Adding the server_ip argument
    parser.add_argument("--server_ip", type=str, help="The IP address of the server", default=None)
    
    # Parsing the arguments
    args = parser.parse_args()
    
    
    width = 1024
    height = 512
    num_inference_steps = 2
    gpt_model = "gpt-4"
    
    pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16").to("cuda")
    gh = GradioHolder(pipe, width=width, height=height)
    
    txt_instructions = "You are a brilliant prompt engineer. Here is an example of your work: {good_prompts}. Make a list or more prompts like that, be creative and not repetitive! Don't make a numbered list, just give me the descriptions, without any formatting elements, do it JUST LIKE IN THE EXAMPLES!"
    
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                img0 = gr.Image(label="seed1")
            with gr.Column():
                b_render = gr.Button('generate image')
                b_good = gr.Button('save this prompt', variant='primary')
                current_prompt = gr.Textbox(label="current prompt", interactive=True)

        b_render.click(gh.generate_img, inputs=[current_prompt], outputs=[img0])
        b_good.click(gh.save, inputs=[current_prompt], outputs=None)
        
    
    demo.launch(share=True, inbrowser=True, inline=False, server_name=args.server_ip)

        
