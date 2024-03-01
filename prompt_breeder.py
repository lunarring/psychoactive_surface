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
from latentblending.blending_engine import BlendingEngine
import datetime

warnings.filterwarnings('ignore')
torch.set_grad_enabled(False)
torch.backends.cudnn.benchmark = False
import json
import lunar_tools as lt
from datetime import datetime


class GradioHolder():
    def __init__(
            self,
            pipe,
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
        self.gpt4 = lt.GPT4(model="gpt-4-turbo-preview")
        self.current_prompt = ""
        self.start_time = datetime.now()
        self.filename = "promptbreeder_" + self.start_time.strftime("%y%m%d_%H%M") + ".txt"


    def generate_img(self, prompt):
        return self.pipe(guidance_scale=0.0, num_inference_steps=1, prompt=prompt).images[0]
    
    def make_prompt_and_img(self, good_prompts, instructions, temp, subsamp):

        if "{good_prompts}" not in instructions:
            print("bad! need {good_prompts} in instructions")
            return None
        
        
        good_prompts_subsamp = good_prompts.split("\n")
        if len(good_prompts_subsamp) > subsamp:
            good_prompts_subsamp = list(np.random.choice(good_prompts_subsamp, subsamp, replace=False))
        good_prompts_subsamp = "\n".join(good_prompts_subsamp)
        
        str_prompt = instructions.replace("{good_prompts}", good_prompts_subsamp)
        
        chat_completion = self.gpt4.client.chat.completions.create(
            messages=[
                {"role": "user", "content": str_prompt},
            ],
            model=self.gpt4.model,
            temperature = temp,
            stop = "\n",
        )
        current_prompt = chat_completion.choices[0].message.content
        print(f"generated prompt: {current_prompt}")
        img0 = self.generate_img(current_prompt)
        # img1 = self.generate_img(prompt)
        self.current_prompt = current_prompt
        return img0, current_prompt, good_prompts

    def save_prompt_and_gen_next(self, good_prompts, instructions, temp, subsamp):
        good_prompts += f"\n{self.current_prompt}"
        with open(self.filename, "w") as file:
            file.write(good_prompts)
    
        return self.make_prompt_and_img(good_prompts, instructions, temp, subsamp)
        



if __name__ == "__main__":
    
    # width = 786
    # height = 512
    num_inference_steps = 1
    
    pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16").to("cuda")
    
    gh = GradioHolder(pipe)
    
    
    

    
    with gr.Blocks() as demo:
        
        with gr.Row():
            with gr.Column():
                img0 = gr.Image(label="seed1")
            with gr.Column():
                timestamp_file = gr.Textbox(label="output filename", value=gh.filename, interactive=False)
                current_prompt = gr.Textbox(label="last prompt made by GPT4", interactive=False)
                
                b_bad = gr.Button('dont save & gen next', variant='primary')
                b_good = gr.Button('save & gen next!', variant='primary')
                temp = gr.Slider(label="temperature", value=0.7, minimum=0.0, maximum=2.0, step=0.01)
                subsamp = gr.Slider(label="subsample good_prompts", value=3, minimum=1, maximum=6, step=1)
            
        with gr.Row():
            instructions = gr.Textbox(label="instructions. use {good_prompts}", value="You are a brilliant prompt engineer. Here is an example of your work: {good_prompts}. Make one more like that, but stay with the theme. Directly give the next prompt! make it super trippy and weird. keep it short however and vary within the theme, be creative and not repetitive!")
        with gr.Row():
            good_prompts = gr.Textbox(label="good_prompts", value="macro photo of an insect\nsuper high resolution photo of a weird bug")


        b_bad.click(gh.make_prompt_and_img, inputs=[good_prompts, instructions, temp, subsamp], outputs=[img0, current_prompt, good_prompts])
        
        b_good.click(gh.save_prompt_and_gen_next, inputs=[good_prompts, instructions, temp, subsamp], outputs=[img0, current_prompt, good_prompts])
        
    

    demo.launch(share=gh.share, inbrowser=True, inline=False, server_name="10.40.49.100")
