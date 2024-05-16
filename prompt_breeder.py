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
            gpt_model,
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
        self.gpt4 = lt.GPT4(model=gpt_model)
        self.current_prompt = ""
        self.start_time = datetime.now()
        self.filename = "promptbreeder_" + self.start_time.strftime("%y%m%d_%H%M")
        self.idx_prompt = 0 
        self.width = 512
        self.height = 512


    def generate_img(self, prompt):
        return self.pipe(guidance_scale=0.0, num_inference_steps=1, prompt=prompt, width=self.width, height=self.height).images[0]
    
    def generate_prompts(self, good_prompts, instructions, temp, subsamp, max_tokens):

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
            max_tokens=max_tokens,
            model=self.gpt4.model,
            temperature = temp,
        )
        generated_prompts = chat_completion.choices[0].message.content
        generated_prompts = self.cleanup_prompts(generated_prompts)
        self.idx_prompt = 0
        return generated_prompts
    
    def cleanup_prompts(self, multiprompt):
        list_generated_prompts = multiprompt.split("\n")
        list_generated_prompts = [l for l in list_generated_prompts if len(l) > 5]
        list_cleaned = []
        for current_prompt in list_generated_prompts:
            idx_first_space = current_prompt.index(" ")
            if idx_first_space < 3:
                current_prompt = current_prompt[idx_first_space+1:]
            list_cleaned.append(current_prompt)
        multiprompt = "\n".join(list_cleaned)
        return multiprompt
        
    
    def reject(self, generated_prompts, good_prompts):
        return self.generate_next_img(generated_prompts, good_prompts)
    
    def accept(self, generated_prompts, good_prompts):
        good_prompts += f"\n{self.current_prompt}"
        with open(self.filename+".txt", "w") as file:
            file.write(good_prompts)
            
        return self.generate_next_img(generated_prompts, good_prompts)
    
    def generate_next_img(self, generated_prompts, good_prompts):
        list_generated_prompts = generated_prompts.split("\n")
        if self.idx_prompt >= len(list_generated_prompts):
            img0 = Image.new('RGB', (512, 512), (0, 0, 0))
            current_prompt = "WE REACHED THE LAST PROMPT!!!"
            return img0, current_prompt, good_prompts
        else:
            current_prompt = list_generated_prompts[self.idx_prompt]

        
        print(f"making image with prompt: {current_prompt}")
        img0 = self.generate_img(current_prompt)
        self.current_prompt = current_prompt
        self.idx_prompt += 1
        return img0, current_prompt, good_prompts
    
        

    def save_prompt_and_gen_next(self, good_prompts, instructions, temp, subsamp):
        good_prompts += f"\n{self.current_prompt}"
        with open(self.filename, "w") as file:
            file.write(good_prompts)
    
        return self.make_prompt_and_img(good_prompts, instructions, temp, subsamp)
        



if __name__ == "__main__":
    
    # Creating the parser
    parser = argparse.ArgumentParser(description="Process the server IP.")
    
    # Adding the server_ip argument
    parser.add_argument("--server_ip", type=str, help="The IP address of the server", default=None)
    
    # Parsing the arguments
    args = parser.parse_args()
    
    
    
    width = 1024
    # width = 512
    height = 512
    num_inference_steps = 1
    gpt_model = "gpt-4"
    
    pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16").to("cuda")
    gh = GradioHolder(pipe, gpt_model=gpt_model)
    gh.width = width
    gh.height = height
    
    txt_instructions = "You are a brilliant prompt engineer. Here is an example of your work: {good_prompts}. Make a list or more prompts like that, be creative and not repetitive! Don't make a numbered list, just give me the descriptions, without any formatting elements, do it JUST LIKE IN THE EXAMPLES!"
    
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                img0 = gr.Image(label="seed1")
            with gr.Column():
                b_render = gr.Button('just render current prompt')
                b_gpt = gr.Button('generate prompts', variant='primary')
                b_bad = gr.Button('reject & check next', variant='primary')
                b_good = gr.Button('accept & check next', variant='primary')
                temp = gr.Slider(label="temperature", value=0.7, minimum=0.0, maximum=2.0, step=0.01)
                subsamp = gr.Slider(label="subsample good_prompts", value=3, minimum=1, maximum=15, step=1)
                max_tokens = gr.Slider(label="max tokens", value=800, minimum=100, maximum=3000, step=100)
                output_file = gr.Textbox(label="output filename", value=gh.filename, interactive=True)
                current_prompt = gr.Textbox(label="current prompt", interactive=True)
            
        with gr.Row():
            instructions = gr.Textbox(label="instructions. use {good_prompts}", value=txt_instructions)
        with gr.Row():
            good_prompts = gr.Textbox(label="good_prompts", value="macro photo of an insect\nsuper high resolution photo of a weird bug")
        with gr.Row():
            generated_prompts = gr.Textbox(label="generated_prompts", value="", interactive=False)

        b_render.click(gh.generate_img, inputs=[current_prompt], outputs=[img0])
        b_gpt.click(gh.generate_prompts, inputs=[good_prompts, instructions, temp, subsamp, max_tokens], outputs=[generated_prompts])
        b_bad.click(gh.reject, inputs=[generated_prompts, good_prompts], outputs=[img0, current_prompt, good_prompts])
        b_good.click(gh.accept, inputs=[generated_prompts, good_prompts], outputs=[img0, current_prompt, good_prompts])
        
    
    if args.server_ip is None:
        demo.launch(share=gh.share, inbrowser=True, inline=False)
    else:
        demo.launch(share=gh.share, inbrowser=True, inline=False, server_name=args.server_ip)
        
