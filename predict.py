import sys

sys.path.append("/CLIP")
sys.path.append("/taming-transformers")
sys.path.append("/k-diffusion")
# Slightly modified version of: https://github.com/CompVis/stable-diffusion/blob/main/scripts/txt2img.py
import os
import sys
import time
import math
# Code to turn kwargs into Jupyter widgets
from collections import OrderedDict
from contextlib import contextmanager, nullcontext
from glob import glob
from time import time

import librosa
import numpy as np
import torch
from cog import BasePredictor, Input, Path
from einops import rearrange, repeat
from googletrans import Translator
from helpers import sampler_fn, save_samples
from k_diffusion import sampling
from k_diffusion.external import CompVisDenoiser
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
from PIL import Image
from pytorch_lightning import seed_everything
from scripts.txt2img import chunk, load_model_from_config
from torch import autocast
#from tqdm.auto import tqdm, trange  # NOTE: updated for notebook
from tqdm import tqdm, trange  # NOTE: updated for notebook
from typing import Iterator


def get_amplitude_envelope(signal, hop_length):
    """Calculate the amplitude envelope of a signal with a given frame size nad hop length."""
    amplitude_envelope = []
    
    # calculate amplitude envelope for each frame
    for i in range(0, len(signal), hop_length): 
        amplitude_envelope_current_frame = max(np.abs(signal[i:i+hop_length]) )
        amplitude_envelope.append(amplitude_envelope_current_frame)
    
    return np.array(amplitude_envelope)  

class Predictor(BasePredictor):


    @torch.inference_mode()
    def setup(self):
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        options = get_default_options()
        self.options = options

        self.model = load_model(self.options, self.device)
        self.model_wrap = CompVisDenoiser(self.model)

        self.translator= Translator()

    @torch.inference_mode()
    def predict(
        self,
        prompts: str = Input(
            default="""A painting of a moth
A painting of a killer dragonfly by paul klee, intricate detail
Two fishes talking to eachother in deep sea, art by hieronymus bosch"""),
        style_suffix: str = Input(
            default="by paul klee, intricate details",
            description="Style suffix to add to the prompt. This can be used to add the same style to each prompt.",
        ),
        audio_file: Path = Input(
            default=None, 
            description="input audio file"),
        prompt_scale: float = Input(
            default=15.0,
            description="Determines influence of your prompt on generation.",
        ),
        random_seed: int = Input(
            default=13,
            description="Each seed generates a different image",
        ),
        diffusion_steps: int = Input(
            default=20,
            description="Number of diffusion steps. Higher steps could produce better results but will take longer to generate. Maximum 30 (using K-Euler-Diffusion).",
        ),
        audio_smoothing: float = Input(
            default=0.8,
            description="Audio smoothing factor.",
        ),
        audio_noise_scale: float = Input(
            default=0.3,
            description="Larger values mean audio will lead to bigger changes in the image.",
        ),
        audio_loudness_type: str = Input(
            default="peak",
            description="Type of loudness to use for audio. Options are 'rms' or 'peak'.",
            choices=["rms", "peak"],
        ),
        frame_rate: float = Input(
            default=16,
            description="Frames per second for the generated video.",
        ),
        width: int = Input(
            default=384,
            description="Width of the generated image. The model was really only trained on 512x512 images. Other sizes tend to create less coherent images.",
        ),
        height: int = Input(
            default=512,
            description="Height of the generated image. The model was really only trained on 512x512 images. Other sizes tend to create less coherent images.",
        ),
        batch_size: int = Input(
            default=24,
            description="Number of images to generate at once. Higher batch sizes will generate images faster but will use more GPU memory i.e. not work depending on resolution.",
        ),
        frame_interpolation: bool = Input(
            default=True,
            description="Whether to interpolate between frames using FFMPEG or not.",
        )
    ) -> Iterator[Path]:

        start_time = time()

        init_image = None
        init_image_strength = 0.7

        os.system("rm -r ./outputs")
        os.system("mkdir -p  ./outputs")

        if init_image is not None:
            init_image = str(init_image)
            print("using init image", init_image)
        
        # num_frames_per_prompt = abs(min(num_frames_per_  prompt, 15))
        
        # add style suffix to each prompt
        prompts = [prompt + "." + style_suffix for prompt in prompts.split("\n")]


        # start with only style prompt
        prompts = [style_suffix] + prompts

        options = self.options
        options['prompts'] = prompts
        options['prompts'] = [self.translator.translate(prompt.strip()).text for prompt in options['prompts'] if prompt.strip()]
        print("translated prompts", options['prompts'])
        options['n_samples'] = batch_size
        options['audio_noise_scale'] = audio_noise_scale
        
        options['scale'] = prompt_scale
        options['seed'] = random_seed
        options['H'] = height
        options['W'] = width
        options['steps'] = diffusion_steps
        options['init_image'] = init_image
        options['init_image_strength'] = init_image_strength
        options['audio_smoothing'] = audio_smoothing
       
        y, sr = librosa.load(audio_file, sr=22050)
        print("using audio file", audio_file)
        # calculate hop length based on frame rate
        hop_length = int(22050 / frame_rate)
        print("hop length", hop_length, "audio length", len(y), "audio sr", sr)
        
        if audio_loudness_type == "peak":
            # get amplitude envelope
            amplitude_envelope = get_amplitude_envelope(y, hop_length)
            # normalize
            options["audio_intensities"] = amplitude_envelope / amplitude_envelope.max()
        else:
            # get rms
            rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop_length)
            # get amplitude envelope

            # normalize
            options["audio_intensities"] = rms[0] / rms[0].max()

        print("length of audio intensities", len(options["audio_intensities"]))
        audio_length = len(options["audio_intensities"])
        num_prompts = len(options['prompts'])

        num_frames_per_prompt = audio_length // max(1,(num_prompts-1))
        
        print("num frames per prompt", num_frames_per_prompt)
        options['num_interpolation_steps'] = num_frames_per_prompt

        precision_scope = autocast if options.precision=="autocast" else nullcontext
        with precision_scope("cuda"):
            for image_path in run_inference(options, self.model, self.model_wrap, self.device):
                yield Path(image_path)



        #if num_frames_per_prompt == 1:
        #    return Path(options['output_path'])     
        encoding_options = "-c:v libx264 -crf 20 -preset slow -vf format=yuv420p -c:a aac -movflags +faststart"
        os.system("ls -l ./outputs")

        # calculate the frame rate of the video so that the length is always 8 seconds
        
        os.system("nvidia-smi")

        end_time = time()
        audio_options = ""
        if audio_file is not None:
            audio_options = f"-i {audio_file} -map 0:v -map 1:a -shortest"
        

        print("total time", end_time - start_time)
        
        os.system(f'ffmpeg -y -r {frame_rate} -i {options["outdir"]}/%*.png  {audio_options} {encoding_options} /tmp/z_interpollation.mp4')




        # print(f'ffmpeg -y -r {frame_rate} -i {options["outdir"]}/%*.png {audio_options} ${frame_interpolation_flag} {encoding_options} /tmp/z_interpollation.mp4')

        yield Path("/tmp/z_interpollation.mp4")

        if frame_interpolation:
            # convert previously generated video to 54 fps
            os.system(f'ffmpeg -y -i /tmp/z_interpollation.mp4 -filter:v "minterpolate=\'fps=60\'" {encoding_options} /tmp/z_interpollation_60fps.mp4')
            yield Path("/tmp/z_interpollation_40fps.mp4")



def load_model(opt,device):
    """Seperates the loading of the model from the inference"""

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")

    if opt.precision == "autocast":
        model = model.half()

    model = model.to(device)
    
    return model

def slerp(t, v0, v1, DOT_THRESHOLD=0.9995, nonlinear=False):
    """ helper function to spherically interpolate two arrays v1 v2 """


    if nonlinear:
        # a smooth function that goes from 0 to 1 but grows quickly and then slows down
        t = 1 - math.exp(-(t+0.025) * 15)
    
    if not isinstance(v0, np.ndarray):
        inputs_are_torch = True
        input_device = v0.device
        v0 = v0.cpu().numpy()
        v1 = v1.cpu().numpy()

    dot = np.sum(v0 * v1 / (np.linalg.norm(v0) * np.linalg.norm(v1)))
    if np.abs(dot) > DOT_THRESHOLD:
        v2 = (1 - t) * v0 + t * v1
    else:
        theta_0 = np.arccos(dot)
        sin_theta_0 = np.sin(theta_0)
        theta_t = theta_0 * t
        sin_theta_t = np.sin(theta_t)
        s0 = np.sin(theta_0 - theta_t) / sin_theta_0
        s1 = sin_theta_t / sin_theta_0
        v2 = s0 * v0 + s1 * v1

    if inputs_are_torch:
        v2 = torch.from_numpy(v2).to(input_device)

    return v2



def run_inference(opt, model, model_wrap, device):
    """Seperates the loading of the model from the inference
    
    Additionally, slightly modified to display generated images inline
    """
    seed_everything(opt.seed)

    # if opt.plms:
    #     sampler = PLMSSampler(model)
    # else:
    #     sampler = DDIMSampler(model)

    outpath = opt.outdir
    os.makedirs(outpath, exist_ok=True)

    batch_size = opt.n_samples
    prompts = opt.prompts

    
    # add first prompt to end to create a video for single prompts or no inteprolations
    single_prompt = False
    if len(prompts) == 1:
        single_prompt = True
        prompts = prompts + [prompts[0]]


    if (not single_prompt) and (opt.num_interpolation_steps == 1):
        prompts = prompts + [prompts[-1]]

    print("embedding prompts")
    precision_scope = autocast if opt.precision=="autocast" else nullcontext
    with precision_scope("cuda"):
        datas =[model.get_learned_conditioning(prompt) for prompt in prompts]

    print("prompt 0 shape", datas[0].shape)

    os.makedirs(outpath, exist_ok=True)
    
    start_code_a = None
    start_code_b = None
    





    start_code_a = torch.randn([1, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)
    start_code_b = torch.randn([1, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)
    
    start_codes = []
    smoothed_intensity = 0      
    for audio_intensity in opt.audio_intensities:
        smoothed_intensity =  (smoothed_intensity * (opt.audio_smoothing)) + (audio_intensity * (1-opt.audio_smoothing))
        noise_t = smoothed_intensity * opt.audio_noise_scale
        start_code = slerp(float(noise_t), start_code_a, start_code_b)
        start_codes.append(start_code)


    interpolated_prompts = []
    for data_a,data_b in zip(datas,datas[1:]):         
        interpolated_prompts = interpolated_prompts + [slerp(float(t), data_a, data_b, nonlinear=True) for t in np.linspace(0, 1, opt.num_interpolation_steps)]

    print("len smoothed_audio_intensities",len(start_codes), "len interpolated_prompts",len(interpolated_prompts))

    print("interp prompts 0 shape", interpolated_prompts[0].shape, "start_codes 0 shape", start_codes[0].shape)


    with torch.no_grad():
        with model.ema_scope():
            # chunk interpolated_prompts into batches
            for i in range(0, len(interpolated_prompts), batch_size):
                data_batch = torch.cat(interpolated_prompts[i:i+batch_size])
                start_code_batch = torch.cat(start_codes[i:i+batch_size])

                print("data_batch",data_batch.shape, "start_code_batch",start_code_batch.shape)
                images = diffuse(start_code_batch, data_batch, len(data_batch), opt, model, model_wrap, device)
                

                for i2, image in enumerate(images):
                    image_path = os.path.join(outpath, f"{i+i2:05}.png")
                    image.save(image_path)
                    print(f"Saved {image_path}")

                    if i2 == len(images)-1:
                        yield image_path




    print(f"Your samples have been saved to: \n{outpath} \n"
          f" \nEnjoy.")




def diffuse(start_code, c, batch_size, opt, model, model_wrap,  device):
    precision_scope = autocast if opt.precision=="autocast" else nullcontext
    with precision_scope("cuda"):
        #print("diffusing with batch size", batch_size)
        uc = None
        if opt.scale != 1.0:
            uc = model.get_learned_conditioning(batch_size * [""])
        try:
            samples = sampler_fn(
                c=c,
                uc=uc,
                args=opt,
                model_wrap=model_wrap,
                init_latent=start_code,
                device=device,
                # cb=callback
                )
        except:
            print("diffuse failed. returning empty list.")
            return []
        print("samples_ddim", samples.shape)
        x_samples = model.decode_first_stage(samples)
        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
        images = []
        for x_sample in x_samples:
            x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
            images.append(Image.fromarray(x_sample.astype(np.uint8))) # .save(image_path)
        return images



class WidgetDict2(OrderedDict):
    def __getattr__(self,val):
        return self[val]


def get_default_options():
    options = WidgetDict2()
    options['outdir'] ="./outputs"
    options['precision'] = "autocast"
    options['sampler'] = "euler"
    options['ddim_steps'] = 50
    options['plms'] = True
    options['ddim_eta'] = 0.0
    options['n_iter'] = 1
    options['C'] = 4
    options['f'] = 8
    options['n_samples'] = 1
    options['n_rows'] = 0
    options['from_file'] = None
    options['config'] = "configs/stable-diffusion/v1-inference.yaml"
    options['ckpt'] ="/stable-diffusion-checkpoints/v1-5-pruned-emaonly.ckpt"
    options['use_init'] = True
    # Extra option for the notebook
    options['display_inline'] = False
    options["audio_intensities"] = None
    return options


# bs 8: 77.5s
# bs 1: 160s
