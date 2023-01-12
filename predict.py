import sys

sys.path.append("/CLIP")
sys.path.append("/taming-transformers")
sys.path.append("/k-diffusion")
# Slightly modified version of: https://github.com/CompVis/stable-diffusion/blob/main/scripts/txt2img.py
import os
import sys
import time
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

count_image_save = 0

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
            default="""A lively and whimsical apothecary where chrome robots shop grows from the stalk of a giant mushroom, cgsociety, siggraph, oleg oprisco, conrad roset, anka zhuravleva, gediminas pranckevicius
        A lively and whimsical dark apothecary shop, cinematic framing, rain lit, chrome robots on single wheels shop, the shop grows from the stalk of a giant mushroom, cgsociety, siggraph, dystopian scifi, concept art, set design, oleg oprisco, conrad roset, anka zhuravleva, gediminas pranckevicius, cornell, kawasaki
        Surreal gouache painting, by yoshitaka amano, by ruan jia, by conrad roset, by kilian eng, by good smile company, detailed anime 3 d render of floating molecules and a robot artist holding an icosahedron with stars, clouds, and rainbows in the background, cgsociety, artstation, modular patterned mechanical costume and headpiece, retrowave atmosphere"""        ),
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
            default=0.7,
            description="Audio smoothing factor.",
        ),
        frame_rate: int = Input(
            default=12,
            description="Frames per second for the generated video.",
        ),
        width: int = Input(
            default=512,
            description="Width of the generated image. The model was really only trained on 512x512 images. Other sizes tend to create less coherent images.",
        ),
        height: int = Input(
            default=512,
            description="Height of the generated image. The model was really only trained on 512x512 images. Other sizes tend to create less coherent images.",
        ),
        batch_size: int = Input(
            default=4,
            description="Batch size. Higher values will result in quicker generation but may cause memory problems depending on the resolution.",
        )
    ) -> Path:
        global count_image_save
        count_image_save = 0

        start_time = time()
        init_image = None
        init_image_strength = 0.7

        os.system("rm -r ./outputs")
        os.system("mkdir -p  ./outputs")

        if init_image is not None:
            init_image = str(init_image)
            print("using init image", init_image)
        


        # num_frames_per_prompt = abs(min(num_frames_per_prompt, 15))
        
        options = self.options
        options['prompts'] = prompts.split("\n")
        options['prompts'] = [self.translator.translate(prompt.strip()).text for prompt in options['prompts'] if prompt.strip()]
        print("translated prompts", options['prompts'])

        
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
        # get rms
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop_length)
        # normalize
        options["audio_intensities"] = rms[0] / rms[0].max()

        print("length of audio intensities", len(options["audio_intensities"]))
        audio_length = len(options["audio_intensities"])
        num_prompts = len(options['prompts'])

        num_frames_per_prompt = audio_length // max((1,(num_prompts-1)))
        
        print("num frames per prompt", num_frames_per_prompt)
        options['num_interpolation_steps'] = num_frames_per_prompt

        with torch.autocast("cuda"):
            run_inference(options, self.model, self.model_wrap, self.device, batch_size)

        #if num_frames_per_prompt == 1:
        #    return Path(options['output_path'])     
        encoding_options = "-c:v libx264 -crf 20 -preset slow -vf format=yuv420p -c:a aac -movflags +faststart"
        os.system("ls -l ./outputs")

        # calculate the frame rate of the video so that the length is always 8 seconds
        
        

        end_time = time()

        audio_options = ""
        if audio_file is not None:
            audio_options = f"-i {audio_file} -map 0:v -map 1:a -shortest"
        os.system(f'ffmpeg -y -r {frame_rate} -i {options["outdir"]}/%*.png {audio_options} {encoding_options} -r {frame_rate} /tmp/z_interpollation.mp4')
        
        os.system(f"ls -l {options['outdir']}/")
        print("total time", end_time - start_time)
        return Path("/tmp/z_interpollation.mp4")


def load_model(opt,device):
    """Seperates the loading of the model from the inference"""
    
    # if opt.laion400m:
    #     print("Falling back to LAION 400M model...")
    #     opt.config = "configs/latent-diffusion/txt2img-1p4B-eval.yaml"
    #     opt.ckpt = "models/ldm/text2img-large/model.ckpt"
    #     opt.outdir = "outputs/txt2img-samples-laion400m"

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")

    model = model.half().to(device)
    # model = model.to(device)
    
    return model

def slerp(t, v0, v1, DOT_THRESHOLD=0.9995):
    """ helper function to spherically interpolate two arrays v1 v2 """

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




def run_inference(opt, model, model_wrap, device, accumulate_batch_size=4):
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
    cs = [model.get_learned_conditioning(prompt) for prompt in prompts]

    datas = [[batch_size * c] for c in cs] 

    os.makedirs(outpath, exist_ok=True)
    
    base_count = 0
    
    start_code_a = None
    start_code_b = None
    




    if opt.init_image:
        init_image = load_img(opt.init_image, shape=(opt.W, opt.H)).to(device)
        init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
        start_code_a = model.get_first_stage_encoding(model.encode_first_stage(init_image))     
        start_code_b = start_code_a
    else:
        start_code_a = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)
        start_code_b = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)
    audio_intensity = 0

    precision_scope = autocast if opt.precision=="autocast" else nullcontext

    # If more than one prompt we only interpolate the text conditioning
    if not single_prompt and opt.audio_intensities is None:
        start_code_b = start_code_a

    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                for data_a,data_b in zip(datas,datas[1:]):          
                    for t in np.linspace(0, 1, opt.num_interpolation_steps):
                        #print("data_a",data_a)

                        data = [slerp(float(t), data_a[0], data_b[0])]
                        
                        t_max = 0.15 #min((1, opt.num_interpolation_steps / 10))

                        if opt.audio_intensities is not None:
                            if base_count >= len(opt.audio_intensities):
                                print("end of audio file reached. returning")
                                return
                            audio_intensity = (audio_intensity * (opt.audio_smoothing)) + (opt.audio_intensities[base_count] * (1-opt.audio_smoothing))
                            noise_t = audio_intensity * t_max
                        else:
                            noise_t = t * t_max 
                        # calculate interpolation for init noise. this only applies if we have only on text prompt
                        # otherwise noise stays constant for now (due to start_code_a == start_code_b)
                        
                        opt["scale"] = audio_intensity * 2+17
                                                
                
                        start_code = slerp(float(noise_t), start_code_a, start_code_b) #slerp(audio_intensity, start_code_a, start_code_b)
                        for c in data:
                            diffuse(accumulate_batch_size, start_code, c, opt, model, model_wrap, outpath, device)
                            base_count += 1
            # make sure to diffuse last incomplete batch
            diffuse_remaining_batch(opt, model, model_wrap, outpath, device)




    print(f"Your samples have been saved to: \n{outpath} \n"
          f" \nEnjoy.")




accumulated_start_code = None
accumulated_c = None

def diffuse(accumulate_batch, start_code, c, opt, model, model_wrap, outpath, device):
    global accumulated_start_code, accumulated_c

    if accumulated_start_code is None:
        accumulated_start_code = start_code
        accumulated_c = c
    else:
        # acccumulate along axis 0
        accumulated_start_code = torch.cat((accumulated_start_code, start_code), axis=0)
        accumulated_c = torch.cat((accumulated_c, c), axis=0)
    
    if len(accumulated_start_code) < accumulate_batch:
        return
    
    diffuse_batch(accumulated_start_code, accumulated_c, len(accumulated_start_code), opt, model, model_wrap, outpath, device)

    accumulated_start_code = None
    accumulated_c = None

def diffuse_remaining_batch(opt, model, model_wrap, outpath, device):
    global accumulated_start_code, accumulated_c

    if accumulated_start_code is None:
        return
    
    diffuse_batch(accumulated_start_code, accumulated_c, len(accumulated_start_code), opt, model, model_wrap, outpath, device)

    accumulated_start_code = None
    accumulated_c = None

def diffuse_batch(start_code, c, batch_size, opt, model, model_wrap, outpath, device):
    global count_image_save
    print("diffusing with batch size", batch_size, "start code shape", start_code.shape, "c shape", c.shape)
    uc = None
    if opt.scale != 1.0:
        uc = model.get_learned_conditioning(batch_size * [""])

    t_enc = 0
    if opt.init_image is not None:
        t_enc = round(opt.steps * (1.0 - opt.init_image_strength))
    print("using init image", opt.init_image, "for", t_enc, "steps")
    #if args.sampler in ["klms","dpm2","dpm2_ancestral","heun","euler","euler_ancestral"]:


    samples = sampler_fn(
        c=c,
        uc=uc,
        args=opt,
        model_wrap=model_wrap,
        init_latent=start_code,
        t_enc=t_enc,
        device=device,
        # cb=callback
        )
    # samples, _ = sampler.sample(S=opt.ddim_steps,
    #                                 conditioning=c,
    #                                 batch_size=batch_size,
    #                                 shape=shape,
    #                                 verbose=False,
    #                                 unconditional_guidance_scale=opt.scale,
    #                                 unconditional_conditioning=uc,
    #                                 eta=opt.ddim_eta,
    #                                 x_T=start_code)   
    print("samples_ddim", samples.shape)
    x_samples = model.decode_first_stage(samples)
    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
    if not opt.skip_save:
        for x_sample in x_samples:
            x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
            image_path = os.path.join(outpath, f"{count_image_save:05}.png")
            prompt_path = os.path.join(outpath, f"{count_image_save:05}.txt")
            Image.fromarray(x_sample.astype(np.uint8)).save(image_path)
            print("saved", image_path)
            count_image_save += 1




class WidgetDict2(OrderedDict):
    def __getattr__(self,val):
        return self[val]


def get_default_options():
    options = WidgetDict2()
    options['outdir'] ="./outputs"
    options['sampler'] = "euler_ancestral"
    options['skip_save'] = False
    options['ddim_steps'] = 50
    options['plms'] = True
    options['laion400m'] = False
    options['ddim_eta'] = 0.0
    options['n_iter'] = 1
    options['C'] = 4
    options['f'] = 8
    options['n_samples'] = 1
    options['n_rows'] = 0
    options['from_file'] = None
    options['config'] = "configs/stable-diffusion/v1-inference.yaml"
    options['ckpt'] ="/stable-diffusion-checkpoints/v1-5-pruned-emaonly.ckpt"
    options['precision'] = "full"  # or "full" "autocast"
    options['use_init'] = True
    # Extra option for the notebook
    options['display_inline'] = False
    options["audio_intensities"] = None
    return options


def load_img(path, shape):
    if path.startswith('http://') or path.startswith('https://'):
        image = Image.open(requests.get(path, stream=True).raw).convert('RGB')
    else:
        image = Image.open(path).convert('RGB')

    image = image.resize(shape, resample=Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2. * image - 1.


# batch size 1: 98.12
# batch size 4: 26.34
# batch size 1: 580.77
# batch size 4: 511.64