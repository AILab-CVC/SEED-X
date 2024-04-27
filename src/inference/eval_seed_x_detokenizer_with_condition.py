import hydra
from omegaconf import OmegaConf
import torch
import os
import pyrootutils
from PIL import Image
from diffusers import AutoencoderKL, UNet2DConditionModel, EulerDiscreteScheduler

pyrootutils.setup_root(__file__, indicator='.project-root', pythonpath=True)

condition_image_path = 'demo_images/men_condition.jpg'
target_image_path = 'demo_images/men.jpg'
save_path = 'vis/men_recon_with_condition.jpg'

device = 'cuda'
dtype = torch.float16

adapter_cfg_path = 'configs/sdxl_adapter/sdxl_qwen_vit_resampler_l4_q64_full_with_latent_image_pretrain_no_normalize.yaml'
discrete_model_cfg_path = 'configs/discrete_model/discrete_identity.yaml'
visual_encoder_cfg_path = 'configs/visual_encoder/qwen_vitg_448.yaml'
image_transform_cfg_path = 'configs/processer/qwen_448_transform.yaml'

image_transform_cfg = OmegaConf.load(image_transform_cfg_path)
adapter_cfg = OmegaConf.load(adapter_cfg_path)
visual_encoder_cfg = OmegaConf.load(visual_encoder_cfg_path)
discrete_model_cfg = OmegaConf.load(discrete_model_cfg_path)
image_transform_cfg = OmegaConf.load(image_transform_cfg_path)

diffusion_model_path = 'pretrained/stable-diffusion-xl-base-1.0'

noise_scheduler = EulerDiscreteScheduler.from_pretrained(diffusion_model_path, subfolder="scheduler")
tokenizer = None
text_encoder = None
print('init vae')
vae = AutoencoderKL.from_pretrained(diffusion_model_path, subfolder="vae").to(device, dtype=dtype)
print('init unet')
unet = UNet2DConditionModel.from_pretrained(diffusion_model_path, subfolder="unet").to(device, dtype=dtype)

print('init discrete model')
discrete_model = hydra.utils.instantiate(discrete_model_cfg).to(device).eval()

print('init ip adapter')
adapter = hydra.utils.instantiate(adapter_cfg, unet=unet).to(device, dtype=dtype).eval()

print('init visual encoder')
visual_encoder = hydra.utils.instantiate(visual_encoder_cfg).to(device).eval()

image_transform = hydra.utils.instantiate(image_transform_cfg)
print('init done')

adapter.init_pipe(vae=vae,
                  scheduler=noise_scheduler,
                  visual_encoder=visual_encoder,
                  image_transform=image_transform,
                  dtype=dtype,
                  device=device)

condition_image = Image.open(condition_image_path).convert('RGB')
condition_image  = condition_image.resize((1024, 1024))

target_image = Image.open(target_image_path).convert('RGB')
generated_images = adapter.generate(target_image, latent_image=condition_image, num_inference_steps=50)

generated_images[0].save(save_path)
