import hydra
import torch
import os
import pyrootutils
from PIL import Image
from omegaconf import OmegaConf
from diffusers import AutoencoderKL, UNet2DConditionModel, EulerDiscreteScheduler


pyrootutils.setup_root(__file__, indicator='.project-root', pythonpath=True)

BOI_TOKEN = '<img>'
EOI_TOKEN = '</img>'
IMG_TOKEN = '<img_{:05d}>'

device = 'cuda'
device_2 = 'cuda'
dtype = torch.float16
dtype_str = 'fp16'
num_img_in_tokens = 64
num_img_out_tokens = 64

gen_prompt = '{caption}<img>'

tokenizer_cfg_path = 'configs/tokenizer/clm_llama_tokenizer_224loc_anyres.yaml'
image_transform_cfg_path = 'configs/processer/qwen_448_transform.yaml'
visual_encoder_cfg_path = 'configs/visual_encoder/qwen_vitg_448.yaml'
llm_cfg_path = 'configs/clm_models/llm_seed_x.yaml'
agent_cfg_path = 'configs/clm_models/agent_seed_x.yaml'
adapter_cfg_path = 'configs/sdxl_adapter/sdxl_qwen_vit_resampler_l4_q64_pretrain_no_normalize.yaml'
discrete_model_cfg_path = 'configs/discrete_model/discrete_identity.yaml'

diffusion_model_path = 'pretrained/stable-diffusion-xl-base-1.0'

save_dir = 'vis'
os.makedirs(save_dir, exist_ok=True)

tokenizer_cfg = OmegaConf.load(tokenizer_cfg_path)
tokenizer = hydra.utils.instantiate(tokenizer_cfg)

image_transform_cfg = OmegaConf.load(image_transform_cfg_path)
image_transform = hydra.utils.instantiate(image_transform_cfg)

visual_encoder_cfg = OmegaConf.load(visual_encoder_cfg_path)
visual_encoder = hydra.utils.instantiate(visual_encoder_cfg)
visual_encoder.eval().to(device_2, dtype=dtype)
print('Init visual encoder done')

llm_cfg = OmegaConf.load(llm_cfg_path)
llm = hydra.utils.instantiate(llm_cfg, torch_dtype=dtype)
print('Init llm done.')

agent_model_cfg = OmegaConf.load(agent_cfg_path)
agent_model = hydra.utils.instantiate(agent_model_cfg, llm=llm)

agent_model.eval().to(device, dtype=dtype)
print('Init agent mdoel Done')

noise_scheduler = EulerDiscreteScheduler.from_pretrained(diffusion_model_path, subfolder="scheduler")
print('init vae')
vae = AutoencoderKL.from_pretrained(diffusion_model_path, subfolder="vae").to(device_2, dtype=dtype)
print('init unet')
unet = UNet2DConditionModel.from_pretrained(diffusion_model_path, subfolder="unet").to(device_2, dtype=dtype)

adapter_cfg = OmegaConf.load(adapter_cfg_path)
adapter = hydra.utils.instantiate(adapter_cfg, unet=unet).to(device_2, dtype=dtype).eval()

discrete_model_cfg = OmegaConf.load(discrete_model_cfg_path)
discrete_model = hydra.utils.instantiate(discrete_model_cfg).to(device_2).eval()
print('Init adapter done')

adapter.init_pipe(vae=vae,
                  scheduler=noise_scheduler,
                  visual_encoder=visual_encoder,
                  image_transform=image_transform,
                  discrete_model=discrete_model,
                  dtype=dtype,
                  device=device_2)

print('Init adapter pipe done')

caption = 'A super math wizard cat, richly textured oil painting.'
prompt = gen_prompt.format_map({'caption': caption})
prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
input_ids = torch.tensor([tokenizer.bos_token_id] + prompt_ids).to(device, dtype=torch.long).unsqueeze(0)
output = agent_model.generate(tokenizer=tokenizer, input_ids=input_ids, num_img_gen_tokens=num_img_out_tokens)
print(output['has_img_output'])
print(output['text'])

if output['has_img_output']:
    images = adapter.generate(image_embeds=output['img_gen_feat'].to(device_2), num_inference_steps=50)
    save_path = os.path.join(save_dir, caption.replace('.', '') + '.png')
    images[0].save(save_path)
torch.cuda.empty_cache()
