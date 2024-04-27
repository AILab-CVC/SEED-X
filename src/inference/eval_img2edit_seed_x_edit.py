import hydra
import torch
import os
import re
import pyrootutils
from PIL import Image
from omegaconf import OmegaConf
from diffusers import AutoencoderKL, UNet2DConditionModel, EulerDiscreteScheduler, Transformer2DModel
from any_res import process_anyres_image

pyrootutils.setup_root(__file__, indicator='.project-root', pythonpath=True)

BOI_TOKEN = '<img>'
BOP_TOKEN = '<patch>'
EOI_TOKEN = '</img>'
EOP_TOKEN = '</patch>'
IMG_TOKEN = '<img_{:05d}>'

resolution_grids = ['1x1']
base_resolution = 448

device = 'cuda'
dtype = torch.float16
dtype_str = 'fp16'
num_img_in_tokens = 64
num_img_out_tokens = 64
instruction_prompt = '[INST] {instruction} [/INST]\n'

save_dir = 'vis'
os.makedirs(save_dir, exist_ok=True)

tokenizer_cfg_path = 'configs/tokenizer/clm_llama_tokenizer_224loc_anyres.yaml'
image_transform_cfg_path = 'configs/processer/qwen_448_transform.yaml'
visual_encoder_cfg_path = 'configs/visual_encoder/qwen_vitg_448.yaml'
llm_cfg_path = 'configs/clm_models/llm_seed_x_edit.yaml'
agent_cfg_path = 'configs/clm_models/agent_seed_x_edit.yaml'
adapter_cfg_path = 'configs/sdxl_adapter/sdxl_qwen_vit_resampler_l4_q64_full_with_latent_image_pretrain_no_normalize.yaml'
discrete_model_cfg_path = 'configs/discrete_model/discrete_identity.yaml'

diffusion_model_path = 'pretrained/stable-diffusion-xl-base-1.0'

tokenizer_cfg = OmegaConf.load(tokenizer_cfg_path)
tokenizer = hydra.utils.instantiate(tokenizer_cfg)

image_transform_cfg = OmegaConf.load(image_transform_cfg_path)
image_transform = hydra.utils.instantiate(image_transform_cfg)

visual_encoder_cfg = OmegaConf.load(visual_encoder_cfg_path)
visual_encoder = hydra.utils.instantiate(visual_encoder_cfg)
visual_encoder.eval().to(device, dtype=dtype)
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
vae = AutoencoderKL.from_pretrained(diffusion_model_path, subfolder="vae").to(device, dtype=dtype)
print('init unet')
unet = UNet2DConditionModel.from_pretrained(diffusion_model_path, subfolder="unet").to(device, dtype=dtype)

adapter_cfg = OmegaConf.load(adapter_cfg_path)
adapter = hydra.utils.instantiate(adapter_cfg, unet=unet).to(device, dtype=dtype).eval()

discrete_model_cfg = OmegaConf.load(discrete_model_cfg_path)
discrete_model = hydra.utils.instantiate(discrete_model_cfg).to(device).eval()
print('Init adapter done')

adapter.init_pipe(vae=vae,
                scheduler=noise_scheduler,
                visual_encoder=visual_encoder,
                image_transform=image_transform,
                dtype=dtype,
                device=device)

print('Init adapter pipe done')
boi_token_id = tokenizer.encode(BOI_TOKEN, add_special_tokens=False)[0]
eoi_token_id = tokenizer.encode(EOI_TOKEN, add_special_tokens=False)[0]

bop_token_id = tokenizer.encode(BOP_TOKEN, add_special_tokens=False)[0]
eop_token_id = tokenizer.encode(EOP_TOKEN, add_special_tokens=False)[0]

grid_pinpoints = []
for scale in resolution_grids:
    s1, s2 = scale.split('x')
    grid_pinpoints.append([int(s1)*base_resolution, int(s2)*base_resolution])
grid_pinpoints = grid_pinpoints


image_path = 'demo_images/car.jpg'
instruction = 'Make it under the sunset'

image = Image.open(image_path).convert('RGB')
source_image = image.resize((1024, 1024))

image_tensor, patch_pos_tensor = process_anyres_image(image, image_transform, grid_pinpoints, base_resolution)
embeds_cmp_mask = torch.tensor([True]*image_tensor.shape[0]).to(device, dtype=torch.bool)

patch_pos = [patch_pos_tensor]
patch_position = torch.cat(patch_pos, dim=0)

image_tensor = image_tensor.to(device, dtype=dtype)

patch_length = image_tensor.shape[0]
image_tokens = ''
for _ in range(patch_length-1):
    image_tokens +=  BOP_TOKEN + ''.join(IMG_TOKEN.format(int(item)) for item in range(num_img_in_tokens)) + EOP_TOKEN
image_tokens += BOI_TOKEN + ''.join(IMG_TOKEN.format(int(item)) for item in range(num_img_in_tokens)) + EOI_TOKEN

prompt = instruction_prompt.format_map({'instruction': image_tokens + instruction})

input_ids = tokenizer.encode(prompt, add_special_tokens=False)
input_ids = [tokenizer.bos_token_id] + input_ids

input_ids = torch.tensor(input_ids).to(device, dtype=torch.long)

ids_cmp_mask = torch.zeros_like(input_ids, dtype=torch.bool)

boi_indices = torch.where(torch.logical_or(input_ids == boi_token_id, input_ids == bop_token_id))[0].tolist()
eoi_indices = torch.where(torch.logical_or(input_ids == eoi_token_id, input_ids == eop_token_id))[0].tolist()

for boi_idx, eoi_idx in zip(boi_indices, eoi_indices):
    ids_cmp_mask[boi_idx + 1:eoi_idx] = True

input_ids = input_ids.unsqueeze(0)
ids_cmp_mask = ids_cmp_mask.unsqueeze(0)

with torch.no_grad():
    image_embeds = visual_encoder(image_tensor)
    output = agent_model.generate(tokenizer=tokenizer,
                                input_ids=input_ids,
                                image_embeds=image_embeds,
                                embeds_cmp_mask=embeds_cmp_mask,
                                patch_positions=patch_position,
                                ids_cmp_mask=ids_cmp_mask,
                                max_new_tokens=512,
                                num_img_gen_tokens=num_img_out_tokens)
text = re.sub('<[^>]*>', '', output['text'])
print(text)

if output['has_img_output']:
    images = adapter.generate(image_embeds=output['img_gen_feat'], latent_image=source_image, num_inference_steps=50)

    save_path = os.path.join(save_dir, str(len(os.listdir(save_dir))) + '_' + instruction + '.jpg')
    images[0].save(save_path)
torch.cuda.empty_cache()
