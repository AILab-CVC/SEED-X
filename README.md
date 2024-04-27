# SEED-X
[![arXiv](https://img.shields.io/badge/arXiv-2404.14396-b31b1b.svg)](https://arxiv.org/abs/2404.14396)
[![Demo](https://img.shields.io/badge/Gradio-Demo-orange)](https://139a5c1d085953f17b.gradio.live/)
[![Static Badge](https://img.shields.io/badge/Model-Huggingface-yellow)](https://huggingface.co/AILab-CVC/SEED-X-17B/tree/main)

We introduce SEED-X, a unified and versatile foundation model, which can serve as various multimodal AI assistants **in the real world** after different instruction tuning, capable of responding to a variety of user needs through unifying **multi-granularity comprehension and generation**.

All models and inference code are released! 

## News
**2024-04-22** :hugs: We release the [models](https://huggingface.co/AILab-CVC/SEED-X-17B/tree/main) including the pre-trained foundation model **SEED-X**, the general instruction-tuned model **SEED-X-I**, the editing model **SEED-X-Edit**, and our de-tokenier, which can generate realistic images from ViT features (w/o or w/ a condition image).

**2024-04-22** :hugs: We release an online [gradio demo](https://139a5c1d085953f17b.gradio.live/) of a general instruction-tuned model SEED-X-I. SEED-X-I can follow multimodal instruction (including images with dynamic resolutions) and make responses with images, texts and bounding boxes in multi-turn conversation. SEED-X-I **does not support image manipulation**. If you want to experience SEED-X-Edit for high-precision image editing, the inference code and model will be released soon.

## TODOs
- [x] Release the multimodal foundation model SEED-X.
- [x] Release the instruction-tuned model SEED-X-Edit for high-precision image editing.
- [ ] Release 3.7M in-house image editing data.

![image](https://github.com/AILab-CVC/SEED-X/blob/main/demos/teaser.jpg?raw=true)

![image](https://github.com/AILab-CVC/SEED-X/blob/main/demos/case_example.jpg?raw=true)


## Usage

### Dependencies
- Python >= 3.8 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))
- [PyTorch >=2.0.1](https://pytorch.org/)
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)

### Installation
Clone the repo and install dependent packages

  ```bash
  git clone https://github.com/AILab-CVC/SEED-X.git
  cd SEED-X
  pip install -r requirements.txt
  ```

### Model Weights
We release the pretrained De-Tokenizer, the pre-trained foundation model **SEED-X**, the general instruction-tuned model **SEED-X-I**, the editing model **SEED-X-Edit** in in [SEED-X-17B Hugging Face](https://huggingface.co/AILab-CVC/SEED-X-17B/tree/main).

Please download the checkpoints and save them under the folder `./pretrained`. For example, `./pretrained/seed_x`.

You also need to download [stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) and [Qwen-VL-Chat](https://huggingface.co/Qwen/Qwen-VL-Chat), and save them under the folder `./pretrained`. Please use the following script to extract the weights of visual encoder in Qwen-VL-Chat.
```bash
python3 src/tools/reload_qwen_vit.py
```
### Inference with SEED-X De-tokenizer
```bash
# For image reconstruction with ViT image features
python3 src/inference/eval_seed_x_detokenizer.py
# For image reconstruction with ViT image features and conditional image
python3 src/inference/eval_seed_x_detokenizer_with_condition.py
```

### Inference with pre-trained model SEED-X
```bash
# For image comprehension and detection
python3 src/inference/eval_img2text_seed_x.py
# For image generation
python3 src/inference/eval_text2img_seed_x.py
```

### Inference with the general instruction-tuned model SEED-X-I
```bash
# For image comprehension and detection
python3 src/inference/eval_img2text_seed_x_i.py
# For image generation
python3 src/inference/eval_text2img_seed_x_i.py
```

### Inference with the editing model SEED-X-Edit
```bash
# For image editing
python3 src/inference/eval_img2edit_seed_x_edit.py
```

## Citation
If you find the work helpful, please consider citing:
```bash
@article{ge2024seed,
  title={SEED-X: Multimodal Models with Unified Multi-granularity Comprehension and Generation},
  author={Ge, Yuying and Zhao, Sijie and Zhu, Jinguo and Ge, Yixiao and Yi, Kun and Song, Lin and Li, Chen and Ding, Xiaohan and Shan, Ying},
  journal={arXiv preprint arXiv:2404.14396},
  year={2024}
}
```


## License
`SEED` is licensed under the Apache License Version 2.0 except for the third-party components listed in [License](License_Seed-X.txt). 

During training SEED-X, we freeze the original parameters of LLaMA2 and optimize the LoRA module.
