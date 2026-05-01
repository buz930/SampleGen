# [SampleGen: Generating the next resolution by the Next-scale generative method](https://arxiv.org/abs/2504.14032)


## Contents
- [Install](#install)
- [Inference & Example Usage](#inference--example-usage)
- [Registering a New Custom Upsampler (e.g., FeatUp)](#registering-a-new-custom-upsampler-eg-featup)
- [Evaluation on Downstream Tasks](#evaluation-on-downstream-tasks)
- [Training LoftUp upsamplers](#training-loftup-upsamplers)
- [Citation](#citation)

## Install

In general, LoftUp can run with most recent pytorch environments. We encourage the users to try out LoftUp in their exisitng environment first.

We also provide two yaml file for installation. To use them, simply run:

```bash
conda env create -f environment_cuda11.yaml
# or 
conda env create -f environment.yaml
```

## Inference & Example Usage

All pre-trained upsamplers are available on 🤗 here: https://huggingface.co/models?search=loftup.

我们提供了一个单图推理的示例脚本 [example_usage.py](example_usage.py)。
在脚本中，你可以非常方便地替换并调用不同的上采样器来提取高分辨率特征 (`hr_feats`)。

### 1. 使用官方预训练的 LoftUp 模型
```python
import torch

# 通过 torch.hub 快速加载预训练权重
upsampler = torch.hub.load('andrehuang/loftup', 'loftup_dinov2s', pretrained=True)
upsampler = upsampler.to('cuda')

# 提取特征并根据参考图指导上采样
lr_feats = model(normalized_img_tensor) # 1, dim, lr_size, lr_size
hr_feats = upsampler(lr_feats, normalized_img_tensor) # 1, dim, 224, 224
```
目前支持的 `torch_hub_name` 包括：`loftup_dinov2s`, `loftup_dinov2b`, `loftup_dinov2s_reg`, `loftup_clip`, `loftup_siglip`, `loftup_siglip2` 等。

### 2. 使用自定义上采样器 (如新增的 FeatUp)
当你在工程中注册了类似 `FeatUp` 这样的新机制后，可以在示例文件中利用本地的工厂函数直接按名称初始化：
```python
from upsamplers import get_upsampler, load_upsampler_weights
from featurizers import get_featurizer

# 1. 实例化上采样器 (将 upsampler_type 改为你注册的名称 "featup")
upsampler = get_upsampler(upsampler_type="featup", n_dim=dim, lr_size=lr_size)

# 2. 如果新模型有预训练权重，可以通过自带函数加载
# upsampler = load_upsampler_weights(upsampler, "path/to/featup.ckpt", dim)
upsampler = upsampler.to('cuda')

# 3. 提取特征并由该上采样器输出高分辨率特征
lr_feats = model(normalized_img_tensor)
hr_feats = upsampler(lr_feats, normalized_img_tensor) 
```


## Registering a New Custom Upsampler (e.g., FeatUp)

如果要引入一个新的自定义上采样器（如 `FeatUp`），请参考 `upsamplers/README.md` 的指引，进行简单的全局注册：

1. **统一定义接口**：确保你的自定义网络模型（如 `FeatUp`）其 `forward` 方法能够接受低分辨率特征和高分辨率指导图：`forward(self, lr_feats, img)`。
2. **在工厂函数中注册**：在相关的 `upsamplers.py` 文件（或者你存放的统一下发文件）的 `get_upsampler` 函数中添加该模型的分支：
   ```python
   def get_upsampler(upsampler_type, n_dim, lr_size=16, cfg=None):
       # ... existing code ...
       elif upsampler_type == "bilinear":
           return Bilinear()
       elif upsampler_type == "featup":    # 新增注册你的自定义模型
           return FeatUp(n_dim, ...)
       # ... existing code ...
   ```


## Evaluation on Downstream Tasks

### Dataset Preparation
See [Preparing Datasets for Evaluation](datasets/README.md).

### Semantic Segmentation
对于语义分割评测，我们的代码 `eval_seg.py` 内部已经自动接入了上述的工厂函数流程。这意味着当你注册了新的模型（如 `featup`），**无需修改由于网络变更导致的后续繁杂代码**，只需在命令行中传入对应的模型名称即可。

```bash
# 测试原生的 LoftUp 
python eval_seg.py ++upsampler_type="loftup" ++upsampler_path="/path/to/your/loftup.ckpt" ++model_type="dinov2"

# 测试刚才自定义注册的模型（如 FeatUp）
python eval_seg.py ++upsampler_type="featup" ++upsampler_path="/path/to/your/featup_weights.ckpt" ++model_type="dinov2"
```

*内部底层逻辑简要说明 (`eval_seg.py`)*:
代码在初始化时，会自动根据传入名称分配权重机制：
```python
if upsampler_type != "no":
    # 自动根据注册名称初始化模型架构
    upsampler = get_upsampler(upsampler_type, n_dim, lr_size=final_size, cfg=cfg)
    if upsampler_type != "bilinear":
        # 自动加载此模型所需的 `.ckpt`
        upsampler = load_upsampler_weights(upsampler, upsampler_path, n_dim)
```
相关通用配置可以调整：[configs/eval_seg.yaml](configs/eval_seg.yaml)。


### Video Object Segmentation
For video object segmentation on DAVIS, our code is modified from the implementation in [LiFT](https://github.com/saksham-s/lift). Extract segmentation results by running:
```bash
python eval_davis.py --dataroot /your_davis_data_dir --model_type "dinov2" --output_dir /your_output_dir --imsize 224 --upsampler_type "featup" --upsampler_path /path/to/your/custom_weights.ckpt
```
Then run the evaluation script:
```bash
python davis2017-evaluation/evaluation_method.py --davis_path /your_davis_data_dir --task semi-supervised --results_path /your_output_dir/davis_vidseg_224 --imsize 224
```

### Others
For interactive segmentation, please check out [iSegProbe](https://github.com/havrylovv/iSegProbe).
For open-vocabulary segmentation, please check out [ProxyCLIP](https://github.com/mc-lan/ProxyCLIP).
For depth and normal estimation, please check out [Probe3D](https://github.com/mbanani/probe3d).

## Training different upsamplers

### Training LoftUp Upsamplers

This repository contains training scripts for training LoftUp upsamplers. The training is done in two stages:

#### Stage 1: Basic Feature Upsampling
Stage 1 training (`train_loftup_stage1.py`) trains upsamplers to convert low-resolution features to high-resolution features using reconstruction loss.
```bash
python train_loftup_stage1.py ++dataset="sa1b" ++epochs=1 ++batch_size=2 ++num_gpus=4 ++model_type="dinov2" ++pytorch_data_dir='datasets' ++upsampler_type="loftup" ++sam_mask_alpha=0.8 ++load_size=224 ++upsample_size=224 ++tv_weight=0.001 ++clamp_featup=True
```

#### Stage 2: High-Resolution Supervision
Stage 2 training (`train_loftup_stage2.py`) fine-tunes the Stage 1 upsampler with high-resolution supervision for improved quality.
```bash
python train_loftup_stage2.py ++dataset="sa1b" ++epochs=1 ++hr_res=896 ++batch_size=2 ++consistency_method="bilinear" ++model_type="dinov2" ++num_gpus=4 ++affinity_loss=True ++pytorch_data_dir='datasets' ++pretrained_upsampler="path/to/stage1_checkpoint.ckpt" ++upsampler_type="loftup" ++sam_mask_hr_alpha=0.5 ++sam_mask_reg=0.0 ++lr=1e-3 ++use_featup=False ++aug_size ++n_jitters=2
```

