# Upsamplers 模块介绍 (`upsamplers.py`)

## 1. 文件的作用
`upsamplers.py` 主要提供了将低分辨率 (LR) 特征上采样到高分辨率 (HR) 特征的模型和相关工具函数。在网络推理或特征提取过程中，该文件中的模块可以利用高分辨率原图的信息对降采样或深层网络输出的低分辨率特征进行精细化还原，以获得高质量的高分辨率特征图。

## 2. 可用的上采样模型及使用方法

目前文件中内置了以下几种上采样模型：

### 2.1 LoftUp (Cross-Attention 上采样)
* **介绍**: 该模型采用高分辨率图像的傅里叶特征 (Fourier Features) 作为输入，与低分辨率特征进行交叉注意力 (Cross Attention) 计算。结合位置编码（支持 sine 或 learnable），可以非常精细地融合纹理与结构信息来输出高分辨率特征。
* **使用方法**:
  * 实例化模型：可以通过 `get_upsampler(upsampler="loftup", dim=..., lr_size=...)` 直接获取未初始化的模型。
  * 加载预训练权重：使用自带的便捷函数 `load_loftup_checkpoint(upsampler_path, n_dim)` 或者 `load_upsampler_weights(upsampler, upsampler_path, dim)` 来加载指定路径的 `.ckpt` 权重，此时会自动打包包含特征通道归一化 (`ChannelNorm`) 的网络结构 (`UpsamplerwithChannelNorm`)。

### 2.2 Bilinear (双线性插值基线)
* **介绍**: 提供一个最基础的强基线模型，直接利用 PyTorch 内置的 `F.interpolate` 将低频特征使用双线性插值缩放到与目标图像同样的 `(H, W)` 尺寸上。不需要参数训练。
* **使用方法**: 
  * 通过调用 `get_upsampler("bilinear", dim=0, ...)`，或者直接实例化 `Bilinear()`。传入特征和目标图像时，它会自动计算目标形状并返回插值结果。

## 3. 如何创造一个新的上采样模型

如果你需要实现自己定义的上采样算法，请按照以下三个步骤进行扩展：

1. **定义模型类**: 创建一个继承自 `torch.nn.Module` 的新类。
2. **重写 `forward` 方法**: 确保统一的输入接口，即 `forward(self, lr_feats, img)`。
   * `lr_feats`: 形状为 `(B, C, H_lr, W_lr)` 的低分辨率特征输入。
   * `img`: 形状为 `(B, 3, H_hr, W_hr)` 的高分辨率目标参考图。
   * 确保 `forward` 方法最终返回形状为 `(B, C_out, H_hr, W_hr)` 的高分辨率特征矩阵。
3. **注册新模型**: 在该文件的 `get_upsampler` 函数中添加新的分支以支持通过字符串调用，例如：
   ```python
   def get_upsampler(upsampler, dim, lr_size=16, n_freqs=20, cfg=None, lr_pe_type="sine"):
       # ... existing code ...
       elif upsampler == "bilinear":
           return Bilinear()
       elif upsampler == "your_new_model":   # 注册新的上采样器
           return YourNewModelClass(dim, ...)
       # ... existing code ...
   ```

