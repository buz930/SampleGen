# 进度汇报
成员：周佳妮 谭笑

时间：2026.6.18

负责任务(1)(2)
## 任务1：复现

摘要：截止目前，我们基于NAF仓库，完成了文献阅读、计划制定、基础环境配置、四种上采样方法在部分数据集上的复现工作，分以下几个部分简要介绍进度
>目前云服务器消费共约130元

- **文献阅读**
- **复现计划**
- **当前进度**
- **遇到的问题**

### 文献阅读
我们阅读了选题所列出的参考文献，最终选择论文
>*NAF: Zero-Shot Feature Upsampling via Neighborhood Attention Filtering*

中的指标进行复现，文章提出了NAF上采样方法，并在VOC、COCO、ADE20K等经典数据集上，结合DINOv3等下游模型，执行语义分割、深度估计、开放词汇分割、视频任务分割等下游任务，与JBF、JBU、AnyUp等经典上采样方法进行评价指标的对比。

### 复现计划
我们将复现任务大致分为基础环境搭建、数据集下载、参数配置、指标复现几个部分。

在实际实验中发现，数据集的下载与配置最为繁琐，且资源占用最多；同时不同下游任务对应的数据集往往不尽相同，因此**在分工时以数据集作为依据**（一人负责几个数据集）。

计划复现的指标以表格形式列出，表格中计划填入的指标：
**语义分割 (Segmentation)：**
 mIoU
**深度估计 (Depth Estimation):**
RMSE (Root Mean Squared Error)
REL (Relative Error, 相对误差)
**开放词汇划分(open-vocabulary segmentation)：**
 mIoU
**视频分割 (Video Segmentation):**
J&F (Jaccard Index and F-measure 的平均值)

### 复现进度
对比NAF、ANYUP以及Upsample Anything三篇论文关于对比模型的选取以及下游任务的完成度，我们最终选取了以NAF这篇论文为基础，以Dinov3-B为模型，围绕语义分割，深度估计，开放词汇划分以及视频分割四个下游任务，进行NAF、ANYUP、JAFAR、Bilinear四种方法的复现与数据对比。下游视觉模型均采用DINOv3-B。

填表格式为：复现数据 / 文献数据


| Method   | Semantic Segmentation mIoU - COCO | Semantic Segmentation mIoU - VOC | Semantic Segmentation mIoU - ADE20K | Semantic Segmentation mIoU - Cityscapes | depth-estimation - NYUv2 | video-segmentation - DAVIS | open-vocabulary segmentation-Pascal VOC |
|----------|--------------------------|-------------------------|----------------------------|--------------------------------|--------------------------|---------------------------|----------------------------------------|
| NAF      |                          | 87.81/87.85             | 47.36/47.41                           | 64.95/64.98                    | 85.62/86.73                       | 70.88/70.55               | 64.09/63.86                                       |
| ANYUP    |                          | 86.59/86.62                        |                            |60.87/60.35                                |85.46/86.36                          |66.44/65.90                           |63.54/63.41                                        |
| JAFAR    |                          | 86.95/87.10                        |                            |  62.68/62.36                              | 85.32/86.37                         |69.89/69.24                           | 63.82/63.72                                       |
| Bilinear |                          | 86.91/86.99                        |                            |65.46/63.08                                | 85.10/86.10                         |70.41/70.00                           |62.15/62.21                                        |


#### 说明：
- **语义分割**: 语义分割需跑四个数据集，目前四个数据集的代码均已跑通，但是COCO数据集的评估效率较低，在RTX 4080 Ti上一小时跑不完1个epoch，故先搁置；

- **深度估计**: 深度估计在NAF原始仓库代码中并未给出详细的数据格式以及评估代码（描述中仅提及按Prob3D仓库流程评估），且loss仅给出公式和原理，无相应代码，我们按论文要求补全后进行复现工作，**mIOU相比论文原始数据稍微偏低一点点（已尽力）**，但模型之间的性能对比结论与论文一致，姑且认为完成了复现任务；

- **开放词汇划分**: 分析文献后发现这部分作者是利用ProxyCLIP仓库来评估上采样表示，将其默认的双线性上采样替换为包括NAF在内的不同上采样器，由于该仓库要求的**Torch版本老旧**，而NAF是基于新Torch写的，且需要新版natten库，**移植起来非常非常非常不方便**，但我们攻坚克难完成了复现任务，且指标与论文中基本一致；

- **视频分割**: 作者在文献中提到这部分仍然是利用ProxyCLIP来评估，但NAF原始仓库中给出了视频分割的python代码文件，我们首先根据文献附录将仓库配置文件参数与文献中的记录保持一致，复现出的数据略高于文献数据，后续会进一步比对NAF仓库与ProxyCLIP仓库中关于视频分割的实现，进一步校正复现数据。

- **复现源代码**:当前在AutoDL服务器上，计划等完成所有任务后上传至本分支。

#### 运行指令
```bash
python evaluation/eval_seg_probing.py dataset=voc eval.model_ckpt=output/naf_release.pth
```
> 语义分割VOC数据集运行指令

```bash
python evaluation/eval_seg_probing.py dataset=nyu eval.model_ckpt=output/naf_release.pth
```
> 深度估计NYUv2数据集运行指令

```bash
python NAF-main/evaluation/eval_video_seg.py dataset=davis dataroot=/root/autodl-tmp/data/DAVIS backbone.name=vit_base_patch16_dinov3.lvd1689m eval.model_ckpt=/root/NAF-main/NAF-main/output/naf_release.pth
```
> 视频分割DAVIS数据集运行指令

```bash
python evaluation/eval_seg_probing.py dataset=cityscapes dataroot=/root/NAF-main/datasets_local/ eval.model_ckpt=/root/NAF-main/NAF-main/output/naf_release.pth
```
> 语义分割cityscapes数据集运行指令

```bash
python evaluation/eval_seg_probing.py dataset=ade20k dataroot=/root/NAF-main/datasets_local/ eval.model_ckpt=/root/NAF-main/NAF-main/output/naf_release.pth
```
> 语义分割ADE20k数据集运行指令

```bash
python bin/run_patchcore_config.py \
  --config configs/patchcore_mvtec.yaml \
  --method naf \
  --scale 2 \
  --class all
```
> 开发词汇分割VOC数据集运行指令(基于ProxyCLIP仓库)

#### 后续计划
除语义分割的COCO数据集、部分ADE数据集的复现任务外，复现工作已基本完成，后续将继续补全。

### 遇到的问题
- **COCO数据集过大，训练迭代次数较多，复现花费时间长，租服务器的资金较多（完成所有方法在该数据集的评估，单卡4080预估需要20小时）**；

## 任务2：下游任务扩展
摘要：基于任务2要求，我们选取NAF论文未提及的下游任务进行了扩展实验，对比NAF、ANYUP、JAFAR、Bilinear四种方法的表现。评估方法为特征图可视化+评价指标对比。

### 进度
目前完成三个拓展任务（LoveDA、ISIC2018、DA-2K，数据集由刘思瑶同学提供）。三组实验都使用冻结的DINOv3 backbone，只训练很轻量的任务头。比较的上采样方法，与任务一一致，为 bilinear、AnyUp、JAFAR、NAF。
>对于较大数据集，取子集（约500pics）作轻量评估，DA-2K数据集已完成全量评估。

### 共同设置

- backbone: vit_base_patch16_dinov3.lvd1689m
- 输入尺寸: 448 x 448
- backbone: 冻结
- upsampler: 冻结
- 优化器: AdamW
- 学习率: 5e-4
- weight decay: 1e-5
- scheduler: one-cycle cosine

### LoveDA 语义分割

数据集规模:

- 数据集: LoveDA_5class_100each_segmentation
- 类别数: 5
- 类别: building、road、water、forest、agricultural
- 总样本数: 约 500 张
- 划分: 80% train，20% val，seed=0

超参数:

- epochs: 20
- batch size: 2
- num workers: 4
- 数据增强: 随机水平翻转
- 损失: cross entropy
- 忽略标签: 255

实验流程:

```text
RGB image
-> frozen DINOv3 backbone
-> bilinear / AnyUp / JAFAR / NAF 上采样特征到 448 x 448
-> 1x1 Conv segmentation probe
-> 每像素 5 类 logits
-> argmax 得到分割 mask
-> 计算 mIoU 和 aAcc
```

评估指标:

| 方法 | mIoU (%) |
|---|---:|
| bilinear | 88.70 |
| AnyUp | 87.86 |
| JAFAR | 87.67 |
| **NAF** | 88.72 |

### ISIC2018 Task1 皮肤病变分割

数据集规模:

- 数据集: ISIC2018 Task1 子集
- 图像数: 500 张
- mask: 500 张
- 类别数: 2
- 类别: background、lesion
- 划分: 400 train，100 val，seed=0

超参数:

- epochs: 5
- batch size: 2
- num workers: 4
- 数据增强: 随机水平翻转
- 损失: cross entropy

实验流程:

```text
ISIC image
-> frozen DINOv3 backbone
-> bilinear / AnyUp / JAFAR / NAF 上采样特征到 448 x 448
-> 1x1 Conv binary segmentation probe
-> 每像素 background / lesion logits
-> argmax 得到 lesion mask
-> 计算 Dice、IoU、aAcc
```

指标:

| 方法 | Dice (%) | IoU (%) | aAcc (%) |
|---|---:|---:|---:|
| bilinear | 85.69 | 74.96 | 94.69 |
| **AnyUp** | 86.75 | 76.61 | 95.11 |
| JAFAR | 86.20 | 75.74 | 94.84 |
| NAF | 85.99 | 75.42 | 94.81 |

### DA-2K 相对深度估计

数据集规模:

- 数据集: DA-2K
- 图像数: 1027 张
- 标注: 点对相对深度标注，约 2K 个 pair
- 划分: 826 train，207 val，seed=0
- 验证点对数: 425

超参数:

- epochs: 5
- batch size: 1
- num workers: 2
- 损失: pair ranking loss
- 训练目标: point1 比 point2 更近

实验流程:

```text
RGB image
-> frozen DINOv3 backbone
-> bilinear / AnyUp / JAFAR / NAF 上采样特征到 448 x 448
-> 1x1 Conv depth/closeness probe
-> 输出单通道 score map
-> 在标注点对位置采样 score
-> 判断 score(point1) > score(point2)
-> 计算 pairwise accuracy 和 WHDR
```

指标:

| 方法 | Pairwise accuracy (%) | WHDR (%) | best epoch |
|---|---:|---:|---:|
| bilinear | 82.82 | 17.18 | 5 |
| AnyUp | 81.65 | 18.35 | 4 |
| JAFAR | 82.59 | 17.41 | 2 |
| **NAF** | 85.18 | 14.82 | 3 |

