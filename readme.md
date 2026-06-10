# 进度汇报
## 复现部分
成员：周佳妮 谭笑

时间：2026.6.10

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
| ANYUP    |                          | 86.59/86.62                        |                            |60.87/60.35                                |85.46/86.36                          |                           |63.54/63.41                                        |
| JAFAR    |                          | 86.95/87.10                        |                            |  62.68/62.36                              | 85.32/86.37                         |                           | 63.82/63.72                                       |
| Bilinear |                          | 86.91/86.99                        |                            |                                | 85.10/86.10                         |                           |62.15/62.21                                        |


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


#### 后续计划
中期验收前：补全表格（除语义分割的COCO数据集）

### 遇到的问题
- **COCO数据集过大，训练迭代次数较多，复现花费时间长，租服务器的资金较多（完成所有方法在该数据集的评估，单卡4080预估需要20小时）**；


