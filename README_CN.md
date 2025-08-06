<div align="center">
<h2>一步扩散：细节丰富且时间一致的视频超分辨率</h2>

[孙玉静](https://yjsunnn.github.io/)<sup>1,2, *</sup> | 
[孙凌辰](https://scholar.google.com/citations?hl=zh-CN&tzom=-480&user=ZCDjTn8AAAAJ)<sup>1,2, *</sup> | 
[刘帅正](https://scholar.google.com/citations?user=wzdCc-QAAAAJ&hl=en)<sup>1,2</sup> | 
[吴荣远](https://scholar.google.com/citations?user=A-U8zE8AAAAJ&hl=zh-CN)<sup>1,2</sup> | 
[张正强](https://scholar.google.com.tw/citations?user=UX26wSMAAAAJ&hl=en)<sup>1,2</sup> | 
[张磊](https://www4.comp.polyu.edu.hk/~cslzhang)<sup>1,2</sup>

<sup>1</sup>香港理工大学，<sup>2</sup>OPPO研究院
</div>

<div>
    <h4 align="center">
        <a href="https://yjsunnn.github.io/DLoRAL-project/" target='_blank'>
        <img src="https://img.shields.io/badge/💡-项目页面-gold">
        </a>
        <a href="https://arxiv.org/pdf/2506.15591" target='_blank'>
        <img src="https://img.shields.io/badge/arXiv-2312.06640-b31b1b.svg">
        </a>
        <a href="https://www.youtube.com/embed/Jsk8zSE3U-w?si=jz1Isdzxt_NqqDFL&vq=hd1080" target='_blank'>
        <img src="https://img.shields.io/badge/演示视频-%23FF0000.svg?logo=YouTube&logoColor=white">
        </a>
        <a href="https://www.youtube.com/embed/xzZL8X10_KU?si=vOB3chIa7Zo0l54v" target="_blank">
        <img src="https://img.shields.io/badge/2分钟讲解-brightgreen?logo=YouTube&logoColor=white">
        </a>
        </a>
        <a href="https://github.com/yjsunnn/Awesome-video-super-resolution-diffusion" target="_blank">
        <img src="https://img.shields.io/badge/GitHub-优秀视频超分辨率扩散模型-181717.svg?logo=github&logoColor=white">
        </a>
        </a>
        <a href="https://colab.research.google.com/drive/1QAEn4uFe4GNqlJbogxxhdGFhzMr3rfGm?usp=sharing" target="_blank">
        <img src="https://img.shields.io/badge/Colab演示-F9AB00?style=flat&logo=googlecolab&logoColor=white">
        </a>
        <a href="https://github.com/yjsunnn/DLoRAL" target='_blank' style="text-decoration: none;"><img src="https://visitor-badge.laobi.icu/badge?page_id=yjsunnn/DLoRAL"></a>
    </h4>
</div>

<p align="center">

<img src="assets/visual_results.svg" alt="视觉效果">

</p>

## ⏰ 更新

- **2025.07.14**: [Colab演示](https://colab.research.google.com/drive/1QAEn4uFe4GNqlJbogxxhdGFhzMr3rfGm?usp=sharing)现已可用。✨ **无需本地GPU或设置** - 只需上传并增强！
- **2025.07.08**: 推理代码和预训练权重已可用。
- **2025.06.24**: 项目页面已可用，包括简短的2分钟讲解视频、更多视觉效果和相关研究。
- **2025.06.17**: 代码仓库已发布。

:star: 如果DLoRAL对您的视频或项目有帮助，请帮忙给这个仓库点星。谢谢！:hugs:

😊 您可能还想查看我们的相关作品：

1. **OSEDiff (NIPS2024)** [论文](https://arxiv.org/abs/2406.08177) | [代码](https://github.com/cswry/OSEDiff/)  

   已应用于OPPO Find X8系列的实时图像超分辨率算法。

2. **PiSA-SR (CVPR2025)** [论文](https://arxiv.org/pdf/2412.03017) | [代码](https://github.com/csslc/PiSA-SR) 

   图像超分辨率中双LoRA范式的开创性探索。

3. **优秀视频超分辨率扩散模型** [仓库](https://github.com/yjsunnn/Awesome-video-super-resolution-diffusion)

   使用扩散模型进行视频超分辨率(VSR)的精选资源列表。

## 👀 待办事项
- [x] 发布推理代码。
- [x] 提供便捷测试的Colab演示。
- [x] 发布训练代码。
- [ ] 发布训练数据。


## 🌟 框架概述

<p align="center">

<img src="assets/pipeline.svg" alt="DLoRAL框架">

</p>

**训练**: 动态双阶段训练方案在优化时间一致性（一致性阶段）和细化高频空间细节（增强阶段）之间交替，通过平滑损失插值确保稳定性。

**推理**: 在推理过程中，C-LoRA和D-LoRA都合并到冻结的扩散UNet中，实现低质量输入到高质量输出的一步增强。


## 🔧 依赖和安装

1. 克隆仓库
    ```bash
    git clone https://github.com/yjsunnn/DLoRAL.git
    cd DLoRAL
    ```

2. 安装依赖包
    ```bash
    conda create -n DLoRAL python=3.10 -y
    conda activate DLoRAL
    pip install -r requirements.txt
    # mim安装mmedit和mmcv
    pip install openmim
    mim install mmcv-full mmengine
    pip install mmedit
    ```

3. 下载模型 
#### 依赖模型
* [RAM](https://huggingface.co/spaces/xinyu1205/recognize-anything/blob/main/ram_swin_large_14m.pth) --> 放入 **/path/to/DLoRAL/preset/models/ram_swin_large_14m.pth**
* [DAPE](https://drive.google.com/file/d/1KIV6VewwO2eDC9g4Gcvgm-a0LDI7Lmwm/view?usp=drive_link) --> 放入 **/path/to/DLoRAL/preset/models/DAPE.pth**
* [预训练权重](https://drive.google.com/file/d/1vpcaySpRx_K-tXq2D2EBqFZ-03Foky8G/view?usp=sharing) --> 放入 **/path/to/DLoRAL/preset/models/checkpoints/model.pkl**

每个路径都可以根据自身需求进行修改，相应的更改也应应用于命令行和代码中。

## 🖼️ 快速推理
对于真实世界视频超分辨率：

```
python src/test_DLoRAL.py     \
--pretrained_model_path stabilityai/stable-diffusion-2-1-base     \
--ram_ft_path /path/to/DLoRAL/preset/models/DAPE.pth     \
--ram_path '/path/to/DLoRAL/preset/models/ram_swin_large_14m.pth'     \
--merge_and_unload_lora False     \
--process_size 512     \
--pretrained_model_name_or_path stabilityai/stable-diffusion-2-1-base     \
--vae_encoder_tiled_size 4096     \
--load_cfr     \
--pretrained_path /path/to/DLoRAL/preset/models/checkpoints/model.pkl     \
--stages 1     \
-i /path/to/input_videos/     \
-o /path/to/results
```

```
python src/test_DLoRAL.py --pretrained_model_path stabilityai/stable-diffusion-2-1-base --ram_ft_path preset/models/DAPE.pth --ram_path 'preset/models/ram_swin_large_14m.pth' --merge_and_unload_lora False --process_size 512 --pretrained_model_name_or_path stabilityai/stable-diffusion-2-1-base --vae_encoder_tiled_size 4096 --load_cfr --pretrained_path preset/models/checkpoints/model.pkl --stages 1 -i input_videos/ -o results
```

## ⚙️ 训练
对于真实世界视频超分辨率：

```
bash train_scripts.sh
```

一些关键参数及其对应含义：
参数 | 描述 | 示例值
--- | --- | ---
`--quality_iter` | 从一致性阶段切换到质量阶段的初始步数 | `5000`
`--quality_iter_1_final` | 从质量阶段切换到一致性阶段所需的步数 | `13000`
`--quality_iter_2` | `quality_iter_1_final`之后切换回质量阶段的相对步数（实际切换发生在`quality_iter_1_final + quality_iter_2`） | `5000`
`--lsdir_txt_path` | 第一阶段数据集路径 | `"/path/to/your/dataset"`
`--pexel_txt_path` | 第二阶段数据集路径 | `"/path/to/your/dataset"`



## 💬 联系方式：
如果您有任何问题（不仅是关于DLoRAL，还包括突发/视频超分辨率相关的问题），请随时通过yujingsun1999@gmail.com联系我

### 引用
如果我们的代码对您的研究或工作有帮助，请考虑引用我们的论文。
以下是BibTeX引用：

```
@misc{sun2025onestepdiffusiondetailrichtemporally,
      title={One-Step Diffusion for Detail-Rich and Temporally Consistent Video Super-Resolution}, 
      author={Yujing Sun and Lingchen Sun and Shuaizheng Liu and Rongyuan Wu and Zhengqiang Zhang and Lei Zhang},
      year={2025},
      eprint={2506.15591},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2506.15591}, 
}

``` 