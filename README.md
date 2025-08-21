<p align="center" >
    <img src="docs/img/logo.png"  width="60%" >
</p>

## <div align="center" >4DSloMo: 4D Reconstruction for High Speed Scene with Asynchronous Capture<div align="center">

###  <div align="center"> SIGGRAPH Asia 2025 </div>
<div align="center">
  <a href="https://yutian10.github.io">Yutian Chen</a>, 
  <a href="https://guoshi28.github.io">Shi Guo</a>, 
  <a href="https://tianshuoy.github.io">Tianshuo Yang</a>, 
  <a href="https://dinglihe.github.io">Lihe Ding</a>, 
  <a href="">Xiuyuan Yu</a>, 
  <a href="http://www.gujinwei.org">Jinwei Gu</a>, 
  <a href="https://tianfan.info/">Tianfan Xue</a>
</div>

<br>

<p align="center"> <a href='https://openimaginglab.github.io/4DSloMo/'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp;
<a href="https://arxiv.org/pdf/2507.05163"><img src="https://img.shields.io/static/v1?label=Arxiv&message=4DSloMo&color=red&logo=arxiv"></a> &nbsp;
 <a href='https://huggingface.co/yutian05/4DSloMo/tree/main'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-yellow'></a> &nbsp;
 <a href=''><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Data(comming soon)-orange'></a> &nbsp;
</p>

<p align="center" width="100%">
    <img src="docs/img/teaser3.gif"  width="90%" >
</p>

## TODO List

- [ ] Upload Datasets (Expected before UTC 2025.09.1）

## 🛠️ Environment Setup

###  1. Clone Repository and Setup Environment
``` 
git clone https://github.com/OpenImagingLab/4DSloMo.git
cd 4DSloMo
conda create -n 4dslomo python=3.10 -y
conda activate 4dslomo
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```
###  2. Download Models
4DSloMo relies on two sets of weights. Please download them and place them in the `./checkpoints` folder.

- Wan2.1 I2V 14B 720P [[download](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P/tree/main)]
- 4DSloMo Lora weights [[download](https://huggingface.co/yutian05/4DSloMo/tree/main)]
## 🚀 Quick Start

### 1. Initialize 4D Gaussian Splatting
``` 
python train.py --config ./configs/default.yaml --model_path ./output/dance_demo10 --source_path ./datasets/dance_demo10
``` 

### 2. Run Artifact-fix Model
```
# Render 4D Gaussian Splatting
python render.py --model_path ./output/dance_demo10/ --loaded_pth=./output/dance_demo10/chkpnt7000.pth

# Prepare data for artifact-fix model
python process_video.py --input_folder "./output/dance_demo10/test/ours_None/" --max_frames 33

# Inference artifact-fix model
## Note: 5 denoising steps can achieve about 80% of the final quality; use 50 steps for the best results.
CUDA_VISIBLE_DEVICES=0,1  torchrun --nproc_per_node=2 test_lora.py --input_folder ./output/dance_demo10 --output_folder ./datasets/dance_demo10_wan/ --model_path ./checkpoints/4DSloMo_LoRA.ckpt --num_inference_steps 5
```
### 3. Repair 4D Gaussian Splatting
```
# Prepare camera pose and timestamp 
cp ./datasets/dance_demo10/transforms_test_demo.json ./datasets/dance_demo10_wan/transforms_test.json; cp ./datasets/dance_demo10/transforms_train_stage2.json ./datasets/dance_demo10_wan/transforms_train.json; cp ./datasets/dance_demo10/points3d.ply ./datasets/dance_demo10_wan


python train.py --config ./configs/default.yaml --model_path ./output/dance_demo10_wan --source_path ./datasets/dance_demo10_wan
```

## 💗 Acknowledgments
Thanks to these great repositories: [4D Gaussian Splatting](https://github.com/fudan-zvg/4d-gaussian-splatting), [Wan2.1](https://github.com/Wan-Video/Wan2.1) and [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio).
## 🔗 Citation
If you find our work helpful, please cite it:
```
@article{chen20254dslomo,
    title={4DSloMo: 4D Reconstruction for High Speed Scene with Asynchronous Capture},
    author={Chen, Yutian and Guo, Shi and Yang, Tianshuo and Ding, Lihe and Yu, Xiuyuan and Gu, Jinwei and Xue, Tianfan},
    journal={arXiv preprint arXiv:2507.05163},
    year={2025}
}
```
