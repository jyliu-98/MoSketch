<div align="center">

<h1>Multi-Object Sketch Animation by Scene Decomposition and Motion Planning (ICCV 2025)</h1>

<div align="center">
  <a href="https://rucmm.github.io/MoSketch"><img src="https://img.shields.io/static/v1?label=Project&message=Page&color=green"></a>&ensp;
  <a href="https://arxiv.org/abs/2503.19351"><img src="https://img.shields.io/badge/arXiv-2503.19351-b31b1b.svg"></a> &ensp;
  &ensp;
  <a href="https://github.com/jyliu-98/MoSketch">
  <img src="https://img.shields.io/github/stars/jyliu-98/MoSketch?style=social"></a> &ensp;
 &ensp;
</div>

<div>
    <a href='https://scholar.google.com/citations?user=u7Dqok8AAAAJ
    ' target='_blank'>Jingyu Liu</a>&emsp;
    <a href='https://xxayt.github.io/' target='_blank'>Zijie Xin</a>&emsp;
    <a href='https://github.com/jyliu-98/MoSketch' target='_blank'>Yuhan Fu</a>&emsp;
    <a href='https://ruixiangzhao.github.io/' target='_blank'>Ruixiang Zhao</a>&emsp;
    <a href='https://github.com/jyliu-98/MoSketch' target='_blank'>Bangxiang Lan</a>&emsp;
    <a href='http://lixirong.net/' target='_blank'>Xirong Li</a><sup>†</sup>&emsp;
</div>
<div>
    Renmin University of China
</div>
<img src="repo_image/intro.png"/>
</div>

## :new: Latest Update
- **[2025.08.02]** 🔥 We released more created multi-object sketches! There are 560 multi-object sketches now!
- **[2025.07.29]** 🔥 We released the **[project page](https://rucmm.github.io/MoSketch)** for **MoSketch**.
- **[2025.07.26]** 🔥 We released the **[code](https://github.com/jyliu-98/MoSketch)** for **MoSketch**.
- **[2025.06.26]** 🎉 MoSketch is accepted by ICCV 2025!
- **[2025.03.25]** 🔥 We released the **[MoSketch Paper](https://arxiv.org/abs/2503.19351)**. MoSketch is an iterative 
optimization based and thus **training-data free** method, aiming to animate a multi-object sketch *w.r.t.* a specific textual instruction.

## 📄 Abstract
Sketch animation, which brings static sketches to life by generating dynamic video sequences, 
has found widespread applications in GIF design, cartoon production, and daily entertainment. 
While current methods for sketch animation perform well in single-object sketch animation, 
they struggle in *multi*-object scenarios. By analyzing their failures, 
we identify two major challenges of transitioning from single-object to multi-object sketch animation: 
object-aware motion modeling and complex motion optimization. For multi-object sketch animation, 
we propose MoSketch based on iterative optimization through Score Distillation Sampling (SDS) and thus animating a multi-object sketch in a training-data free manner. 
To tackle the two challenges in a divide-and-conquer strategy, MoSketch has four novel modules, 
*i.e.*, LLM-based scene decomposition, LLM-based motion planning, multi-grained motion refinement, and compositional SDS. 
Extensive qualitative and quantitative experiments demonstrate the superiority of our method over existing sketch animation approaches. 
MoSketch takes a pioneering step towards multi-object sketch animation, opening new avenues for future research and applications.

## 🔧 Setup
Download the code of MoSketch.
```
git clone https://github.com/jyliu-98/MoSketch.git
cd MoSketch
```
### 📌 Environment
Create a new anaconda environment and install dependencies.
```
# create a new anaconda env
conda create -n mosketch python==3.8 -y
conda activate mosketch

# install torch and dependencies
pip install -r requirements.txt
```
Install diffvg
```
# install diffvg's dependencies
conda install -y -c anaconda cmake
conda install -y -c conda-forge ffmpeg
# ************************************************
# if 'conda install -y -c conda-forge ffmpeg' gets stuck, try these:
# conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
# conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
# conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
# conda config --set show_channel_urls yes
# conda install -y ffmpeg
# ************************************************

# install diffvg
git clone https://github.com/BachiLi/diffvg.git
cd diffvg
git submodule update --init --recursive
python setup.py install
cd ..
rm -rf diffvg
```
### ⬇️ Checkpoint of T2V Diffusion Model
Download the checkpoint of [ModelScopeT2V](https://huggingface.co/ali-vilab/text-to-video-ms-1.7b/tree/main), and
put it in the code of MoSketch (`./text-to-video-ms-1.7b`).

## ✍️ Input Sketch
The input sketches should be provided in SVG format, where a sketch is composed of strokes 
and each stroke is a cubic Bézier curve controlled by four points, as described in the paper. 
Make sure the input sketches can be processed with diffvg.

The **60 multi-object sketches** used in the paper are provided in `./data/raw/60sketches`. They can be processed with diffvg.

For own sketch preparation, if you want to generate sketches automatically, we recommend [CLIPasso](https://clipasso.github.io/clipasso/), 
an image-to-sketch method that produces sketches in vector format; if you want to create sketches manually, you can use some free online tools, 
such as [js.design](https://js.design/special/article/svg-online-editors.html). 
Note that sketches created by [CLIPasso](https://clipasso.github.io/clipasso/) can be processed with diffvg, 
while sketches created by [js.design](https://js.design/special/article/svg-online-editors.html) should go through `./preprocess.py`.

**Our preparation process:**
* Creating some single-object vector sketches by [CLIPasso](https://clipasso.github.io/clipasso/).
* Using [js.design](https://js.design/special/article/svg-online-editors.html) to gather the single-object vector sketches in a reasonable scene.
* Using [js.design](https://js.design/special/article/svg-online-editors.html) to edit the multi-object sketch (*e.g.*, add or delete strokes).
* Save the multi-object sketch, and run `./preprocess.py` to make sure the sketch can be processed with diffvg.
<img src="repo_image/creation.png"/>

**We release 500 more created sketches!** There are 560 vector multi-object sketches now! (`./data/raw/560sketches.zip`)

## 🎨 Generate a Video!
### 🚀 Quick Start
The scene decomposition (`_decomp.txt`), point assignment (`_semantic.txt`) and motion plan (`_traj.txt`) of the **60 multi-object sketches**
are provided in `./data/processed`. The text instruction of each sketch can be found in `./data/raw/60sketches/caption.txt`. 
Run the following command to get the animation of one sketch (*e.g.*, 'basketball5'):
```
CUDA_VISIBLE_DEVICES=0 python animate_mosketch.py \
        --sketch 'basketball5' \
        --caption 'The player soars through the air with a basketball, arm extended for an electrifying slam dunk to a hoop.' \
        --num_iter 500 \
        --seed 130 \
        --num_frames 20
```
The output video will be saved in `./output/basketball5`.
