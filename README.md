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
- **[2025.07.28]** 🔥 We released the **project page** for **[MoSketch](https://rucmm.github.io/MoSketch)**.
- **[2025.07.25]** 🔥 We released the **code** for **[MoSketch](https://github.com/jyliu-98/MoSketch)**.
- **[2025.06.26]** 🎉 MoSketch is accepted by ICCV 2025!
- **[2025.03.25]** 🔥 We released the **[MoSketch Paper](https://arxiv.org/abs/2503.19351)**. MoSketch is an iterative 
optimization based and thus **training-data free** method, aiming to animate a multi-object sketch *w.r.t.* a specific textual instruction.

## 🔧 Setup
Download the code.
```
git clone https://github.com/jyliu-98/MoSketch.git
cd MoSketch
```
### 📌 Environment
Create a new anaconda environment and install dependencies.
```
# create a new anaconda env
conda create -n momosketch python==3.8 -y
conda activate momosketch

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





