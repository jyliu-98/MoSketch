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
- **[2025.08.02]** 🔥 We released **more created multi-object sketches**! There are **560** multi-object sketches now!
- **[2025.07.29]** 🔥 We released the **[project page](https://rucmm.github.io/MoSketch)** for **MoSketch**.
- **[2025.07.26]** 🔥 We released the **[code](https://github.com/jyliu-98/MoSketch)** for **MoSketch**.
- **[2025.06.26]** 🎉 **MoSketch** is accepted by ICCV 2025!
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

**Recommended sketch format:**
* Rendering size: 256x256
* Number of strokes: <300
* Strokes type: cubic Bezier curves
* Stroke width: 1~3

The **60 multi-object sketches** used in the paper are provided in `./data/raw/60sketches`. They can be processed with diffvg.

For own sketch preparation, if you want to generate sketches automatically, we recommend [CLIPasso](https://clipasso.github.io/clipasso/), 
an image-to-sketch method that produces sketches in vector format; if you want to create sketches manually, you can use some free online tools, 
such as [js.design](https://js.design/special/article/svg-online-editors.html). 
Note that sketches created by [CLIPasso](https://clipasso.github.io/clipasso/) can be processed with diffvg, 
while sketches created by [js.design](https://js.design/special/article/svg-online-editors.html) should go through `./preprocess.py`:
```
python preprocess.py --svg_path './data/raw/Yours.svg'
```
After that, get the PNG format of your own sketch, which will be used in the scene decomposition, the stroke(point) assignment, 
and the motion planning. Your can save the PNG in [js.design](https://js.design/special/article/svg-online-editors.html), or run the following Python code to turn SVG to PNG:
``` python
import cairosvg

svg_path = './data/raw/Yours.svg'
png_path = './data/raw/Yours.png'
cairosvg.svg2png(url=svg_path,
                 write_to=png_path,
                 scale=1, background_color="white")
```

**Our preparation process:**
* Creating some single-object vector sketches by [CLIPasso](https://clipasso.github.io/clipasso/).
* Using [js.design](https://js.design/special/article/svg-online-editors.html) to gather the single-object vector sketches in a reasonable scene.
* Using [js.design](https://js.design/special/article/svg-online-editors.html) to edit the multi-object sketch (*e.g.*, add or delete strokes).
* Save the multi-object sketch (both SVG and PNG), and run `./preprocess.py` to make sure the SVG can be processed with diffvg.
<img src="repo_image/creation.png"/>

**We release 500 more created sketches!** There are 560 vector multi-object sketches now! (`./data/raw/560sketches.zip`)

## 🎥 Generate A Video!
### 🚀 Quick Start
The scene decomposition (`_decomp.txt`), the stroke(point) assignment (`_semantic.txt`) and the motion plan (`_traj.txt`) of the **60 multi-object sketches**
are provided in `./data/processed`. The text instruction of each sketch can be found in `./data/raw/60sketches/caption.txt`. 
Run this command to get the animation of one sketch (*e.g.*, 'basketball5'):
```
CUDA_VISIBLE_DEVICES=0 python animate_mosketch.py \
        --sketch 'basketball5' \
        --caption 'The player soars through the air with a basketball, arm extended for an electrifying slam dunk to a hoop.' \
        --num_iter 500 \
        --seed 130 \
        --num_frames 20
```
The output video will be saved in `./output/basketball5`.

The scene decomposition, stroke(point) assignment and motion plan of the **500 more created sketches** are provided in `./data/processed/560sketches.zip`.
**Use the above command to animate them!**

<hr>

### 👩‍🎨 Animate Your Own Sketch
The scene decomposition, the stroke(point) assignment and the motion plan of your own multi-object sketch should be provided before animation,
and the format should follow the 60 created sketches. Make a new folder `./data/processed/Yours`, and put the vector sketch (`Yours.svg`) in it.
#### 🧩 Scene Decomposition
We use LLM to get the scene decomposition of the multi-object sketch. The LLM is not limited.
We recommend GPT-4, especially ChatGPT-4, to get the result and check it in real time. 
We should provide the sketch (`Yours.png`) and the text caption (`Yours_Text_Instruction`).
The GPT-4 instruction and examples are provided in `./data/examples-for-scene-decomposition`. 
Save the result in `./data/processed/Yours/Yours_decomp.txt`, 
and the format should be the same as the 60 created sketches (*e.g.*, `./data/processed/aircrafter3/aircrafter3_decomp.txt`). 
#### 🧮 Stroke(point) Assignment
The stroke(point) assignment aims to assign the strokes, as well as the control points, to their belonging objects. 
The stroke(point) assignment is actually the object segmentation task in vector sketch. 
We employ [GoundingDino](https://github.com/IDEA-Research/Grounded-Segment-Anything) to conduct 
object grounding on the multi-object sketch, and then assign strokes to objects based on the bounding boxes. 

First, Your should install [GoundingDino](https://github.com/IDEA-Research/Grounded-Segment-Anything).
Then copy the code `MoSketch/stroke_assignment.py` to the GoundingDino project.
Make a new folder `sketch` in GoundingDino project, and copy the SVG `Yours.svg` and PNG `Yours.png` of the sketch in it.
Run `Grounded-Segment-Anything/stroke_assignment.py` (do not forget adding object names in the parameter`--text_prompt`):
```
export CUDA_VISIBLE_DEVICES=0
python stroke_assignment.py \
  --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --grounded_checkpoint groundingdino_swint_ogc.pth \
  --sam_checkpoint sam_vit_h_4b8939.pth \
  --sketch_dir "sketch" \
  --sketch_img Yours.png \
  --box_threshold 0.2 \
  --text_threshold 0.2 \
  --iou_w 1.0 \
  --text_prompt Your_Object_Names \
  --device "cuda"
```
<div style="display: flex; justify-content: space-between;">
  <img src="repo_image/plan/1.png" width="12%">
  <img src="repo_image/plan/1c.png" width="12%">
  <img src="repo_image/plan/2.png" width="12%">
  <img src="repo_image/plan/2c.png" width="12%">
  <img src="repo_image/plan/3.png" width="12%">
  <img src="repo_image/plan/3c.png" width="12%">
  <img src="repo_image/plan/4.png" width="12%">
  <img src="repo_image/plan/4c.png" width="12%">
</div>

<div style="display: flex; justify-content: space-between;">
  <img src="repo_image/plan/5.png" width="12%">
  <img src="repo_image/plan/5c.png" width="12%">
  <img src="repo_image/plan/6.png" width="12%">
  <img src="repo_image/plan/6c.png" width="12%">
  <img src="repo_image/plan/7.png" width="12%">
  <img src="repo_image/plan/7c.png" width="12%">
  <img src="repo_image/plan/8.png" width="12%">
  <img src="repo_image/plan/8c.png" width="12%">
</div>

The stroke(point) assignment are saved in `Yours_semantic.txt`, which lists objects and their strokes (IDs in SVG).
Objects' bounding boxes are written in `Yours_bbox.txt`. `Yours_color.svg` is the visualization of stroke(point) assignment, 
and you can check it with color-object pairs printed in the output. Copy these files to the processed folder (`MoSketch/data/processed/Yours`).

**Note that:**
* `--box_threshold` and `--text_threshold` are the semantic thresholds in GoundingDino. 
`--iou_w` is a parameter about the object overlap (more `--iou_w` means less tolerance for overlap).
Adjust these three parameters flexibly during the stroke(point) assignment.
* Sometimes object names should be replaced to get the correct object grounding in GoundingDino, 
*e.g.*, 'basketball` &rarr; 'ball', 'player' &rarr; 'man'.
* If there are more than one objects in sketch sharing the same name, repeat the name in `--text_prompt`. For example, 
'hurdle9' in the 60 created sketches has three athletes:
```
export CUDA_VISIBLE_DEVICES=0
python stroke_assignment.py \
  --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --grounded_checkpoint groundingdino_swint_ogc.pth \
  --sam_checkpoint sam_vit_h_4b8939.pth \
  --sketch_dir "sketch" \
  --sketch_img huldres6.png \
  --box_threshold 0.2 \
  --text_threshold 0.2 \
  --iou_w 1.0 \
  --text_prompt "hurdle,athlete,athlete,athlete" \
  --device "cuda"
```

#### 🚗 Motion Planning
We use LLM to get the motion plan of the multi-object sketch. 
We also recommend ChatGPT-4 to get the result and check it in real time. 
We should provide the sketch (`Yours.png`) and the text caption (`Yours_Text_Instruction`) 
and the object bounding boxes (already saved in `Yours_bbox.txt` during stroke(point) assignment).
The GPT-4 instruction and examples are provided in `./data/examples-for-motion-planning`. 
Save the result in `./data/processed/Yours/Yours_traj.txt`, 
and the format should be the same as the 60 created sketches (*e.g.*, `./data/processed/aircrafter3/aircrafter3_traj.txt`). 

We **highly recommend** to check the motion plan in time. Run `./view_plan.py` to visualize the motion plan. 
The motion plan should be aligned with `Yours_Text_Instruction`.
If you are not satisfy with the result, instruct the LLM for modification in time.
The incorrect motion planning will lead to the failed animation.
```
python view_plan.py \
  --sketch_dir './data/processed/Yours' \
  --sketch_name 'Yours' \
  --frame_num 20
```
The motion plan will be saved as `./data/processed/Yours/Yours_color.gif`

<div style="display: flex; justify-content: space-between;">
  <img src="repo_image/plan/1.png" width="12%">
  <img src="repo_image/plan/1.gif" width="12%">
  <img src="repo_image/plan/2.png" width="12%">
  <img src="repo_image/plan/2.gif" width="12%">
  <img src="repo_image/plan/3.png" width="12%">
  <img src="repo_image/plan/3.gif" width="12%">
  <img src="repo_image/plan/4.png" width="12%">
  <img src="repo_image/plan/4.gif" width="12%">
</div>

<div style="display: flex; justify-content: space-between;">
  <img src="repo_image/plan/5.png" width="12%">
  <img src="repo_image/plan/5.gif" width="12%">
  <img src="repo_image/plan/6.png" width="12%">
  <img src="repo_image/plan/6.gif" width="12%">
  <img src="repo_image/plan/7.png" width="12%">
  <img src="repo_image/plan/7.gif" width="12%">
  <img src="repo_image/plan/8.png" width="12%">
  <img src="repo_image/plan/8.gif" width="12%">
</div>

<hr>

After getting the scene decomposition (`Yours_decomp.txt`), the stroke(point) assignment (`Yours_semantic.txt`) 
and the motion plan (`Yours_traj.txt`) of your own multi-object sketch in the folder `./data/processed/Yours`, 
run `./animate_mosketch.py` to get the final animation of your own sketch: 
```
CUDA_VISIBLE_DEVICES=0 python animate_mosketch.py \
        --sketch 'Yours' \
        --caption Yours_Text_Instruction \
        --num_iter 500 \
        --seed 130 \
        --num_frames 20
```
The output video will be saved in `./output/Yours`.

## 🤝 Acknowledgement
This implementation relies on resources from [Live-Sketch](https://github.com/yael-vinker/live_sketch), [FlipSketch](https://github.com/hmrishavbandy/FlipSketch) and [GoundingDino](https://github.com/IDEA-Research/Grounded-Segment-Anything), 
we thank the original authors for their excellent contributions and for making their work publicly available.

## 📧 Contact
If you have any questions, please raise an issue or contact us at [liujingyu2023@ruc.edu.cn](mailto:liujingyu2023@ruc.edu.cn).

## 📜 Licence
This work is licensed under a **[MIT](https://opensource.org/licenses/MIT)** License.

## 📎 Citation
If you find this work useful, please consider cite this paper:

```bibtex
@inproceedings{liu2025multi,
  title={Multi-Object Sketch Animation by Scene Decomposition and Motion Planning},
  author={Liu, Jingyu and Xin, Zijie and Fu, Yuhan and Zhao, Ruixiang and Lan, Bangxiang and Li, Xirong},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year={2025}
}
```



