import torch
import pydiffvg
import numpy as np
import imageio
import os
import cairosvg
from torchvision import transforms


def get_ln_rpe_check(alpha, beta, gamma, regionum):
    node_rpe_check = np.zeros(regionum+1)
    node_rpe_check[1:alpha] = np.arange(1, alpha)
    for i in range(alpha, gamma+1):
        node_rpe_check[i] = alpha + np.floor((beta - alpha) * np.log(i / alpha) / np.log(gamma / alpha))
    for i in range(gamma+1, regionum+1):
        node_rpe_check[i] = beta

    return node_rpe_check


def read_traj(traj_str):
    object_xywh_list = traj_str[1:-1].replace(' ', '').split('],[')
    frame_num = len(object_xywh_list)
    # pdb.set_trace()
    origin_object_w = float(object_xywh_list[0].split(',')[2])
    origin_object_h = float(object_xywh_list[0].split(',')[3])
    obj_centers = []
    obj_scales = []
    obj_whs = []
    for i in range(frame_num):
        object_w = float(object_xywh_list[i].split(',')[2])
        object_h = float(object_xywh_list[i].split(',')[3])
        object_x = float(object_xywh_list[i].split(',')[0]) + object_w / 2
        object_y = float(object_xywh_list[i].split(',')[1]) + object_h / 2
        obj_centers.append(torch.Tensor([object_x, object_y]).unsqueeze(0))
        obj_scale_w = object_w / origin_object_w
        obj_scale_h = object_h / origin_object_h
        obj_scales.append(torch.Tensor([obj_scale_w, obj_scale_h]).unsqueeze(0))
        obj_whs.append(torch.Tensor([object_w, object_h]).unsqueeze(0))
    obj_centers = torch.cat(obj_centers, dim=0)  # frame_num, 2
    obj_scales = torch.cat(obj_scales, dim=0)  # frame_num, 2
    obj_whs = torch.cat(obj_whs, dim=0)  # frame_num, 2

    return torch.cat((obj_centers, obj_scales, obj_whs), dim=1)


def get_polar_rpe(src_nodes, tgt_nodes, rpr_dis_alpha, rpr_dis_beta, rpr_dis_gamma, rpr_dis_regionnum, rpr_angle):
    """
        positional encodings in Transformer in Motion Refinement Network.
        ref paper: "Transformer-based stroke relation encoding for online handwriting and sketches"
        this encoding scheme replace spatial encodings in Cartesian Coordinates (x, y) to Polar Coordinates (angle, distance)
    """
    num_1 = src_nodes.shape[0]
    num_2 = tgt_nodes.shape[0]
    src_nodes_repeat = np.repeat(np.expand_dims(src_nodes, axis=1), num_2, axis=1)
    tgt_nodes_repeat = np.repeat(np.expand_dims(tgt_nodes, axis=0), num_1, axis=0)
    rpe_mat = tgt_nodes_repeat - src_nodes_repeat  # num1, num2, 2
    rpe_mat_x = rpe_mat[:, :, 0]
    rpe_mat_y = rpe_mat[:, :, 1]
    # angle
    rpe_mat_angle = np.arctan2(rpe_mat_y, rpe_mat_x)
    rpe_mat_angle_norm = (rpe_mat_angle + np.pi) * rpr_angle / (2 * np.pi)
    rpe_angle_int = rpe_mat_angle_norm.astype(int) # num1, num2
    rpe_angle_int = np.where(rpe_angle_int > rpr_angle-1, rpr_angle-1, rpe_angle_int)
    # distance
    rpe_mat_dis = np.linalg.norm(rpe_mat, axis=2, keepdims=False)
    rpe_mat_dis_region = rpe_mat_dis * rpr_dis_regionnum
    rpe_dis_check = get_ln_rpe_check(rpr_dis_alpha, rpr_dis_beta, rpr_dis_gamma, rpr_dis_regionnum)
    rpe_mat_dis_positive = np.where(rpe_mat_dis_region > 0, 1, 0)
    rpe_mat_dis_region_ = rpe_mat_dis_region.astype(int) + rpe_mat_dis_positive
    rpe_mat_dis_region_ = np.where(rpe_mat_dis_region_ > rpr_dis_regionnum, rpr_dis_regionnum, rpe_mat_dis_region_)
    rpe_dis_int = rpe_dis_check[rpe_mat_dis_region_]

    return rpe_angle_int, rpe_dis_int


def get_augmentations(cfg):
    augemntations = []
    augemntations.append(transforms.RandomPerspective(
        fill=1, p=1.0, distortion_scale=0.5))
    augemntations.append(transforms.RandomResizedCrop(
        size=(cfg.render_size_h, cfg.render_size_w), scale=(0.4, 1), ratio=(1.0, 1.0)))
    augment_trans = transforms.Compose(augemntations)
    return augment_trans


def frames_to_vid(video_frames, output_vid_path):
    """
    Saves an mp4 file from the given frames
    """
    writer = imageio.get_writer(output_vid_path, fps=8)
    for im in video_frames:
        writer.append_data(im)
    writer.close()

def render_frames_to_tensor(frames_shapes, frames_shapes_grous, w, h, render, device):
    """
    Given a list with the points parameters, render them frame by frame and return a tensor of the rasterized frames ([16, 256, 256, 3])
    """
    frames_init = []
    for i in range(len(frames_shapes)):
        shapes = frames_shapes[i]
        shape_groups = frames_shapes_grous[i]
        scene_args = pydiffvg.RenderFunction.serialize_scene(w, h, shapes, shape_groups)
        cur_im = render(w, h, 2, 2, 0, None, *scene_args)
    
        cur_im = cur_im[:, :, 3:4] * cur_im[:, :, :3] + \
               torch.ones(cur_im.shape[0], cur_im.shape[1], 3, device=device) * (1 - cur_im[:, :, 3:4])
        cur_im = cur_im[:, :, :3]
        frames_init.append(cur_im)
    return torch.stack(frames_init)

def save_mp4_from_tensor(frames_tensor, output_vid_path):
    # input is a [16, 256, 256, 3] video
    frames_copy = frames_tensor.clone()
    frames_output = []
    for i in range(frames_copy.shape[0]):
        cur_im = frames_copy[i]
        cur_im = cur_im[:, :, :3].detach().cpu().numpy()
        cur_im = (cur_im * 255).astype(np.uint8)
        frames_output.append(cur_im)
    frames_to_vid(frames_output, output_vid_path=output_vid_path)
    
def save_vid_svg(frames_svg, output_folder, step, w, h):
    if not os.path.exists(f"{output_folder}/svg_step{step}"):
        os.mkdir(f"{output_folder}/svg_step{step}")
    for i in range(len(frames_svg)):
        pydiffvg.save_svg(f"{output_folder}/svg_step{step}/frame{i:03d}.svg", w, h, frames_svg[i][0], frames_svg[i][1])

def svg_to_png(path_to_svg_files, dest_path):
    svgs = sorted(os.listdir(path_to_svg_files))
    filenames = [k for k in svgs if "svg" in k]
    for filename in filenames:        
        dest_path_ = f"{dest_path}/{os.path.splitext(filename)[0]}.png"
        cairosvg.svg2png(url=f"{path_to_svg_files}/{filename}", write_to=dest_path_, scale=4, background_color="white")
  
def save_gif_from_pngs(path_to_png_files, gif_dest_path):
    pngs = sorted(os.listdir(path_to_png_files))
    filenames = [k for k in pngs if "png" in k]
    images = []
    for filename in filenames:
        im = imageio.imread(f"{path_to_png_files}/{filename}")
        images.append(im)
    imageio.mimsave(f"{gif_dest_path}", images, 'GIF', loop=0, fps=8)

def save_hq_video(path_to_outputs, iter_=1000):
    dest_path_png = f"{path_to_outputs}/png_files_ite{iter_}"
    os.makedirs(dest_path_png, exist_ok=True)

    svg_to_png(f"{path_to_outputs}/svg_logs/svg_step{iter_}", dest_path_png)

    gif_dest_path = f"{path_to_outputs}/HQ_gif.gif"
    save_gif_from_pngs(dest_path_png, gif_dest_path)
    print(f"GIF saved to [{gif_dest_path}]")

