import os
import numpy as np
import torch
import imageio
import shutil
import pydiffvg
import cairosvg
import argparse


def read_traj(traj_txt, all_point, semantic_index_dict, frame_num):
    points_allframe = torch.zeros_like(all_point).unsqueeze(0).repeat(frame_num, 1, 1, 1)  # fr, num, 4, 2
    with open(traj_txt, 'r') as file:
        for line in file:
            if len(line.strip()) != 0:
                line_list = line.strip().split('\t')
                semantic = line_list[0]
                index = semantic_index_dict[semantic]
                object_xywh_list = line_list[1][1:-1].replace(' ', '').replace('\n', '').split('],[')
                origin_object_w = float(object_xywh_list[0].split(',')[2])
                origin_object_h = float(object_xywh_list[0].split(',')[3])
                origin_object_x = float(object_xywh_list[0].split(',')[0]) + origin_object_w / 2
                origin_object_y = float(object_xywh_list[0].split(',')[1]) + origin_object_h / 2
                object_origin_center = torch.tensor([origin_object_x, origin_object_y]).unsqueeze(0)  # 1, 2
                obj_centers = []
                obj_scales = []
                for i in range(frame_num):
                    object_w = float(object_xywh_list[i].split(',')[2])
                    object_h = float(object_xywh_list[i].split(',')[3])
                    object_x = float(object_xywh_list[i].split(',')[0]) + object_w / 2
                    object_y = float(object_xywh_list[i].split(',')[1]) + object_h / 2
                    obj_centers.append(torch.Tensor([object_x, object_y]).unsqueeze(0))
                    obj_scale_w = object_w / origin_object_w
                    obj_scale_h = object_h / origin_object_h
                    obj_scales.append(torch.Tensor([obj_scale_w, obj_scale_h]).unsqueeze(0))
                obj_centers = torch.cat(obj_centers, dim=0)  # frame_num, 2
                obj_scales = torch.cat(obj_scales, dim=0)  # frame_num, 2
                # 计算
                object_delta = all_point[index] - object_origin_center.unsqueeze(0)  # group_size, 4, 2
                object_delta_strike = object_delta.unsqueeze(0).repeat(frame_num, 1, 1, 1) \
                                      * obj_scales.unsqueeze(1).unsqueeze(1)  # frame, group_size, 4, 2
                object_strike = object_delta_strike + obj_centers.unsqueeze(1).unsqueeze(1)  # strike, group_size, 4, 2
                points_allframe[:, index] = object_strike

    return points_allframe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sketch_dir", type=str, required=True)
    parser.add_argument("--sketch_name", type=str, required=True)
    parser.add_argument("--frame_num", type=int, default=16)

    args = parser.parse_args()
    sketch_dir = args.sketch_dir
    sketch_name = args.sketch_name
    frame_num = args.frame_num
    start_svg_path = os.path.join(sketch_dir, sketch_name + '_color.svg')
    semantic_index_txt = os.path.join(sketch_dir, sketch_name + '_semantic.txt')
    traj_txt = os.path.join(sketch_dir, sketch_name + '_traj.txt')

    semantic_index_dict = {}
    with open(semantic_index_txt, 'r') as file:
        for line in file:
            if len(line.strip()) != 0:
                semantic = line.strip().split('\t')[0]
                index = line.strip().split('\t')[1].split(',')
                semantic_index_dict[semantic] = []
                for i in index:
                    semantic_index_dict[semantic].append(int(i))

    # init the canvas_width, canvas_height
    canvas_width, canvas_height, shapes_init_, shape_groups_init = pydiffvg.svg_to_scene(start_svg_path)
    point_num = 0
    stroke_num = 0
    all_points_ = []
    for s_ in shapes_init_:
        point_num += s_.points.shape[0]
        all_points_.append(s_.points)
        stroke_num += 1
    print(f"{sketch_name} contains {point_num} points")
    if point_num % stroke_num != 0:
        print(f"{sketch_name} is not a multiple of 4 points")
    all_points_ = torch.vstack(all_points_).reshape(-1, 4, 2)  # num, 4, 2

    # read traj
    points_allframe = read_traj(traj_txt, all_points_, semantic_index_dict, frame_num)  # frame, num, 4, 2
    frames_init, frames_svg = [], []
    for i in range(frame_num):
        new_shapes, new_shape_groups = [], []
        for j in range(len(shapes_init_)):
            shape, shapes_group = shapes_init_[j], shape_groups_init[j]
            path = pydiffvg.Path(
                num_control_points=shape.num_control_points, points=points_allframe[i, j],
                stroke_width=shape.stroke_width, is_closed=shape.is_closed)
            new_shapes.append(path)
            path_group = pydiffvg.ShapeGroup(
                shape_ids=torch.tensor([len(new_shapes) - 1]),
                fill_color=shapes_group.fill_color,
                stroke_color=shapes_group.stroke_color)
            new_shape_groups.append(path_group)
        scene_args = pydiffvg.RenderFunction.serialize_scene(canvas_width, canvas_height, new_shapes,
                                                             new_shape_groups)
        cur_im = pydiffvg.RenderFunction.apply(canvas_width, canvas_height, 2, 2, 0, None, *scene_args)
        cur_im = cur_im[:, :, 3:4] * cur_im[:, :, :3] + \
                 torch.ones(cur_im.shape[0], cur_im.shape[1], 3).type_as(cur_im) * (1 - cur_im[:, :, 3:4])
        cur_im = cur_im[:, :, :3]
        frames_init.append(cur_im)
        frames_svg.append((new_shapes, new_shape_groups))
    frames_output = []
    for i in range(frame_num):
        cur_im = frames_init[i]
        cur_im = cur_im[:, :, :3].detach().cpu().numpy()
        cur_im = (cur_im * 255).astype(np.uint8)
        frames_output.append(cur_im)
    # save
    temp_dir = os.path.join(sketch_dir, 'temp')
    os.mkdir(temp_dir)
    for i in range(len(frames_svg)):
        pydiffvg.save_svg(f"{temp_dir}/frame{i:03d}.svg", canvas_width, canvas_height, frames_svg[i][0],
                          frames_svg[i][1])
        cairosvg.svg2png(url=f"{temp_dir}/frame{i:03d}.svg",
                         write_to=f"{temp_dir}/frame{i:03d}.png",
                         scale=1, background_color="white")
    gif_dest_path = os.path.join(sketch_dir, sketch_name + '_color.gif')
    png_filenames = sorted([k for k in os.listdir(temp_dir) if "png" in k])
    images = []
    for filename in png_filenames:
        im = imageio.imread(f"{temp_dir}/{filename}")
        images.append(im)
    imageio.mimsave(f"{gif_dest_path}", images, 'GIF', loop=0, fps=8)
    shutil.rmtree(temp_dir)

main()

