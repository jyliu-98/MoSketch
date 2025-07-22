import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
import pydiffvg

import pdb

import utils


class Painter(torch.nn.Module):
    def __init__(self,
                 args,
                 num_frames: int,
                 device):
        super(Painter, self).__init__()
        sketch_name = args.sketch
        self.svg_path = os.path.join(args.input_folder, f'{sketch_name}.svg')
        self.semantic_txt = os.path.join(args.input_folder, f'{sketch_name}_semantic.txt')
        self.prompt_txt = os.path.join(args.input_folder, f'{sketch_name}_decomp.txt')
        self.traj_txt = os.path.join(args.input_folder, f'{sketch_name}_traj.txt')
        self.num_frames = num_frames
        self.device = device
        self.move_scale = args.move_scale
        self.render = pydiffvg.RenderFunction.apply
        self.init_shapes(args)
        self.model = MotionRefinement(num_frames = num_frames,
                                      objects = self.semanticinput_index,
                                      hidden_dim = args.hidden_dim,
                                      rpr_angle = args.rpr_angle,
                                      rpr_dis = args.rpr_dis_beta,
                                      head_num = args.head_num,
                                      att_dim = args.att_dim,
                                      layer_num = args.layer_num,
                                      rotation_weight = args.rotation_weight,
                                      scale_weight = args.scale_weight,
                                      shear_weight = args.shear_weight,
                                      translation_weight = args.translation_weight
                                      )
        self.model.to(device)

    def init_shapes(self, args):
        # init the canvas_width, canvas_height
        self.canvas_width, self.canvas_height, shapes_init_, shape_groups_init = pydiffvg.svg_to_scene(self.svg_path)
        self.points_per_frame = 0
        all_points_ = []
        for s_ in shapes_init_:
            self.points_per_frame += s_.points.shape[0]
            all_points_.append(s_.points)
        all_points_ = torch.vstack(all_points_) # num*4, 2
        center_g = all_points_.mean(dim=0, keepdim=True) # 1, 2
        print(f"A single frame contains {self.points_per_frame} points")

        self.frames_shapes = shapes_init_
        self.frames_shapes_group = shape_groups_init

        # save the original center
        self.original_center = center_g.clone()
        self.original_center.requires_grad = False
        self.original_center = self.original_center.to(self.device)

        # =========================================================================================================
        # stroke features  ########################################################################################
        deltas_from_center = (all_points_ - center_g).to(self.device) # num*4, 2
        self.stroke_point = deltas_from_center.reshape(-1, 4, 2) # num, 4, 2
        self.stroke_center = self.stroke_point.mean(dim=1, keepdim=True) # num, 1, 2
        self.stroke_input = torch.cat((self.stroke_point, self.stroke_center), dim=1).flatten(1) / self.canvas_height  # num, 10

        # semantic_strokeindex #################################################################################
        # self.semantic_txt: stroke assignment results
        with open(self.semantic_txt, 'r', encoding='gb18030', errors='ignore') as f:
            lines = f.readlines()
        self.semantic_index = {}  # key: object, value: strokes' ids
        for line in lines:
            line = line.strip()
            object_name = line.split('\t')[0]
            group_index_ = line.split('\t')[1].split(',')
            group_index = []
            for i_ in group_index_:
                group_index.append(int(i_))
            self.semantic_index[object_name] = group_index

        # decomposted instructions :##################################################################################
        with open(self.prompt_txt, 'r', encoding='gb18030', errors='ignore') as f:
            lines = f.readlines()
        # prompts which involve one object
        self.node_prompt_index = {}  # key: instruction, value: involved strokes' ids
        # prompts which involve at least two object
        self.edge_prompt_index = {}  # key: instruction, value: involved strokes' ids
        for line in lines:
            line = line.strip()
            prompt = line.split('\t')[0]
            object_list = line.split('\t')[1].split(',')
            involved_index_ = []
            for group in object_list:
                involved_index_ += self.semantic_index[group.strip()]
            if len(object_list) == 1:
                self.node_prompt_index[prompt] = involved_index_
            elif len(object_list) > 1:
                self.edge_prompt_index[prompt] = involved_index_
            else:
                raise NotImplementedError("have wrong in prompt-index process.")

        # motion planning results #######################################################################
        semantic_traj_dict = {}  # key: object, value: series of xywh
        with open(self.traj_txt, 'r') as file:
            for line in file:
                if len(line.strip()) != 0:
                    semantic = line.strip().split('\t')[0]
                    traj_th = utils.read_traj(line.strip().split('\t')[1])  # fr, 6
                    semantic_traj_dict[semantic] = traj_th[:args.num_frames]

        # objects' center and width and height of all frames  #################################################
        self.semantic_center = {}
        self.semantic_wh = {}
        self.semantic_scales = {}
        for name in self.semantic_index.keys():
            self.semantic_center[name] = (semantic_traj_dict[name][:, :2] - center_g).to(self.device)  # fr, 2
            self.semantic_wh[name] = semantic_traj_dict[name][:, 4:6].to(self.device)  # fr, 2
            self.semantic_scales[name] = semantic_traj_dict[name][:, 2:4].to(self.device)  # fr, 2

        # object input features  #############################################################################
        semantic_input_xywh = []
        semantic_input_scale = []
        self.semanticinput_index = [] # all strokes' ids ordered by objects
        for name, xy in self.semantic_center.items():
            semantic_input_xywh.append(torch.cat((xy, self.semantic_wh[name]), dim=1).unsqueeze(0)) # 1, fr, 4
            semantic_input_scale.append(self.semantic_scales[name].unsqueeze(0)) # 1, fr, 2
            self.semanticinput_index.append(self.semantic_index[name])
        self.semantic_input = torch.cat(semantic_input_xywh, dim=0) / self.canvas_height  # obj, fr, 4
        self.semantic_scale = torch.cat(semantic_input_scale, dim=0)  # obj, fr, 2

        # position encodings #################################################################################
        # positional encodings in Transformer in Motion Refinement Network.
        # ref paper: "Transformer-based stroke relation encoding for online handwriting and sketches"
        # this encoding scheme replace spatial encodings in Cartesian Coordinates (x, y) to Polar Coordinates (angle, distance)
        rpe_src_tgt = np.array(torch.cat((self.semantic_input[:, 0, :2],
                                 self.stroke_input[:, 8:]), dim=0).cpu()) # obj_num+stroke_num, 2
        rpe_int_a_, rpe_int_d_ = utils.get_polar_rpe(rpe_src_tgt, 
                                                     rpe_src_tgt,
                                                     args.rpr_dis_alpha,
                                                     args.rpr_dis_beta,
                                                     args.rpr_dis_gamma,
                                                     args.rpr_dis_regionnum,
                                                     args.rpr_angle)  # obj_num+stroke_num, obj_num+stroke_num
        self.rpe_int_a = torch.LongTensor(rpe_int_a_).to(self.device)  # obj_num+stroke_num, obj_num+stroke_num
        self.rpe_int_d = torch.LongTensor(rpe_int_d_).to(self.device)  # obj_num+stroke_num, obj_num+stroke_num
        
        # mask
        obj_num = self.semantic_input.shape[0]
        st_num = self.stroke_input.shape[0]
        mask_mat1 = torch.zeros(obj_num, obj_num).to(self.device)
        mask_mat2 = torch.zeros(obj_num, st_num).to(self.device)
        mask_mat3 = -9e8 * torch.ones(st_num, obj_num).to(self.device)
        for i in range(st_num):
            for j, v in enumerate(self.semanticinput_index):
                if i in v:
                    mask_mat3[i, j] = 0
        mask_mat4 = -9e8 * torch.ones(st_num, st_num).to(self.device)
        for i in range(st_num):
            for v in self.semanticinput_index:
                if i in v:
                    for j in v:
                        mask_mat4[i, j] = 0
        self.mask = torch.cat((torch.cat((mask_mat1, mask_mat2), dim=1),
                               torch.cat((mask_mat3, mask_mat4), dim=1)), dim=0)  # obj_num+stroke_num, obj_num+stroke_num
        

    def render_frames_to_tensor(self, index=None):
        """
            index: strokes' ids, used when computing compositional SDS
        """
        frames_init, frames_svg, all_new_points = [], [], []

        prev_points = self.stroke_input[:, :8].reshape(-1, 4, 2).clone() * self.canvas_height + self.original_center.unsqueeze(0) # st_num, 4, 2

        # Delta Z_o + Delta Z_p in paper
        delta_prediction = self.model(self.semantic_input.unsqueeze(0), 
                                      self.semantic_scale.unsqueeze(0),
                                      self.stroke_input.unsqueeze(0), 
                                      self.rpe_int_a.unsqueeze(0), 
                                      self.rpe_int_d.unsqueeze(0), 
                                      self.mask.unsqueeze(0),
                                      self.semanticinput_index)  # st_num, frame, 4, 2

        # Delta Z_c in paper
        traj_delta = torch.zeros_like(delta_prediction)  # st_num, frame, 4, 2
        for i, strokes_ids in enumerate(self.semanticinput_index):
            traj_dxy = (self.semantic_input.unsqueeze(0)[:, i, :, :2] - self.semantic_input.unsqueeze(0)[:, i, [0], :2])  # 1, fr, 2
            traj_delta[strokes_ids] = traj_dxy.unsqueeze(2).repeat(len(strokes_ids), 1, 4, 1)  # st_num, self.num_frames, 4, 2

        # rendering svg frame by frame
        for i in range(self.num_frames):
            shapes, shapes_groups = self.frames_shapes, self.frames_shapes_group
            new_shapes, new_shape_groups, frame_new_points = [], [], []  # for SVG frames saving
            points_cur_frame = prev_points + delta_prediction[:, i] * self.move_scale + traj_delta[:, i] * self.canvas_height # st_num, 4, 2 

            for j in range(len(shapes)):
                if not index:
                    shape, shapes_group = shapes[j], shapes_groups[j]
                    points_vars = shape.points.clone()  # 4, 2
                    points_vars[:, 0] = points_cur_frame[j, :, 0]
                    points_vars[:, 1] = points_cur_frame[j, :, 1]

                    frame_new_points.append(points_vars.to(self.device))
                    path = pydiffvg.Path(
                        num_control_points=shape.num_control_points, points=points_vars,
                        stroke_width=shape.stroke_width, is_closed=shape.is_closed)
                    new_shapes.append(path)
                    path_group = pydiffvg.ShapeGroup(
                        shape_ids=torch.tensor([len(new_shapes) - 1]),
                        fill_color=shapes_group.fill_color,
                        stroke_color=torch.tensor([0, 0, 0, 1]))
                    new_shape_groups.append(path_group)
                else:
                    if j in index:
                        shape, shapes_group = shapes[j], shapes_groups[j]
                        points_vars = shape.points.clone()  # 4, 2
                        points_vars[:, 0] = points_cur_frame[j, :, 0]
                        points_vars[:, 1] = points_cur_frame[j, :, 1]

                        frame_new_points.append(points_vars.to(self.device))
                        path = pydiffvg.Path(
                            num_control_points=shape.num_control_points, points=points_vars,
                            stroke_width=shape.stroke_width, is_closed=shape.is_closed)
                        new_shapes.append(path)
                        path_group = pydiffvg.ShapeGroup(
                            shape_ids=torch.tensor([len(new_shapes) - 1]),
                            fill_color=shapes_group.fill_color,
                            stroke_color=torch.tensor([0, 0, 0, 1]))
                        new_shape_groups.append(path_group)

            scene_args = pydiffvg.RenderFunction.serialize_scene(self.canvas_width, self.canvas_height, new_shapes,
                                                                 new_shape_groups)
            cur_im = self.render(self.canvas_width, self.canvas_height, 2, 2, 0, None, *scene_args)
            cur_im = cur_im[:, :, 3:4] * cur_im[:, :, :3] + \
                     torch.ones(cur_im.shape[0], cur_im.shape[1], 3, device=self.device) * (1 - cur_im[:, :, 3:4])
            cur_im = cur_im[:, :, :3]
            frames_init.append(cur_im)
            frames_svg.append((new_shapes, new_shape_groups))
            all_new_points.append(frame_new_points)

        return torch.stack(frames_init), frames_svg, all_new_points


    def render_frames_to_tensor_direct_optim(self):
        frames_init, frames_svg, points_init_frame = [], [], []
        for i in range(self.num_frames):
            shapes = self.frames_shapes
            shapes_groups = self.frames_shapes_group
            new_shapes, new_shape_groups = [], []

            deltas_from_center_cur_frame = self.stroke_point
            for j in range(len(shapes)):
                shape, shapes_group = shapes[j], shapes_groups[j]
                point_delta_leanred = deltas_from_center_cur_frame[j]
                points_vars = shape.points.clone()

                points_vars[:, 0] = point_delta_leanred[:, 0] + self.original_center[0][0]
                points_vars[:, 1] = point_delta_leanred[:, 1] + self.original_center[0][1]
                if i == 0: # only for a single frame
                    points_init_frame.append(points_vars)

                path = pydiffvg.Path(
                    num_control_points=shape.num_control_points, points=points_vars,
                    stroke_width=shape.stroke_width, is_closed=shape.is_closed)
                new_shapes.append(path)
                path_group = pydiffvg.ShapeGroup(
                    shape_ids=torch.tensor([len(new_shapes) - 1]),
                    fill_color=shapes_group.fill_color,
                    stroke_color=torch.tensor([0, 0, 0, 1]))
                new_shape_groups.append(path_group)

            scene_args = pydiffvg.RenderFunction.serialize_scene(self.canvas_width, self.canvas_height, new_shapes,
                                                                 new_shape_groups)
            cur_im = self.render(self.canvas_width, self.canvas_height, 2, 2, 0, None, *scene_args)

            cur_im = cur_im[:, :, 3:4] * cur_im[:, :, :3] + \
                     torch.ones(cur_im.shape[0], cur_im.shape[1], 3, device=self.device) * (1 - cur_im[:, :, 3:4])
            cur_im = cur_im[:, :, :3]
            frames_init.append(cur_im)
            frames_svg.append((new_shapes, new_shape_groups))

        return torch.stack(frames_init), frames_svg, points_init_frame

    def get_local_params(self):
        project_stroke_p = list(self.model.project_stroke.parameters())
        project_group_p = list(self.model.project_group.parameters())
        rpe_enc_a_p = list(self.model.rpe_enc_a.parameters())
        rpe_enc_d_p = list(self.model.rpe_enc_d.parameters())
        transformers_p = []
        for transformer in self.model.transformers:
            transformers_p += list(transformer.parameters())
        local_proj_p = list(self.model.local_proj.parameters())

        return project_stroke_p + project_group_p + rpe_enc_a_p + \
            rpe_enc_d_p + transformers_p + local_proj_p

    def get_global_params(self):
        group_gather_proj_p = list(self.model.group_gather_proj.parameters())
        group_proj_share_p = list(self.model.group_proj_share.parameters())
        frames_rigid_translation_p = list(self.model.frames_rigid_translation.parameters())
        frames_rigid_rotation_p = list(self.model.frames_rigid_rotation.parameters())
        frames_rigid_shear_p = list(self.model.frames_rigid_shear.parameters())
        frames_rigid_scale_p = list(self.model.frames_rigid_scale.parameters())
        
        return group_gather_proj_p + group_proj_share_p + frames_rigid_translation_p + frames_rigid_rotation_p + \
            frames_rigid_shear_p + frames_rigid_scale_p

    def log_state(self, output_path):
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        torch.save(self.model.state_dict(), f"{output_path}/model.pt")
        print(f"Model saved to {output_path}/model.pt")

        
# model ========================================================================================================================================
# ==============================================================================================================================================
class Transformer(nn.Module):

    def __init__(self, in_nfths, head_num, att_nfths, ffn_nfths, att_dropout=0., ffn_dropout=0.):
        super(Transformer, self).__init__()
        self.head_num = head_num
        self.att_nfths = att_nfths
        self.w_q = nn.Linear(in_nfths, head_num*att_nfths)
        self.w_k = nn.Linear(in_nfths, head_num*att_nfths)
        self.w_v = nn.Linear(in_nfths, head_num*att_nfths)
        self.w_o = nn.Linear(head_num*att_nfths, in_nfths)
        self.ffn_linear1 = nn.Linear(in_nfths, ffn_nfths)
        self.ffn_linear2 = nn.Linear(ffn_nfths, in_nfths)
        self.ln1 = nn.LayerNorm(in_nfths)
        self.ln2 = nn.LayerNorm(in_nfths)
        self.att_dropout = nn.Dropout(att_dropout)
        self.ffn_1_dropout = nn.Dropout(ffn_dropout)
        self.ffn_2_dropout = nn.Dropout(ffn_dropout)
        self.activation = nn.LeakyReLU()

    def forward(self, x_input, x_input_pos, padding_matrix):
        """
        x_input:            bs, obj_num+st_num*4, dim
        x_input_pos:        bs, obj_num+st_num*4, obj_num+st_num*4, head_num
        padding_matrix      bs, obj_num+st_num*4, obj_num+st_num*4
        """

        batchsize = x_input.shape[0]
        token_num = x_input.shape[1]

        x_q = self.w_q(x_input).view(batchsize, token_num, self.head_num, self.att_nfths).transpose(1, 2)
        x_k = self.w_k(x_input).view(batchsize, token_num, self.head_num, self.att_nfths).transpose(1, 2).transpose(2, 3)
        x_v = self.w_v(x_input).view(batchsize, token_num, self.head_num, self.att_nfths).transpose(1, 2)

        x_k_attention = (torch.matmul(x_q, x_k) + x_input_pos.permute(0, 3, 1, 2)) / math.sqrt(self.att_nfths)
        unnorm_att = x_k_attention + padding_matrix.unsqueeze(1)
        attention_exp = F.softmax(unnorm_att, dim=-1)
        x_attention = torch.matmul(attention_exp, x_v).transpose(1, 2).reshape(batchsize, token_num, self.head_num * self.att_nfths)
        x_attention_out = self.w_o(x_attention)
        x_input = x_input + self.att_dropout(x_attention_out)
        x_input = self.ln1(x_input)
        x_input2 = self.ffn_linear2(self.ffn_1_dropout(self.activation(self.ffn_linear1(x_input))))
        x_input = x_input + self.ffn_2_dropout(x_input2)
        x_input = self.ln2(x_input)

        return x_input


class MotionRefinement(nn.Module):
    def __init__(self, num_frames, objects, hidden_dim, rpr_angle, rpr_dis, head_num, att_dim, layer_num,
                 rotation_weight=1e-2, scale_weight=5e-2, shear_weight=5e-2, translation_weight=1):
        super().__init__()

        self.num_frames = num_frames
        self.hidden_dim = hidden_dim

        # input projection
        self.project_stroke = nn.Sequential(nn.Linear(2, hidden_dim),
                                            nn.LayerNorm(hidden_dim),
                                            nn.LeakyReLU())
        self.project_group = nn.Sequential(nn.Linear(4 * num_frames, hidden_dim),
                                           nn.LayerNorm(hidden_dim),
                                           nn.LeakyReLU())
        
        # positional encoding
        self.rpe_enc_a = nn.Embedding(rpr_angle, head_num)
        self.rpe_enc_d = nn.Embedding(rpr_dis + 1, head_num)

        # transformers
        self.transformers = nn.ModuleList([
            Transformer(hidden_dim, head_num, att_dim, int(4 * hidden_dim))
            for _ in range(layer_num)])
        
        # External Motion Refinement ############################################
        self.rotation_weight = rotation_weight
        self.scale_weight = scale_weight
        self.shear_weight = shear_weight
        self.translation_weight = translation_weight
        obj_num = len(objects)

        # MLP
        self.group_gather_proj = nn.ModuleList([
                                nn.Sequential(nn.Linear(int(len(objects[ii]) * 4 * hidden_dim), hidden_dim),
                                              nn.LayerNorm(hidden_dim),
                                              nn.LeakyReLU(),
                                              nn.Linear(hidden_dim, hidden_dim),
                                              nn.LayerNorm(hidden_dim),
                                              nn.LeakyReLU())
                                for ii in range(obj_num)])
        self.group_proj_share = nn.ModuleList([
                                nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                              nn.LayerNorm(hidden_dim),
                                              nn.LeakyReLU(),
                                              nn.Linear(hidden_dim, hidden_dim),
                                              nn.LayerNorm(hidden_dim),
                                              nn.LeakyReLU())
                                for _ in range(obj_num)])

        # transformation parameters  ###################################################################
        self.frames_rigid_translation = nn.ModuleList([
                                nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                              nn.LayerNorm(hidden_dim),
                                              nn.LeakyReLU(),
                                              nn.Linear(hidden_dim, hidden_dim),
                                              nn.LayerNorm(hidden_dim),
                                              nn.LeakyReLU(),
                                              nn.Linear(hidden_dim, hidden_dim),
                                              nn.LayerNorm(hidden_dim),
                                              nn.LeakyReLU(),
                                              nn.Linear(hidden_dim, self.num_frames * 2))
                                for _ in range(obj_num)])
        self.frames_rigid_rotation = nn.ModuleList([
                                nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                              nn.LayerNorm(hidden_dim),
                                              nn.LeakyReLU(),
                                              nn.Linear(hidden_dim, self.num_frames * 1))
                                for _ in range(obj_num)])
        self.frames_rigid_shear = nn.ModuleList([
                                nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                              nn.LayerNorm(hidden_dim),
                                              nn.LeakyReLU(),
                                              nn.Linear(hidden_dim, self.num_frames * 2))
                                for _ in range(obj_num)])
        self.frames_rigid_scale = nn.ModuleList([
                                nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                              nn.LayerNorm(hidden_dim),
                                              nn.LeakyReLU(),
                                              nn.Linear(hidden_dim, self.num_frames * 2))
                                for _ in range(obj_num)])

        # Internal Motion Modeling ###################################################################
        self.local_proj = nn.ModuleList([
                                nn.Sequential(nn.Linear(int(len(objects[ii]) * 4 * hidden_dim), hidden_dim),
                                        nn.LayerNorm(hidden_dim),
                                        nn.LeakyReLU(),
                                        nn.Linear(hidden_dim, hidden_dim),
                                        nn.LayerNorm(hidden_dim),
                                        nn.LeakyReLU(),
                                        nn.Linear(hidden_dim, int(len(objects[ii]) * 4 * self.num_frames * 2)),
                                        )
            for ii in range(obj_num)])

        self.initialize_parameters()

    def initialize_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, obj_input, obj_scale, st_input0, rpe_a, rpe_d, mask, semantic_index):
        """
        obj_input:            1, g_num, fr, 4
        obj_scale:            1, g_num, fr, 2
        st_input0:          1, st_num, 10
        rpe_a:              1, g_num+st_num, g_num+st_num
        rpe_d:              1, g_num+st_num, g_num+st_num
        mask:               1, g_num+st_num, g_num+st_num
        semantic_index      [int_list]
        """

        obj_num = obj_input.shape[1]
        st_input = st_input0[:, :, :8].reshape(1, -1, 2)  # 1, st_num*4, 2
        obj_emb = self.project_group(obj_input.flatten(2))
        st_emb = self.project_stroke(st_input)
        obj_st_emb = torch.cat((obj_emb, st_emb), dim=1)  # 1, obj_num+st_num*4, d

        # rpe_refine
        rpe_a_1 = rpe_a[:, :obj_num, :obj_num]
        rpe_a_2 = rpe_a[:, :obj_num, obj_num:].unsqueeze(3).repeat(1, 1, 1, 4).flatten(2)
        rpe_a_3 = rpe_a[:, obj_num:, :obj_num].unsqueeze(2).repeat(1, 1, 4, 1).flatten(1, 2)
        rpe_a_4 = rpe_a[:, obj_num:, obj_num:].unsqueeze(2).repeat(1, 1, 4, 1).flatten(1, 2).unsqueeze(3).repeat(1, 1, 1, 4).flatten(2)
        rpe_a_refine = torch.cat((torch.cat((rpe_a_1, rpe_a_2), dim=2), 
                                 torch.cat((rpe_a_3, rpe_a_4), dim=2)), dim=1)  # 1, obj_num+st_num*4, obj_num+st_num*4
        
        rpe_d_1 = rpe_d[:, :obj_num, :obj_num]
        rpe_d_2  = rpe_d[:, :obj_num, obj_num:].unsqueeze(3).repeat(1, 1, 1, 4).flatten(2)
        rpe_d_3 = rpe_d[:, obj_num:, :obj_num].unsqueeze(2).repeat(1, 1, 4, 1).flatten(1, 2)
        rpe_d_4 = rpe_d[:, obj_num:, obj_num:].unsqueeze(2).repeat(1, 1, 4, 1).flatten(1, 2).unsqueeze(3).repeat(1, 1, 1, 4).flatten(2)
        rpe_d_refine = torch.cat((torch.cat((rpe_d_1, rpe_d_2), dim=2), 
                                 torch.cat((rpe_d_3, rpe_d_4), dim=2)), dim=1)  # 1, obj_num+st_num*4, obj_num+st_num*4
        
        rpe_emb_a = self.rpe_enc_a(rpe_a_refine)
        rpe_emb_d = self.rpe_enc_d(rpe_d_refine)
        polar_emb = rpe_emb_a + rpe_emb_d  # 1, obj_num+st_num*4, obj_num+st_num*4, dim

        # mask_refine
        mask1 = mask[:, :obj_num, :obj_num]  # 1, obj_num, obj_num
        mask2 = mask[:, :obj_num, obj_num:].unsqueeze(3).repeat(1, 1, 1, 4).flatten(2)  # 1, obj_num, st_num*4
        mask3 = mask[:, obj_num:, :obj_num].unsqueeze(2).repeat(1, 1, 4, 1).flatten(1, 2)  # 1, st_num*4, obj_num
        mask4 = mask[:, obj_num:, obj_num:].unsqueeze(2).repeat(1, 1, 4, 1).flatten(1, 2).unsqueeze(3).repeat(1, 1, 1, 4).flatten(2) # 1, st_num*4, st_num*4
        mask_refine = torch.cat((torch.cat((mask1, mask2), dim=2), 
                                 torch.cat((mask3, mask4), dim=2)), dim=1)  # 1, obj_num+st_num*4, obj_num+st_num*4

        for transformer in self.transformers:
            obj_st_emb = transformer(obj_st_emb, polar_emb, mask_refine)  # 1, obj_num+st_num*4, d

        # External Motion Refinement ========================================================================
        obj_h_emb = obj_st_emb[0, :obj_num]  # obj_num, dim
        # Delta Z_o !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        st_delta_obj = torch.zeros_like(st_input0[0][:, :8]).reshape(-1, 1, 4, 2).repeat(1, self.num_frames, 1, 1) # st_num, self.num_frames, 4, 2
        for i, index in enumerate(semantic_index):
            obj_h_share = self.group_proj_share[i](obj_h_emb[i].unsqueeze(0)) + \
                    self.group_gather_proj[i](obj_st_emb[0, obj_num:].reshape(-1, 4, self.hidden_dim)[index].reshape(1, -1))
            # calculate transform matrix parameters
            dx, dy = self.frames_rigid_translation[i](obj_h_share).reshape(1, self.num_frames, 2).chunk(2, axis=-1)
            dx = dx * self.translation_weight
            dy = dy * self.translation_weight

            theta = self.frames_rigid_rotation[i](obj_h_share).reshape(1, self.num_frames, 1) * self.rotation_weight
            cos_theta = torch.cos(theta)
            sin_theta = torch.sin(theta)

            shear_x, shear_y = self.frames_rigid_shear[i](obj_h_share).reshape(1, self.num_frames, 2).chunk(2, axis=-1)

            shear_x = shear_x * self.shear_weight
            shear_y = shear_y * self.shear_weight

            scale_x, scale_y = self.frames_rigid_shear[i](obj_h_share).reshape(1, self.num_frames, 2).chunk(2, axis=-1)
            
            scale_x = torch.ones_like(dx) * obj_scale[:, i, :, 0].unsqueeze(2) + scale_x * self.scale_weight
            scale_y = torch.ones_like(dy) * obj_scale[:, i, :, 1].unsqueeze(2) + scale_y * self.scale_weight

            # prepare transform matrix
            l1 = torch.concat([scale_x * (cos_theta - sin_theta * shear_x), scale_y * (cos_theta * shear_y - sin_theta), dx], axis=-1)
            l2 = torch.concat([scale_x * (sin_theta + cos_theta * shear_x), scale_y * (sin_theta * shear_y + cos_theta), dy], axis=-1)
            l3 = torch.concat([torch.zeros_like(dx), torch.zeros_like(dx), torch.ones_like(dx)], axis=-1)

            transform_mat = torch.stack([l1, l2, l3], axis=-2)  # 1, self.num_frames, 3, 3

            # transformation
            st_in_obj = st_input0[0][index][:, :8].reshape(-1, 2)  # involve_st_num * 4, 2
            st_in_obj_with_z = torch.concat([st_in_obj, torch.ones_like(st_in_obj)[:,0:1]], axis=-1).unsqueeze(0) \
                .unsqueeze(3).repeat(self.num_frames, 1, 1, 1) # self.num_frames, involve_st_num*4, 3, 1
            st_in_obj_trans = torch.matmul(transform_mat.transpose(0, 1).repeat(1, st_in_obj_with_z.shape[1], 1, 1), st_in_obj_with_z)[:, :, 0:2, :]\
                .reshape(self.num_frames, len(index), 4, 2).permute(1, 0, 2, 3) # involve_st_num, self.num_frames, 4, 2
            st_in_obj_delta = st_in_obj_trans - st_in_obj.reshape(-1, 1, 4, 2) # involve_st_num, self.num_frames, 4, 2
            st_delta_obj[index] = st_in_obj_delta

        # Internal Motion Modeling ========================================================================
        st_h_emb = obj_st_emb[0, obj_num:] # st_num*4, dim
        # Delta Z_p !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        st_delta_p = torch.zeros_like(st_input0[0][:, :8]).reshape(-1, 1, 4, 2).repeat(1, self.num_frames, 1, 1) # st_num, self.num_frames, 4, 2
        for i, index in enumerate(semantic_index):
            st_delta_p[index] = self.local_proj[i](st_h_emb.reshape(-1, 4, self.hidden_dim)[index].reshape(1, -1)) \
                            .reshape(-1, self.num_frames, 4, 2) # st_num, self.num_frames, 4, 2

        # Delta Z_o + Delta Z_p
        st_delta = st_delta_obj + st_delta_p

        return st_delta
    

# optimer ==========================================================================================================================
# ==================================================================================================================================
class PainterOptimizer:
    def __init__(self, args, painter):
        self.painter = painter
        self.lr_local = args.lr_local
        self.lr_base_global = args.lr_base_global
        self.lr_init = args.lr_init
        self.lr_final = args.lr_final
        self.lr_delay_mult = args.lr_delay_mult
        self.lr_delay_steps = args.lr_delay_steps
        self.max_steps = args.num_iter
        self.lr_lambda = lambda step: self.learning_rate_decay(step) / self.lr_init
        self.init_optimizers()

    def learning_rate_decay(self, step):
        if self.lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = self.lr_delay_mult + (1 - self.lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / self.lr_delay_steps, 0, 1))
        else:
            delay_rate = 1.
        t = np.clip(step / self.max_steps, 0, 1)
        log_lerp = np.exp(np.log(self.lr_init) * (1 - t) + np.log(self.lr_final) * t)
        return delay_rate * log_lerp

    def init_optimizers(self):
        global_frame_params = self.painter.get_global_params()
        self.global_delta_optimizer = torch.optim.Adam(global_frame_params, lr=self.lr_base_global,
                                                        betas=(0.9, 0.9), eps=1e-6)
        self.scheduler_global = LambdaLR(self.global_delta_optimizer, lr_lambda=self.lr_lambda, last_epoch=-1)

        points_delta_params = self.painter.get_local_params()
        self.points_delta_optimizer = torch.optim.Adam(points_delta_params, lr=self.lr_local,
                                                        betas=(0.9, 0.9), eps=1e-6)
        self.scheduler_points = LambdaLR(self.points_delta_optimizer, lr_lambda=self.lr_lambda, last_epoch=-1)

    def update_lr(self):
        self.scheduler_global.step()
        self.scheduler_points.step()

    def zero_grad_(self):
        self.points_delta_optimizer.zero_grad()
        self.global_delta_optimizer.zero_grad()

    def step_(self):
        self.global_delta_optimizer.step()
        self.points_delta_optimizer.step()

    def get_lr(self, optim="points"):
        if optim == "points":
            return self.points_delta_optimizer.param_groups[0]['lr']
        else:
            return None
