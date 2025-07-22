# Our code is based on Live-Sketch (https://github.com/yael-vinker/live_sketch), we really thank them for their codes.
from paint_mosketch import Painter, PainterOptimizer
from losses import SDSVideoLoss
import utils
import os
import torch
import pydiffvg
from tqdm import tqdm
from pytorch_lightning import seed_everything
import argparse
import shutil

def parse_arguments():
    parser = argparse.ArgumentParser()

    # General
    parser.add_argument("--input_dir", type=str, default="./data/processed")
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--sketch", type=str, default="basketball5")
    parser.add_argument("--caption", type=str, default="The player soars through the air with a basketball, arm extended for an electrifying slam dunk to a hoop.")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--num_frames", type=int, default=20)
    parser.add_argument("--num_iter", type=int, default=1000, help="Number of training iterations")

    # Motion Refinement Network
    # Transformer
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--head_num", type=int, default=4)
    parser.add_argument("--att_dim", type=int, default=64)
    parser.add_argument("--layer_num", type=int, default=2)
    # positional encodings in Transformer in Motion Refinement Network.
    # ref paper: "Transformer-based stroke relation encoding for online handwriting and sketches"
    # this encoding scheme replace spatial encodings in Cartesian Coordinates (x, y) to Polar Coordinates (angle, distance)
    parser.add_argument("--rpr_dis_alpha", type=int, default=5, help="parameter of distance")
    parser.add_argument("--rpr_dis_beta", type=int, default=20, help="parameter of distance")
    parser.add_argument("--rpr_dis_gamma", type=int, default=40, help="parameter of distance")
    parser.add_argument("--rpr_dis_regionnum", type=int, default=50, help="number of distance position")
    parser.add_argument("--rpr_angle", type=int, default=40, help="number of angle position")
    # MLP architecture (points)
    parser.add_argument("--predict_global_frame_deltas", type=float, default=1,
                        help="whether to predict a global delta per frame, the value is the weight of the output")
    parser.add_argument("--predict_only_global", action='store_true', help="whether to predict only global deltas")
    parser.add_argument("--inter_dim", type=int, default=128)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--normalize_input", type=int, default=0)
    parser.add_argument("--translation_layer_norm_weight", type=int, default=0)
    # transformation parameter scale weight
    parser.add_argument("--rotation_weight", type=float, default=0.01,
                        help="Scale factor for global transform matrix 'rotation' terms")
    parser.add_argument("--scale_weight", type=float, default=0.05,
                        help="Scale factor for global transform matrix 'scale' terms")
    parser.add_argument("--shear_weight", type=float, default=0.1,
                        help="Scale factor for global transform matrix 'shear' terms")
    parser.add_argument("--translation_weight", type=float, default=1,
                        help="Scale factor for global transform matrix 'translation' terms")
    parser.add_argument("--move_scale", type=float, default=1, help="Scale factor for glob")

    # Diffusion related & Losses
    parser.add_argument("--model_name", type=str, default="text-to-video-ms-1.7b")
    parser.add_argument("--guidance_scale", type=float, default=30)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--render_size_h", type=int, default=256)
    parser.add_argument("--render_size_w", type=int, default=256)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--sds_timestep_low", type=int, default=50) 
    parser.add_argument("--same_noise_for_frames", action="store_true", help="sample noise for one frame and repeat across all frames")
    parser.add_argument("-augment_frames", type=bool, default=True, help="whether to randomely augment the frames to prevent adversarial results")
    parser.add_argument("--use_xformers", action="store_true", help="Enable xformers for unet")
    parser.add_argument("--del_text_encoders", action="store_true", help="delete text encoder and tokenizer after encoding the prompts")
    parser.add_argument("--loss_node_weight", type=float, default=1.0,
                        help="loss weight for decomposed instructions with one object")
    parser.add_argument("--loss_edge_weight", type=float, default=1.0,
                        help="loss weight for decomposed instructions with at least two objects")

    # Optimization related
    parser.add_argument("--lr_base_global", type=float, default=0.0001, help="Base learning rate for the global path")
    parser.add_argument("--lr_local", type=float, default=0.005)
    parser.add_argument("--lr_init", type=float, default=0.002)
    parser.add_argument("--lr_final", type=float, default=0.0008)
    parser.add_argument("--lr_delay_mult", type=float, default=0.1)
    parser.add_argument("--lr_delay_steps", type=float, default=100)

    # Display related
    parser.add_argument("--display_iter", type=int, default=50)
    parser.add_argument("--save_vid_iter", type=int, default=100)

    args = parser.parse_args()
    seed_everything(args.seed)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pydiffvg.set_use_gpu(torch.cuda.is_available())

    return args


if __name__ == "__main__":
    cfg = parse_arguments()
    sketch_name = cfg.sketch
    cfg.input_folder = os.path.join(cfg.input_dir, sketch_name)
    cfg.output_folder = os.path.join(cfg.output_dir, sketch_name)
    os.makedirs(cfg.output_folder, exist_ok=True)
    os.makedirs(f"{cfg.output_folder}/svg_logs", exist_ok=True)
    os.makedirs(f"{cfg.output_folder}/mp4_logs", exist_ok=True)

    print("=" * 50)
    print("caption:", cfg.caption)
    print("=" * 50)

    # Everything about rasterization and curves is defined in the Painter class
    painter = Painter(cfg, num_frames=cfg.num_frames, device=cfg.device)
    optimizer = PainterOptimizer(cfg, painter)
    data_augs = utils.get_augmentations(cfg)

    # Just to test that the svg and initial frames were loaded as expected
    frames_tensor, frames_svg, points_init_frame = painter.render_frames_to_tensor_direct_optim()
    output_vid_path = f"{cfg.output_folder}/init_vid.mp4"
    utils.save_mp4_from_tensor(frames_tensor, output_vid_path)

    sds_loss = SDSVideoLoss(cfg, cfg.device)
    orig_frames = frames_tensor.unsqueeze(0).permute(0, 1, 4, 2, 3) # (K, 256, 256, 3) -> (1, K, 3, 256, 256)
    orig_frames = orig_frames.repeat(cfg.batch_size, 1, 1, 1, 1)

    t_range = tqdm(range(cfg.num_iter + 1))
    for step in t_range:
        logs = {}
        optimizer.zero_grad_()
        loss = 0

        # global ================================================================================================
        vid_tensor_g, frames_svg_g, new_points_g = painter.render_frames_to_tensor()
        x = vid_tensor_g.unsqueeze(0).permute(0, 1, 4, 2, 3).repeat(cfg.batch_size, 1, 1, 1, 1)
        if cfg.augment_frames:
            augmented_pair = data_augs(torch.cat([x.squeeze(0), orig_frames.squeeze(0)]))
            x_aug = augmented_pair[:cfg.num_frames].unsqueeze(0)
            orig_frames_aug = augmented_pair[cfg.num_frames:].unsqueeze(0)
        else:
            x_aug = x
            orig_frames_aug = orig_frames

        loss_sds = sds_loss(x_aug, cfg.caption)
        loss += loss_sds.detach()
        loss_sds.backward()

        # node ================================================================================================
        # SDS for decomposed instructions with one object.
        for k, v in painter.node_prompt_index.items():
            vid_tensor, frames_svg, new_points = painter.render_frames_to_tensor(v)
            x = vid_tensor.unsqueeze(0).permute(0, 1, 4, 2, 3).repeat(cfg.batch_size, 1, 1, 1, 1)
            if cfg.augment_frames:
                augmented_pair = data_augs(torch.cat([x.squeeze(0), orig_frames.squeeze(0)]))
                x_aug = augmented_pair[:cfg.num_frames].unsqueeze(0)
                orig_frames_aug = augmented_pair[cfg.num_frames:].unsqueeze(0)
            else:
                x_aug = x
                orig_frames_aug = orig_frames

            loss_sds = sds_loss(x_aug, k) * cfg.loss_node_weight / len(painter.node_prompt_index.keys())
            loss += loss_sds.detach()
            loss_sds.backward()

        # edge ================================================================================================
        # SDS for decomposed instructions with at least two objects.
        for k, v in painter.edge_prompt_index.items():
            vid_tensor, frames_svg, new_points = painter.render_frames_to_tensor(v)
            x = vid_tensor.unsqueeze(0).permute(0, 1, 4, 2, 3).repeat(cfg.batch_size, 1, 1, 1, 1)
            if cfg.augment_frames:
                augmented_pair = data_augs(torch.cat([x.squeeze(0), orig_frames.squeeze(0)]))
                x_aug = augmented_pair[:cfg.num_frames].unsqueeze(0)
                orig_frames_aug = augmented_pair[cfg.num_frames:].unsqueeze(0)
            else:
                x_aug = x
                orig_frames_aug = orig_frames

            loss_sds = sds_loss(x_aug, k) * cfg.loss_edge_weight / len(painter.edge_prompt_index.keys())
            loss += loss_sds.detach()
            loss_sds.backward()

        t_range.set_postfix({'loss': loss.item()})

        # step ================================================================================================
        optimizer.step_()
        logs.update({loss: loss.detach().cpu().item()})
        optimizer.update_lr()
        logs.update({"lr_points": optimizer.get_lr("points"), "step": step})

        if step % cfg.save_vid_iter == 0:
            utils.save_mp4_from_tensor(vid_tensor_g, f"{cfg.output_folder}/mp4_logs/{step}.mp4")
            utils.save_vid_svg(frames_svg_g, f"{cfg.output_folder}/svg_logs", step, painter.canvas_width, painter.canvas_height)

            if step > 0:
                painter.log_state(f"{cfg.output_folder}/models/")
        
    # Saves a high quality .gif from the final SVG frames
    shutil.rmtree(f"{cfg.output_folder}/models")
    utils.save_hq_video(cfg.output_folder, iter_=cfg.num_iter)
