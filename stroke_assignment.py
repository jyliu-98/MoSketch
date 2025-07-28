import argparse
import os
import sys
import math
import numpy as np
import json
import torch
from PIL import Image
from shapely.geometry import Polygon
import pdb

sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))
sys.path.append(os.path.join(os.getcwd(), "segment_anything"))


# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap


# segment anything
from segment_anything import (
    sam_model_registry,
    sam_hq_model_registry,
    SamPredictor
)
import cv2
import numpy as np
import matplotlib.pyplot as plt


def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, bert_base_uncased_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    args.bert_base_uncased_path = bert_base_uncased_path
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))
    ax.text(x0, y0, label)


def calcul_area(calcul_list, H, W):
    obj_num = len(calcul_list)
    sum_iou = 0
    iou_count = 0
    for i in range(obj_num):
        for j in range(i + 1, obj_num):
            iou_count += 1
            box1 = calcul_list[i]
            box2 = calcul_list[j]
            x_left = max(box1[0], box2[0])
            y_top = max(box1[1], box2[1])
            x_right = min(box1[2], box2[2])
            y_bottom = min(box1[3], box2[3])
            intersection = max(0, x_right - x_left) * max(0, y_bottom - y_top)
            box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
            box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
            union = box1_area + box2_area - intersection
            iou = intersection / union if union > 0 else 0
            sum_iou += iou
    iou = sum_iou / iou_count
    polygons = [Polygon([(x1, y1), (x1, y2), (x2, y2), (x2, y1)]) for x1, y1, x2, y2 in calcul_list]
    union_polygon = polygons[0]
    for poly in polygons[1:]:
        union_polygon = union_polygon.union(poly)
    area = union_polygon.area / (H * W)

    return iou, area


def read_svg(svg_path):
    points_xy = []
    with open(svg_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        if line[:8] == "<path d=":
            point_xy = []
            path_info = line.split('\"')[1]
            path_split = path_info.split(' ')
            point_xy.append(float(path_split[1]))
            point_xy.append(float(path_split[2]))
            point_xy.append(float(path_split[4]))
            point_xy.append(float(path_split[5]))
            point_xy.append(float(path_split[6]))
            point_xy.append(float(path_split[7]))
            point_xy.append(float(path_split[8]))
            point_xy.append(float(path_split[9]))
            point_xy_np = np.expand_dims(np.array(point_xy), axis=0)
            points_xy.append(point_xy_np)
    return np.concatenate(points_xy, axis=0)


def stroke_seg(points, sam_result):
    sam_result = sorted(sam_result, key=lambda x: -x[4])
    mask_img = torch.zeros(sam_result[0][1].shape[-2:])
    pixel_dict = {}
    seg_stroke_dict = {}
    for idx, obj in enumerate(sam_result):
        mask_img[obj[1].cpu().numpy()[0] == True] = idx + 1
        pixel_dict[idx + 1] = obj[3]
        if obj[3] not in seg_stroke_dict.keys():
            seg_stroke_dict[obj[3]] = []
    sam_result = sorted(sam_result, key=lambda x: x[4])
    stroke_seg_list = []
    for i in range(points.shape[0]):
        middle_x = int((points[i][0] + points[i][2] + points[i][4] + points[i][6]) / 4)
        middle_y = int((points[i][1] + points[i][3] + points[i][5] + points[i][7]) / 4)
        if mask_img[middle_y, middle_x] > 0:
            stroke_semantic = pixel_dict[int(mask_img[middle_y, middle_x])]
        else:
            shot_mark = 0
            for obj in sam_result:
                if (middle_x >= obj[2][0] and middle_x <= obj[2][2] and
                        middle_y >= obj[2][1] and middle_y <= obj[2][3]):
                    stroke_semantic = obj[3]
                    shot_mark = 1
                    break
            if shot_mark == 0:
                min_dist = 100000
                stroke_semantic = None
                for obj in sam_result:
                    obj_x = (obj[2][0] + obj[2][2]) / 2
                    obj_y = (obj[2][1] + obj[2][3]) / 2
                    dist = math.sqrt(math.pow((obj_x - middle_x), 2) + math.pow((obj_y - middle_y), 2))
                    if dist < min_dist:
                        min_dist = dist
                        stroke_semantic = obj[3]
        seg_stroke_dict[stroke_semantic].append(i)
        stroke_seg_list.append(stroke_semantic)

    return seg_stroke_dict, stroke_seg_list


def rewrite_svg(svg_path, new_svg_path, stroke_seg_list, obj_list):
    color_candi = ["rgb(255, 0, 0)",
                   "rgb(0, 255, 0)",
                   "rgb(0, 0, 255)",
                   "rgb(255, 255, 0)",
                   "rgb(255, 0, 255)",
                   "rgb(0, 255, 255)"
                   ]
    colour_name = ["red",
                   "green",
                   "blue",
                   "yellow",
                   "purple",
                   "cyan"
                   ]
    obj_color_dict = {}
    for idx, name in enumerate(obj_list):
        obj_color_dict[name] = color_candi[idx]
        print(name + ': ' + colour_name[idx])
    with open(svg_path, 'r') as f:
        lines = f.readlines()
    idx = 0
    with open(new_svg_path, 'w') as f:
        for line in lines:
            line_strip = line.strip()
            if line_strip[:8] == "<path d=":
                name = stroke_seg_list[idx]
                new_line = line.replace('rgb(0, 0, 0)', obj_color_dict[name])
                f.write(new_line)
                idx += 1
            else:
                f.write(line)


def save_mask_data(output_dir, mask_list, box_list, label_list):
    value = 0  # 0 for background
    mask_img = torch.zeros(mask_list.shape[-2:])
    for idx, mask in enumerate(mask_list):
        mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1
    plt.figure(figsize=(10, 10))
    plt.imshow(mask_img.numpy())
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'mask.jpg'), bbox_inches="tight", dpi=300, pad_inches=0.0)

    json_data = [{
        'value': value,
        'label': 'background'
    }]
    for label, box in zip(label_list, box_list):
        value += 1
        name, logit = label.split('(')
        logit = logit[:-1] # the last is ')'
        json_data.append({
            'value': value,
            'label': name,
            'logit': float(logit),
            'box': box.numpy().tolist(),
        })
    with open(os.path.join(output_dir, 'mask.json'), 'w') as f:
        json.dump(json_data, f)


def main():

    parser = argparse.ArgumentParser("Grounded-Segment-Anything Demo", add_help=True)
    parser.add_argument("--config", type=str, required=True, help="path to config file")
    parser.add_argument(
        "--grounded_checkpoint", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument(
        "--sam_version", type=str, default="vit_h", required=False, help="SAM ViT version: vit_b / vit_l / vit_h"
    )
    parser.add_argument(
        "--sam_checkpoint", type=str, required=False, help="path to sam checkpoint file"
    )
    parser.add_argument(
        "--sam_hq_checkpoint", type=str, default=None, help="path to sam-hq checkpoint file"
    )
    parser.add_argument(
        "--use_sam_hq", action="store_true", help="using sam-hq for prediction"
    )
    parser.add_argument("--sketch_img", type=str, required=True, help="path to image file")
    parser.add_argument("--text_prompt", type=str, required=True, help="text prompt")
    parser.add_argument(
        "--sketch_dir", "-o", type=str, default="outputs", required=True, help="sketch directory"
    )

    parser.add_argument("--box_threshold", type=float, default=0.3, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")

    parser.add_argument("--device", type=str, default="cpu", help="running on cpu only!, default=False")
    parser.add_argument("--bert_base_uncased_path", type=str, required=False, help="bert_base_uncased model path, default=False")
    parser.add_argument("--iou_w", type=float, required=True)
    args = parser.parse_args()

    # cfg
    config_file = args.config  # change the path of the model config file
    grounded_checkpoint = args.grounded_checkpoint  # change the path of the model
    sketch_dir = args.sketch_dir
    sam_version = args.sam_version
    sam_checkpoint = args.sam_checkpoint
    sam_hq_checkpoint = args.sam_hq_checkpoint
    use_sam_hq = args.use_sam_hq
    image_path = os.path.join(sketch_dir, args.sketch_img)
    text_prompt = args.text_prompt

    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    device = args.device
    bert_base_uncased_path = args.bert_base_uncased_path
    iou_w = args.iou_w
    svg_path = image_path.split('.')[0] + '.svg'

    # make dir
    os.makedirs(sketch_dir, exist_ok=True)
    # load image
    image_pil, image = load_image(image_path)
    # load model
    model = load_model(config_file, grounded_checkpoint, bert_base_uncased_path, device=device)
    # run grounding dino model
    boxes_filt, pred_phrases = get_grounding_output(
        model, image, text_prompt, box_threshold, text_threshold, device=device
    )

    # initialize SAM
    if use_sam_hq:
        predictor = SamPredictor(sam_hq_model_registry[sam_version](checkpoint=sam_hq_checkpoint).to(device))
    else:
        predictor = SamPredictor(sam_model_registry[sam_version](checkpoint=sam_checkpoint).to(device))
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)

    size = image_pil.size
    H, W = size[1], size[0]
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]

    boxes_filt = boxes_filt.cpu()
    transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(device)

    masks, _, _ = predictor.predict_torch(
        point_coords = None,
        point_labels = None,
        boxes = transformed_boxes.to(device),
        multimask_output = False,
    )

    # get valid grounded for svg sketch
    object_list = [obj.strip() for obj in text_prompt.lower().split(',')]
    obj_cadi_dict = {}
    for obj in object_list:
        obj_cadi_dict[obj] = []
    for i in range(boxes_filt.size(0)):
        name = pred_phrases[i].split('(')[0]
        if name in object_list:
            score = float(pred_phrases[i].split('(')[1].split(')')[0])
            mask = masks[i]
            bbox = boxes_filt[i]
            area = max(bbox[2] - bbox[0], 0) * max(bbox[3] - bbox[1], 0)
            obj_cadi_dict[name].append((score, mask, bbox, name, area))
    for k, v in obj_cadi_dict.items():
        obj_cadi_dict[k] = sorted(v, key=lambda x:-x[0])

    max_divide = -1 * iou_w
    best_sam = []
    if len(object_list) == 2:
        for obj1 in obj_cadi_dict[object_list[0]]:
            for obj2 in obj_cadi_dict[object_list[1]]:
                per_sam = [obj1, obj2]
                calcul_list = [obj1[2], obj2[2]]
                iou, area = calcul_area(calcul_list, H, W)
                if max_divide < area - iou * iou_w:
                    best_sam = per_sam
                    max_divide = area - iou * iou_w
    if len(object_list) == 3:
        for obj1 in obj_cadi_dict[object_list[0]]:
            for obj2 in obj_cadi_dict[object_list[1]]:
                for obj3 in obj_cadi_dict[object_list[2]]:
                    per_sam = [obj1, obj2, obj3]
                    calcul_list = [obj1[2], obj2[2], obj3[2]]
                    iou, area = calcul_area(calcul_list, H, W)
                    if max_divide < area - iou * iou_w:
                        best_sam = per_sam
                        max_divide = area - iou * iou_w
    if len(object_list) == 4:
        for obj1 in obj_cadi_dict[object_list[0]]:
            for obj2 in obj_cadi_dict[object_list[1]]:
                for obj3 in obj_cadi_dict[object_list[2]]:
                    for obj4 in obj_cadi_dict[object_list[3]]:
                        per_sam = [obj1, obj2, obj3, obj4]
                        calcul_list = [obj1[2], obj2[2], obj3[2], obj4[2]]
                        iou, area = calcul_area(calcul_list, H, W)
                        if max_divide < area - iou * iou_w:
                            best_sam = per_sam
                            max_divide = area - iou * iou_w
    if len(object_list) == 5:
        for obj1 in obj_cadi_dict[object_list[0]]:
            for obj2 in obj_cadi_dict[object_list[1]]:
                for obj3 in obj_cadi_dict[object_list[2]]:
                    for obj4 in obj_cadi_dict[object_list[3]]:
                        for obj5 in obj_cadi_dict[object_list[4]]:
                            per_sam = [obj1, obj2, obj3, obj4, obj5]
                            calcul_list = [obj1[2], obj2[2], obj3[2], obj4[2], obj5[2]]
                            iou, area = calcul_area(calcul_list, H, W)
                            if max_divide < area - iou * iou_w:
                                best_sam = per_sam
                                max_divide = area - iou * iou_w
    if len(object_list) == 6:
        for obj1 in obj_cadi_dict[object_list[0]]:
            for obj2 in obj_cadi_dict[object_list[1]]:
                for obj3 in obj_cadi_dict[object_list[2]]:
                    for obj4 in obj_cadi_dict[object_list[3]]:
                        for obj5 in obj_cadi_dict[object_list[4]]:
                            for obj6 in obj_cadi_dict[object_list[5]]:
                                per_sam = [obj1, obj2, obj3, obj4, obj5, obj6]
                                calcul_list = [obj1[2], obj2[2], obj3[2], obj4[2], obj5[2], obj6[2]]
                                iou, area = calcul_area(calcul_list, H, W)
                                if max_divide < area - iou * iou_w:
                                    best_sam = per_sam
                                    max_divide = area - iou * iou_w

    stroke_points = read_svg(svg_path)
    obj_count = {}
    new_best_sam = []
    new_obj_list = []
    for obj in best_sam:
        score = obj[0]
        mask = obj[1]
        bbox = obj[2]
        name = obj[3]
        area = obj[4]
        if name not in obj_count.keys():
            obj_count[name] = 1
        else:
            obj_count[name] += 1
        if obj_count[name] > 1:
            new_name = name + '_' + str(obj_count[name])
        else:
            new_name = name
        new_best_sam.append((score, mask, bbox, new_name, area))
        new_obj_list.append(new_name)

    seg_stroke_dict, stroke_seg_list = stroke_seg(stroke_points, new_best_sam)
    new_svg_path = svg_path.split('.')[0] + '_color.svg'
    rewrite_svg(svg_path, new_svg_path, stroke_seg_list, new_obj_list)

    # save_semantic
    save_semantic_txt = svg_path.split('.')[0] + '_semantic.txt'
    with open(save_semantic_txt, 'w') as save_file:
        for k, v in seg_stroke_dict.items():
            str_v = [str(i) for i in v]
            save_file.write(k + '\t')
            save_file.write(','.join(str_v) + '\n')
    # save_bbox
    save_bbox_txt = svg_path.split('.')[0] + '_bbox.txt'
    with open(save_bbox_txt, 'w') as save_file:
        for k, v in seg_stroke_dict.items():
            per_points_x = np.min(stroke_points[v][:, [0, 2, 4, 6]]) / 4
            per_points_y = np.min(stroke_points[v][:, [1, 3, 5, 7]]) / 4
            per_points_w = np.max(stroke_points[v][:, [0, 2, 4, 6]]) / 4 - per_points_x
            per_points_h = np.max(stroke_points[v][:, [1, 3, 5, 7]]) / 4 - per_points_y
            xywh = []
            xywh.append(str(per_points_x))
            xywh.append(str(per_points_y))
            xywh.append(str(per_points_w))
            xywh.append(str(per_points_h))
            bbox_str = '[' + ','.join(xywh) + ']'
            save_file.write(k + '\t')
            save_file.write(bbox_str + '\n')

    # draw output image
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    for obj in best_sam:
        show_mask(obj[1].cpu().numpy(), plt.gca(), random_color=True)
        show_box(obj[2].numpy(), plt.gca(), obj[3] + str(obj[0]))

    plt.axis('off')
    plt.savefig(
        os.path.join(sketch_dir, "grounded_sam_sketch_output.jpg"),
        bbox_inches="tight", dpi=300, pad_inches=0.0
    )


main()
