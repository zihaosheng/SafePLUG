from segment_anything import SamPredictor, sam_model_registry
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pycocotools.mask as mask_utils
import argparse
import os
import json
from tqdm import tqdm
import re

def load_sam(ckpt_path):
    sam = sam_model_registry['vit_h'](checkpoint=ckpt_path)
    sam.to(device='cuda')
    predictor = SamPredictor(sam)
    return predictor

def get_mask(predictor, image, bbox):
    bbox = np.array(bbox)
    if all(0 <= b <= 1 for b in bbox):
        bbox = bbox * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
    bbox = bbox.reshape(1, 4)
    bbox = bbox.astype(np.float32)
    predictor.set_image(image)
    masks, _, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=bbox,
        multimask_output=False,
    )
    return masks[0], bbox

def compute_iou(pred, gt):
    pred = np.asarray(pred).astype(bool)
    gt = np.asarray(gt).astype(bool)

    intersection = np.logical_and(pred, gt)
    union = np.logical_or(pred, gt)

    intersection_pixels = np.sum(intersection)
    union_pixels = np.sum(union)   

    if union_pixels == 0:
        iou = 0
    else:
        iou = intersection_pixels / union_pixels
    return iou


def compute_ap(pred_segments, gt_segments, iou_thresh):
    tp = 0
    for pred, gt in zip(pred_segments, gt_segments):
        iou = compute_iou(pred, gt)
        if iou >= iou_thresh:
            tp += 1
    ap = tp / len(pred_segments)
    return ap

def compute_metrics(pred_list, gt_list):
    ap_30 = compute_ap(pred_list, gt_list, iou_thresh=0.3)
    ap_50 = compute_ap(pred_list, gt_list, iou_thresh=0.5)
    ap_70 = compute_ap(pred_list, gt_list, iou_thresh=0.7)
    m_ap = (ap_30 + ap_50 + ap_70) / 3

    ious = [compute_iou(pred, gt) for pred, gt in zip(pred_list, gt_list)]
    dices = [2 * iou / (1 + iou) for iou in ious]
    m_iou = np.mean(ious)
    m_dice = np.mean(dices)

    return {
        'AP@30': ap_30,
        'AP@50': ap_50,
        'AP@70': ap_70,
        'mAP': m_ap,
        'mIoU': m_iou,
        'mDice': m_dice
    }


def decode_seg_mask(mask):
    if isinstance(mask['counts'], str):
        mask['counts'] = mask['counts'].encode('utf-8')
    mask_np = mask_utils.decode(mask)
    return mask_np

def parse_bbox(text):
    if 'bbox_2d' in text: text = text.replace('bbox_2d', 'bbox')
    numbers = re.findall(r'-?\d+\.?\d*', text)
    numbers = [float(n) for n in numbers]
    if len(numbers) == 4:
        return numbers
    elif len(numbers) > 4:
        return numbers[:4]
    else:
        return [0] * (4 - len(numbers)) + numbers


def vis_overlay_masks(original_image_path, prediction_mask, ground_truth_mask, save_path, pred_bbox=None):
    # Read the original image
    original_image = cv2.imread(original_image_path)

    # Convert original image and masks to RGB format
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    prediction_mask = prediction_mask * 255
    ground_truth_mask = ground_truth_mask * 255

    # Create semi-transparent light blue color (for overlay)
    # overlay_color = np.array([255, 0, 0], dtype=np.uint8)
    overlay_color = np.array([118, 158, 224], dtype=np.uint8)
    prediction_overlay = np.zeros_like(original_image)
    ground_truth_overlay = np.zeros_like(original_image)

    # Overlay semi-transparent color on masks
    prediction_overlay[prediction_mask > 0] = overlay_color
    ground_truth_overlay[ground_truth_mask > 0] = overlay_color

    # Merge original image and masks
    prediction_overlay_image = cv2.addWeighted(original_image, 0.5, prediction_overlay, 0.9, 0)
    ground_truth_overlay_image = cv2.addWeighted(original_image, 0.5, ground_truth_overlay, 0.9, 0)

    if pred_bbox is not None:
        x0, y0, x1, y1 = pred_bbox[0]
        prediction_overlay_image = cv2.rectangle(prediction_overlay_image, (int(x0), int(y0)), (int(x1), int(y1)), (255, 0, 0), 5)

    # Save the concatenated image
    cv2.imwrite(save_path.replace('.png', '_pred.png'), cv2.cvtColor(prediction_overlay_image, cv2.COLOR_RGB2BGR))
    cv2.imwrite(save_path.replace('.png', '_gt.png'), cv2.cvtColor(ground_truth_overlay_image, cv2.COLOR_RGB2BGR))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Baseline Segmentation Metrics Evaluation.')
    parser.add_argument('-p', '--pred', default='/home/sky-lab/SHENG_code/SafetyGPT/baseline_eval/Qwen2.5-VL-7B-Instruct/DoTA_grounding_test.jsonl')
    parser.add_argument('-m', '--model', default='qwen')
    parser.add_argument('-c', '--ckpt', default='/data/huggingface-models/sam_vit_h_4b8939.pth')
    parser.add_argument('-v', '--vis_mask', action='store_true')
    args = parser.parse_args()

    predictor = load_sam(args.ckpt)
    f_pred = open(os.path.expanduser(args.pred))

    pred_list = []
    gt_list = []
    idx = 0
    for ans_js in tqdm(f_pred, desc='Evaluating', total=500):
        # idx += 1
        # if idx == 10:
        #     break
        ans = json.loads(ans_js)
        gt_mask = ans['gt_mask']
        gt_mask = decode_seg_mask(gt_mask)

        ans['image_path'] = ans['image_path'].replace('/home/jovyan/shared/zihaosheng/hugging-face-models', '/data')

        pred = ans['text']
        if isinstance(pred, str):
            pred_bbox = parse_bbox(pred)
            image = cv2.imread(ans['image_path'])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pred_mask, pred_bbox = get_mask(predictor, image, pred_bbox)
        else:
            pred_bbox = None
            pred_mask = decode_seg_mask(pred)

        if args.vis_mask:
            iou = compute_iou(pred_mask, gt_mask)
            model_name = args.pred.split('/')[-2]
            test_file_name = args.pred.split('/')[-1].split('.')[0]
            img_name = ans['image_path'].split('/')[-3] + '_' + os.path.basename(ans['image_path'])[:-4]
            save_path = os.path.join('./baseline_eval', model_name, test_file_name + '_seperate', 'iou_'+str(round(iou,4))+'_'+img_name+'.png')
            print(save_path)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            vis_overlay_masks(ans['image_path'], pred_mask, gt_mask, save_path, pred_bbox)
        
        pred_list.append(pred_mask)
        gt_list.append(gt_mask)

    metrics = compute_metrics(pred_list, gt_list)
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
