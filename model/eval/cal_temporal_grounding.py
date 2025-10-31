import numpy as np
import json
import re
from typing import List
from tqdm import tqdm
import argparse
import os

def compute_iou(pred, gt):
    inter_start = max(pred[0], gt[0])
    inter_end = min(pred[1], gt[1])
    inter = max(0, inter_end - inter_start)
    union = max(pred[1], gt[1]) - min(pred[0], gt[0])
    return inter / union if union > 0 else 0

def compute_ap(pred_segments, gt_segments, iou_thresh):
    tp = 0
    for pred, gt in zip(pred_segments, gt_segments):
        iou = compute_iou(pred, gt)
        if iou >= iou_thresh:
            tp += 1
    ap = tp / len(pred_segments)
    return ap

def compute_metrics(pred_segments, gt_segments):
    ap_30 = compute_ap(pred_segments, gt_segments, iou_thresh=0.3)
    ap_50 = compute_ap(pred_segments, gt_segments, iou_thresh=0.5)
    ap_70 = compute_ap(pred_segments, gt_segments, iou_thresh=0.7)
    m_ap = (ap_30 + ap_50 + ap_70) / 3

    ious = [compute_iou(pred, gt) for pred, gt in zip(pred_segments, gt_segments)]
    m_iou = np.mean(ious)

    return {
        "AP@30": ap_30,
        "AP@50": ap_50,
        "AP@70": ap_70,
        "mAP": m_ap,
        "mIoU": m_iou
    }


def parse_temporal_grounding(text: str) -> List[float]:
    """
    Extract two numbers from the text for temporal grounding
    The format is usually "From X to Y" or similar
    If no numbers are found, return [0, 0]
    """
    if not text:
        return [0, 0]
    
    # use regex to find all numbers
    numbers = re.findall(r'-?\d+\.?\d*', text)
    
    # if at least two numbers are found, return the first two
    if len(numbers) >= 2:
        return [float(numbers[0]), float(numbers[1])]
    # if only one number is found, return the number and 0
    elif len(numbers) == 1:
        return [0, float(numbers[0])]
    # if no numbers are found, return [0, 0]
    else:
        return [0, 0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Temporal Grounding Evaluation.')
    parser.add_argument('-p', '--pred', default=None)
    parser.add_argument('-m', '--model', default='safe')
    args = parser.parse_args()

    f_pred = open(os.path.expanduser(args.pred))

    answers_file = args.pred.replace('.jsonl', '_temporal_grounding_score.jsonl')

    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    pred_list = []
    gt_list = []
    idx = 0
    for ans_js in tqdm(f_pred, desc='Evaluating'):
        # if idx == 100:
        #     break
        ans = json.loads(ans_js)
        gt = ans['gt']
        pred = ans['text']

        pred_t = parse_temporal_grounding(pred)
        gt_t = parse_temporal_grounding(gt)

        if 0 <= pred_t[0] <= 1 and 0 <= pred_t[1] <= 1:
            total_frames = ans.get('total_frames', None)
            if total_frames is None:
                total_frames = len(os.listdir(os.path.join('/data', ans['image_path'] if isinstance(ans['image_path'], str) else ans['image_path'][0])))

            pred_t = [pred_t[0] * total_frames, pred_t[1] * total_frames]

        pred_list.append(pred_t)
        gt_list.append(gt_t)

        
        ans_file.write(json.dumps({"question_id": idx,
                                "miou_score": round(compute_iou(pred_t, gt_t)*100, 2),
                                "image_path": ans['image_path'],
                                "gt": ans['gt'],
                                "pred": pred_t,
                                "text": pred,
                                "metadata": {}}) + "\n")
        ans_file.flush()
        idx += 1
    ans_file.close()

    metrics = compute_metrics(pred_list, gt_list)
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
