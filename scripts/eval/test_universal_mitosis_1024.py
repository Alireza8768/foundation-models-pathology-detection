# Trainning/test_universal_mitosis_1024.py

"""
Threshold analysis for mitosis detection.

Runs inference once on a COCO-style dataset and caches predictions.
Then sweeps score thresholds and computes TP/FP/FN using greedy IoU matching,
reporting Precision/Recall/F1 and FP per image. Saves CSV and plots.

To reuse for another model:
- set config_file and checkpoint_file
- set img_root and ann_file
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from tqdm import tqdm
import os
import torch
from mmengine.config import Config
from mmengine.dataset import Compose
from mmdet.apis import init_detector
from mmengine.dataset import default_collate
import sys

project_root = Path(__file__).resolve().parents[2]

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
    
# =====================
# CONFIG
# =====================
config_file = str(project_root / 'configs/faster_rcnn_uni_midogpp.py')
checkpoint_file = str(project_root / 'outputs/work_dirs/faster_rcnn_uni_1024_40epochs/best_coco_bbox_mAP_epoch_30.pth')

img_root = project_root / 'data/Datensatz/'
ann_file = str(project_root / 'data/coco_annotations/patches_1024/midogpp_test.json')

IOU_THR = 0.5
SCORE_THRESHOLDS = np.round(np.arange(0.0, 0.96, 0.05), 2).tolist()

out_dir = Path(project_root / 'outputs/test/uni_threshold_analysis_fpn_downscale')
out_dir.mkdir(parents=True, exist_ok=True)


def iou_matrix(pred_xyxy: np.ndarray, gt_xyxy: np.ndarray) -> np.ndarray:
    """
    pred_xyxy: (P,4), gt_xyxy: (G,4)
    returns IoU matrix shape (P,G)
    """
    if pred_xyxy.size == 0 or gt_xyxy.size == 0:
        return np.zeros((pred_xyxy.shape[0], gt_xyxy.shape[0]), dtype=np.float32)

    px1, py1, px2, py2 = pred_xyxy[:, 0:1], pred_xyxy[:, 1:2], pred_xyxy[:, 2:3], pred_xyxy[:, 3:4]
    gx1, gy1, gx2, gy2 = gt_xyxy[:, 0], gt_xyxy[:, 1], gt_xyxy[:, 2], gt_xyxy[:, 3]

    inter_x1 = np.maximum(px1, gx1)
    inter_y1 = np.maximum(py1, gy1)
    inter_x2 = np.minimum(px2, gx2)
    inter_y2 = np.minimum(py2, gy2)

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h

    p_area = np.maximum(0.0, px2 - px1) * np.maximum(0.0, py2 - py1)
    g_area = np.maximum(0.0, gx2 - gx1) * np.maximum(0.0, gy2 - gy1)

    union = p_area + g_area - inter
    iou = np.where(union > 0, inter / union, 0.0).astype(np.float32)
    return iou

def match_greedy_by_score(gt_xyxy: np.ndarray, pred_xyxy: np.ndarray, pred_scores: np.ndarray, iou_thr: float):
    """
    Greedy matching:
    - sort preds by score desc
    - each pred matches best free GT if IoU>=thr
    """
    G = gt_xyxy.shape[0]
    P = pred_xyxy.shape[0]

    if G == 0 and P == 0:
        return 0, 0, 0
    if G == 0 and P > 0:
        return 0, P, 0
    if G > 0 and P == 0:
        return 0, 0, G

    order = np.argsort(-pred_scores)
    pred_xyxy = pred_xyxy[order]
    pred_scores = pred_scores[order]

    ious = iou_matrix(pred_xyxy, gt_xyxy)
    gt_used = np.zeros((G,), dtype=bool)

    tp = 0
    for i in range(P):
        candidates = np.where(~gt_used)[0]
        if candidates.size == 0:
            break
        best_j = candidates[np.argmax(ious[i, candidates])]
        if ious[i, best_j] >= iou_thr:
            tp += 1
            gt_used[best_j] = True

    fp = P - tp
    fn = G - tp
    return tp, fp, fn

def build_test_pipeline(cfg: Config):
    pipeline = cfg.test_dataloader.dataset.pipeline
    return Compose(pipeline)


@torch.no_grad()
def run_inference(model, pipeline, img_path: Path):
    data = dict(img_path=str(img_path), img_id=0)
    data = pipeline(data)
    
    data_batch = default_collate([data])
    
    results = model.test_step(data_batch)[0]
    preds = results.pred_instances

    if preds is None or len(preds) == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    bboxes = preds.bboxes.detach().cpu().numpy().astype(np.float32)
    scores = preds.scores.detach().cpu().numpy().astype(np.float32)

    return bboxes, scores


def main():
    print("Lade Modell...")
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    model.eval()

    torch.backends.cudnn.benchmark = True

    coco = COCO(ann_file)
    img_ids = coco.getImgIds()

    cfg = Config.fromfile(config_file)

    pipeline = build_test_pipeline(cfg)

    print("Vorhersagen einmalig berechnen (Caching)...")
    cache = []
    valid_images = 0

    for img_id in tqdm(img_ids):
        info = coco.loadImgs(img_id)[0]
        img_path = img_root / info['file_name']
        if not img_path.exists():
            continue

        ann_ids = coco.getAnnIds(imgIds=[img_id])
        anns = coco.loadAnns(ann_ids)

        gt = []
        for a in anns:
            x, y, w, h = a['bbox']
            gt.append([x, y, x + w, y + h])
        gt = np.array(gt, dtype=np.float32)

        pred_bboxes, pred_scores = run_inference(model, pipeline, img_path)

        cache.append((gt, pred_bboxes, pred_scores))
        valid_images += 1

    if valid_images == 0:
        raise RuntimeError("Keine gültigen Bilder gefunden. Pfade prüfen (img_root / file_name).")

    print(f"OK: {valid_images} Bilder gecached.")

    results = []
    print("Threshold sweep...")
    for thr in SCORE_THRESHOLDS:
        TP = FP = FN = 0

        for gt, pb, ps in cache:
            keep = ps >= thr
            pb_k = pb[keep]
            ps_k = ps[keep]
            tp, fp, fn = match_greedy_by_score(gt, pb_k, ps_k, IOU_THR)
            TP += tp
            FP += fp
            FN += fn

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        fp_per_img = FP / valid_images

        results.append((thr, precision, recall, f1, fp_per_img))
        print(f"THR={thr:>4.2f} | P={precision:.3f} R={recall:.3f} F1={f1:.3f} FP/img={fp_per_img:.3f}")

    csv_path = out_dir / "threshold_results.csv"
    header = "thr,precision,recall,f1,fp_per_img\n"
    with open(csv_path, "w") as f:
        f.write(header)
        for r in results:
            f.write(f"{r[0]},{r[1]},{r[2]},{r[3]},{r[4]}\n")

    best = max(results, key=lambda x: x[3])
    print("\n====================")
    print(f"Best by F1: THR={best[0]:.2f} | P={best[1]:.3f} R={best[2]:.3f} F1={best[3]:.3f} FP/img={best[4]:.3f}")
    print("CSV gespeichert:", csv_path)

    thrs = [r[0] for r in results]
    prec = [r[1] for r in results]
    rec = [r[2] for r in results]
    f1s = [r[3] for r in results]
    fps = [r[4] for r in results]

    plt.figure()
    plt.plot(rec, prec, marker='o')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision–Recall (IoU={IOU_THR})')
    plt.grid(True)
    plt.savefig(out_dir / 'precision_recall_curve.png', dpi=200)
    plt.close()

    plt.figure()
    plt.plot(thrs, fps, marker='o')
    plt.xlabel('Score Threshold')
    plt.ylabel('False Positives per Image')
    plt.title('FP/Image vs Score Threshold')
    plt.grid(True)
    plt.savefig(out_dir / 'fp_per_image_vs_threshold.png', dpi=200)
    plt.close()

    plt.figure()
    plt.plot(thrs, f1s, marker='o')
    plt.xlabel('Score Threshold')
    plt.ylabel('F1')
    plt.title('F1 vs Score Threshold')
    plt.grid(True)
    plt.savefig(out_dir / 'f1_vs_threshold.png', dpi=200)
    plt.close()

    print("Plots gespeichert in:", out_dir)

if __name__ == '__main__':
    main()
