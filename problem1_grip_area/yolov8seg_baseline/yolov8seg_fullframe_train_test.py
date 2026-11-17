# !pip install ultralytics pycocotools opencv-python numpy pyyaml
#
# YOLOv8-seg FULL-FRAME comparison (the comparison requested by the reviewer)
#   1) converts the maskrcnn_600p_700x700 dataset (COCO) to the YOLO format
#   2) trains YOLOv8-seg with the Ultralytics defaults and saves the best weights
#   3) prints the standard YOLO metrics (mAP) on the 700x700 test split
#   4) measures mIoU per part on the full-resolution photos
#      (part finding, cropping and IoU: same as time_and_iou_tester_with_cropper_mic.py)
#   5) measures the time per frame

import os
import json
import shutil
import time
import cv2
import numpy as np
import torch
import yaml
from pycocotools.coco import COCO
from ultralytics import YOLO

PROJECT_ROOT = os.environ.get("GRIP_PROJECT_ROOT", os.path.abspath("project_root"))   # see README

# ==============================================================================
# 1. PATHS AND SETTINGS
# ==============================================================================
DATASETS = os.path.join(PROJECT_ROOT, r"datasets\derivative_datasets")
TRAIN_DATASET = os.path.join(DATASETS, "maskrcnn_600p_700x700")        # training data: 404 / 117 / 60 photos (700x700)
EVAL_DATASET = os.path.join(DATASETS, "blackbackground_withlabels")    # mIoU data: full resolution (4928x3264)
EVAL_SPLIT = "test"   # 16 photos (seen by neither model). "valid" = 34 photos (the split the HM R-CNN number was measured on)

WORK_DIR = os.path.dirname(os.path.abspath(__file__))
YOLO_DATASET = os.path.join(WORK_DIR, "yolo_dataset")
RUNS_DIR = os.path.join(WORK_DIR, "runs")
MODEL_SAVE_PATH = os.path.join(WORK_DIR, "sametkalip_yolov8s_seg_fullframe.pt")

TRAIN = True          # False: skip training, load the saved model (MODEL_SAVE_PATH) and only test

CLASS_NAMES = ["alt", "griparea", "ust"]   # YOLO class order (1, 2, 3 in COCO)
GRIP_CLASS = 1

# Training settings: Ultralytics defaults. The only difference is imgsz: 704 instead of 640 because the data is 700x700
# (must be a multiple of 32; with 640 the images would be downscaled)
MODEL_NAME = "yolov8s-seg.pt"   # COCO pre-trained weights, downloaded automatically on the first run
IMGSZ = 704
EPOCHS = 100
BATCH = 8                       # 16 fills the memory of a 6 GB card; use 4 if it is still full

CONF = 0.25                     # Ultralytics default confidence threshold

# Part finding and cropping: same values as time_and_iou_tester_with_cropper_mic.py
PARAMS = {
    'target_w': 160, 'target_h': 1600,
    'brightness_beta': 220,
    'min_area': 5000, 'max_area': 30000000,
    'edge_tolerance': 10, 'rect_ratio_limit': 1.325,
    'safety_crop_percent': 0.0
}

# ==============================================================================
# 2. COCO -> YOLO CONVERTER
# ==============================================================================
def convert_coco_to_yolo():
    yaml_path = os.path.join(YOLO_DATASET, "data.yaml")
    if os.path.exists(yaml_path):
        print(f"YOLO dataset already exists: {YOLO_DATASET}")
        return yaml_path

    print("Converting COCO -> YOLO...")
    for split in ("train", "valid", "test"):
        with open(os.path.join(TRAIN_DATASET, split, "_annotations.coco.json"), encoding="utf-8") as f:
            coco = json.load(f)
        cat_to_cls = {c['id']: CLASS_NAMES.index(c['name']) for c in coco['categories'] if c['name'] in CLASS_NAMES}
        anns_by_img = {}
        for ann in coco['annotations']:
            anns_by_img.setdefault(ann['image_id'], []).append(ann)

        img_dir = os.path.join(YOLO_DATASET, "images", split)
        lbl_dir = os.path.join(YOLO_DATASET, "labels", split)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

        for img in coco['images']:
            shutil.copy2(os.path.join(TRAIN_DATASET, split, img['file_name']), os.path.join(img_dir, img['file_name']))
            w, h = img['width'], img['height']
            lines = []
            for ann in anns_by_img.get(img['id'], []):
                if ann['category_id'] not in cat_to_cls: continue
                for poly in ann['segmentation']:
                    if len(poly) < 6: continue
                    pts = np.array(poly, dtype=np.float64).reshape(-1, 2) / [w, h]   # normalize to 0-1
                    pts = np.clip(pts, 0, 1)
                    lines.append(f"{cat_to_cls[ann['category_id']]} " + " ".join(f"{v:.6f}" for v in pts.reshape(-1)))
            with open(os.path.join(lbl_dir, os.path.splitext(img['file_name'])[0] + ".txt"), "w") as f:
                f.write("\n".join(lines))
        print(f"   {split:<5}: {len(coco['images'])} images")

    with open(yaml_path, "w") as f:
        yaml.safe_dump({'path': YOLO_DATASET, 'train': 'images/train', 'val': 'images/valid', 'test': 'images/test',
                        'names': {i: n for i, n in enumerate(CLASS_NAMES)}}, f, sort_keys=False)
    return yaml_path

# ==============================================================================
# 3. PART FINDING AND CROPPING (same as the HM R-CNN measurement script)
# ==============================================================================
def shed_filter(contours, width, height):
    kept = []
    for c in contours:
        area = cv2.contourArea(c)
        if not (PARAMS['min_area'] <= area <= PARAMS['max_area']): continue
        pts = c.reshape(-1, 2)
        touches_edge = (pts[:,0].min() <= PARAMS['edge_tolerance'] or
                        pts[:,0].max() >= width - 1 - PARAMS['edge_tolerance'] or
                        pts[:,1].min() <= PARAMS['edge_tolerance'] or
                        pts[:,1].max() >= height - 1 - PARAMS['edge_tolerance'])
        if touches_edge: continue
        rect = cv2.minAreaRect(c)
        (_, _), (w, h), _ = rect
        if area == 0 or (w * h / area) >= PARAMS['rect_ratio_limit']: continue
        kept.append(c)
    return kept

def crop_and_rotate_perfect(img, full_gt_mask, contour):
    rect = cv2.minAreaRect(contour)
    center, size, angle = rect

    if size[0] > size[1]:
        angle -= 90
        size = (size[1], size[0])

    w, h = int(size[0]), int(size[1])

    box = cv2.boxPoints(rect)
    x_min = max(0, int(np.min(box[:,0]) - 20))
    y_min = max(0, int(np.min(box[:,1]) - 20))
    x_max = min(img.shape[1], int(np.max(box[:,0]) + 20))
    y_max = min(img.shape[0], int(np.max(box[:,1]) + 20))

    img_roi = img[y_min:y_max, x_min:x_max]
    gt_roi = full_gt_mask[y_min:y_max, x_min:x_max]
    center_roi = (center[0] - x_min, center[1] - y_min)

    M = cv2.getRotationMatrix2D(center_roi, angle, 1.0)
    M[0, 2] += (w / 2) - center_roi[0]
    M[1, 2] += (h / 2) - center_roi[1]

    img_crop = cv2.warpAffine(img_roi, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
    gt_crop = cv2.warpAffine(gt_roi, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    m = int(w * PARAMS['safety_crop_percent'])
    if m > 0:
        img_crop = img_crop[:, m:-m]
        gt_crop = gt_crop[:, m:-m]

    return img_crop, gt_crop

def prepare_inputs_perfect(img_crop, gt_crop):
    h, w = img_crop.shape[:2]
    scale = min(PARAMS['target_w'] / w, PARAMS['target_h'] / h)
    new_w, new_h = int(w * scale), int(h * scale)

    img_resized = cv2.resize(img_crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
    gt_resized = cv2.resize(gt_crop, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    pad_w, pad_h = PARAMS['target_w'] - new_w, PARAMS['target_h'] - new_h
    top, left = pad_h // 2, pad_w // 2

    img_padded = cv2.copyMakeBorder(img_resized, top, pad_h-top, left, pad_w-left, cv2.BORDER_CONSTANT, value=(0,0,0))
    gt_padded = cv2.copyMakeBorder(gt_resized, top, pad_h-top, left, pad_w-left, cv2.BORDER_CONSTANT, value=0)

    return img_padded, gt_padded

def calculate_iou(mask_pred, mask_gt):
    intersection = np.logical_and(mask_pred, mask_gt).sum()
    union = np.logical_or(mask_pred, mask_gt).sum()
    return intersection / union if union > 0 else 0

# ==============================================================================
# 4. YOLO PREDICTION (full frame)
# ==============================================================================
def predict_grip_mask(model, img_orig):
    # The whole frame is given to the model at once; only the griparea masks are taken and merged.
    # retina_masks=True: the mask is upsampled to the original resolution (important for IoU on the narrow strip)
    h_orig, w_orig = img_orig.shape[:2]
    result = model.predict(img_orig, imgsz=IMGSZ, conf=CONF, classes=[GRIP_CLASS], retina_masks=True, verbose=False)[0]
    if result.masks is None:
        return np.zeros((h_orig, w_orig), dtype=np.uint8)
    full_pred_mask = (result.masks.data.sum(dim=0) > 0).cpu().numpy().astype(np.uint8)
    if full_pred_mask.shape != (h_orig, w_orig):
        full_pred_mask = cv2.resize(full_pred_mask, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
    return full_pred_mask

# ==============================================================================
# 5. mIoU PER PART (on full-resolution photos)
# ==============================================================================
def evaluate_miou(predict_fn):
    ann_file = os.path.join(EVAL_DATASET, EVAL_SPLIT, "_annotations.coco.json")
    img_dir = os.path.join(EVAL_DATASET, EVAL_SPLIT)
    coco_gt = COCO(ann_file)
    grip_area_id = [c['id'] for c in coco_gt.loadCats(coco_gt.getCatIds()) if c['name'] == 'griparea'][0]

    global_ious, all_crop_ious = [], []
    print(f"\nData: {EVAL_DATASET} | Split: {EVAL_SPLIT}")
    print(f"{'Image':<25} | {'Parts':<4} | {'IoU (Crop)':<10}")
    print("-" * 50)

    for img_id in coco_gt.getImgIds():
        img_name = coco_gt.loadImgs(img_id)[0]['file_name']
        img_orig = cv2.imread(os.path.join(img_dir, img_name))
        if img_orig is None: continue
        h_orig, w_orig = img_orig.shape[:2]

        # Parts are found as in the HM R-CNN measurement (to compute IoU only on the same parts)
        img_bright = cv2.convertScaleAbs(img_orig, alpha=1.0, beta=PARAMS['brightness_beta'])
        gray = cv2.cvtColor(img_bright, cv2.COLOR_BGR2GRAY)
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_cnts = shed_filter(contours, w_orig, h_orig)
        if not valid_cnts:
            continue

        anns = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=img_id, catIds=[grip_area_id]))
        full_gt_mask = np.zeros((h_orig, w_orig), dtype=np.uint8)
        for ann in anns: full_gt_mask = np.maximum(full_gt_mask, coco_gt.annToMask(ann))

        full_pred_mask = predict_fn(img_orig)

        crop_ious = []
        for cnt in valid_cnts:
            # GT and prediction go through the same crop + 160x1600 canvas
            img_crop, gt_crop = crop_and_rotate_perfect(img_orig, full_gt_mask, cnt)
            _, pred_crop = crop_and_rotate_perfect(img_orig, full_pred_mask, cnt)
            _, gt_input = prepare_inputs_perfect(img_crop, gt_crop)
            _, pred_input = prepare_inputs_perfect(img_crop, pred_crop)
            crop_ious.append(calculate_iou(pred_input, gt_input))

        avg_img_iou = np.mean(crop_ious)
        global_ious.append(avg_img_iou)
        all_crop_ious.extend(crop_ious)
        print(f"{img_name[:25]:<25} | {len(valid_cnts):<4} | %{avg_img_iou*100:.1f}")

    print("=" * 50)
    print(f"Photos: {len(global_ious)} | Parts: {len(all_crop_ious)}")
    print(f"OVERALL TEST IoU (mean over photos, same as the HM R-CNN measurement): %{np.mean(global_ious)*100:.2f}")
    print(f"   Mean over parts: %{np.mean(all_crop_ious)*100:.2f}")
    return np.mean(global_ious)

# ==============================================================================
# 6. TIME PER FRAME
# ==============================================================================
def measure_latency(model):
    img_dir = os.path.join(EVAL_DATASET, EVAL_SPLIT)
    files = [f for f in sorted(os.listdir(img_dir)) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    model.predict(cv2.imread(os.path.join(img_dir, files[0])), imgsz=IMGSZ, conf=CONF, verbose=False)   # warm-up run
    times = []
    for f in files:
        img = cv2.imread(os.path.join(img_dir, f))   # file reading is not included in the time
        if torch.cuda.is_available(): torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.predict(img, imgsz=IMGSZ, conf=CONF, verbose=False)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    print(f"\nTime per frame (pre-processing + model + post-processing, {len(times)} frames): {np.mean(times):.1f} ms")
    return np.mean(times)

# ==============================================================================
# 7. MAIN FLOW
# ==============================================================================
if __name__ == "__main__":
    yaml_path = convert_coco_to_yolo()

    if TRAIN:
        model = YOLO(MODEL_NAME)
        model.train(data=yaml_path, imgsz=IMGSZ, epochs=EPOCHS, batch=BATCH, project=RUNS_DIR, name="train", exist_ok=True)
        shutil.copy2(os.path.join(RUNS_DIR, "train", "weights", "best.pt"), MODEL_SAVE_PATH)
        print(f"\n[OK] Training finished, model saved: {MODEL_SAVE_PATH}")

    model = YOLO(MODEL_SAVE_PATH)

    # Standard YOLO metrics: 700x700 test split (60 photos)
    metrics = model.val(data=yaml_path, split="test", imgsz=IMGSZ, project=RUNS_DIR, name="test", exist_ok=True, verbose=False)
    print(f"\n700x700 test split | mask mAP50: {metrics.seg.map50:.3f} | mask mAP50-95: {metrics.seg.map:.3f}")

    # Metric in the paper: mIoU per part on the full-resolution photos
    evaluate_miou(lambda img: predict_grip_mask(model, img))
    measure_latency(model)
