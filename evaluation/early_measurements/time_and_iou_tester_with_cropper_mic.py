# !pip install roboflow pycocotools opencv-python torch torchvision numpy

import os
import cv2
import torch
import numpy as np
import time
from roboflow import Roboflow
from pycocotools.coco import COCO
import torchvision
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

PROJECT_ROOT = os.environ.get("GRIP_PROJECT_ROOT", os.path.abspath("project_root"))   # see README

# ==============================================================================
# 1. ROBOFLOW AND PATHS
# ==============================================================================
print("Fetching data from Roboflow...")
rf = Roboflow(api_key=os.environ.get("ROBOFLOW_API_KEY", ""))   # API key from the ROBOFLOW_API_KEY environment variable
project = rf.workspace("object-detection-z3tjc").project("sametkalip_only_blackbackground")
version = project.version(5)
dataset = version.download("coco-segmentation")

SPLIT = "valid" # Use "test" if you get an error
DATASET_LOCATION = dataset.location
ANN_FILE = os.path.join(DATASET_LOCATION, SPLIT, "_annotations.coco.json")
IMG_DIR = os.path.join(DATASET_LOCATION, SPLIT)

MODEL_PATH = os.path.join(PROJECT_ROOT, r"codes_results\gripareafinder\deep_learning_method\maskrcnn\model_3\model\sametkalip_maskrcnn_160x1600_2_master.pth")
SAVE_IMAGES = True
OUTPUT_DIR = os.path.join(DATASET_LOCATION, "benchmark_perfect_crops_with_MIC")

PARAMS = {
    'target_w': 160, 'target_h': 1600,
    'brightness_beta': 220,
    'min_area': 5000, 'max_area': 30000000,
    'edge_tolerance': 10, 'rect_ratio_limit': 1.325,
    'safety_crop_percent': 0.0,
    'mask_threshold': 0.5,
    'grip_area_id': 2
}

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# ==============================================================================
# 2. MODEL ARCHITECTURE
# ==============================================================================
def get_custom_maskrcnn(num_classes):
    anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
    aspect_ratios = ((1.0, 2.0, 5.0, 10.0),) * len(anchor_sizes)
    rpn_anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
    mask_roi_pool = MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3'], output_size=28, sampling_ratio=2)
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(
        weights=None, min_size=160, max_size=1600,
        rpn_anchor_generator=rpn_anchor_generator, mask_roi_pool=mask_roi_pool, box_detections_per_img=25
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)
    return model

# ==============================================================================
# 3. CORE ALGORITHM
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

# MAXIMUM INSCRIBED CIRCLE (MIC), not the centroid
def calculate_mic(mask):
    if cv2.countNonZero(mask) == 0:
        return None, 0
    
    # Apply the distance transform
    # Finds, for every pixel inside, the distance to the nearest black edge
    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    
    # Find the largest value (radius) and its coordinate (center)
    _, max_val, _, max_loc = cv2.minMaxLoc(dist_transform)
    
    return max_loc, max_val # ( (cX, cY), radius )

# ==============================================================================
# 4. MAIN BENCHMARK LOOP
# ==============================================================================
def run_benchmark():
    if not os.path.exists(ANN_FILE):
        print(f"\n[ERROR] Annotation file not found -> {ANN_FILE}")
        return

    coco_gt = COCO(ANN_FILE)
    img_ids = coco_gt.getImgIds()
    
    print(f"Loading model: {MODEL_PATH}")
    model = get_custom_maskrcnn(4).to(device)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    except FileNotFoundError:
        print(f"[ERROR] {MODEL_PATH} not found!")
        return
    model.eval()

    if SAVE_IMAGES: os.makedirs(OUTPUT_DIR, exist_ok=True)

    timers = {
        '1_Otsu_FindContours': [], '2_ShedFilter': [], '3_GT_JSON_Extract': [], 
        '4_Crop_and_Align': [], '5_Padding': [], '6_Tensor_To_GPU': [], 
        '7_Model_Inference': [], '8_Mask_Thresholding': [], '9_IoU_Math': [], 
        '10_MIC_Calc': [], '11_Image_Save': [] 
    }

    global_ious = []
    print(f"\nData: {DATASET_LOCATION} | Split: {SPLIT}")
    print(f"{'Image':<25} | {'Parts':<4} | {'Total time (ms)':<15} | {'IoU (Crop)':<10}")
    print("-" * 75)

    for img_id in img_ids:
        img_info = coco_gt.loadImgs(img_id)[0]
        img_name = img_info['file_name']
        img_orig = cv2.imread(os.path.join(IMG_DIR, img_name))
        if img_orig is None: continue
        h_orig, w_orig = img_orig.shape[:2]
        
        # --- 1. Otsu & Contours ---
        t0 = time.perf_counter()
        img_bright = cv2.convertScaleAbs(img_orig, alpha=1.0, beta=PARAMS['brightness_beta'])
        gray = cv2.cvtColor(img_bright, cv2.COLOR_BGR2GRAY)
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        timers['1_Otsu_FindContours'].append((time.perf_counter() - t0)*1000)

        # --- 2. Shed Filter ---
        t0 = time.perf_counter()
        valid_cnts = shed_filter(contours, w_orig, h_orig)
        timers['2_ShedFilter'].append((time.perf_counter() - t0)*1000)

        if not valid_cnts:
            continue

        # --- 3. JSON GT Extraction ---
        t0 = time.perf_counter()
        ann_ids = coco_gt.getAnnIds(imgIds=img_id, catIds=[PARAMS['grip_area_id']])
        anns = coco_gt.loadAnns(ann_ids)
        full_gt_mask = np.zeros((h_orig, w_orig), dtype=np.uint8)
        for ann in anns: full_gt_mask = np.maximum(full_gt_mask, coco_gt.annToMask(ann))
        timers['3_GT_JSON_Extract'].append((time.perf_counter() - t0)*1000)

        crop_ious = []

        for idx, cnt in enumerate(valid_cnts):
            # --- 4. Crop & Align ---
            t0 = time.perf_counter()
            img_crop, gt_crop = crop_and_rotate_perfect(img_orig, full_gt_mask, cnt)
            timers['4_Crop_and_Align'].append((time.perf_counter() - t0)*1000)

            # --- 5. Padding ---
            t0 = time.perf_counter()
            img_input, gt_input = prepare_inputs_perfect(img_crop, gt_crop)
            timers['5_Padding'].append((time.perf_counter() - t0)*1000)

            # --- 6. Tensor preparation ---
            t0 = time.perf_counter()
            img_rgb = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)
            img_tensor = (img_rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)
            img_tensor = torch.from_numpy(img_tensor).unsqueeze(0).to(device)
            timers['6_Tensor_To_GPU'].append((time.perf_counter() - t0)*1000)

            # --- 7. Inference ---
            t0 = time.perf_counter()
            pred_mask_input = np.zeros((PARAMS['target_h'], PARAMS['target_w']), dtype=np.uint8)
            with torch.no_grad():
                prediction = model(img_tensor)[0]
            timers['7_Model_Inference'].append((time.perf_counter() - t0)*1000)

            # --- 8. Mask Thresholding ---
            t0 = time.perf_counter()
            if len(prediction['masks']) > 0:
                for j, label in enumerate(prediction['labels']):
                    if label.item() == PARAMS['grip_area_id'] and prediction['scores'][j].item() > 0.1:
                        pred_mask_input = (prediction['masks'][j, 0].cpu().numpy() > PARAMS['mask_threshold']).astype(np.uint8)
                        break
            timers['8_Mask_Thresholding'].append((time.perf_counter() - t0)*1000)

            # --- 9. IoU Calculation ---
            t0 = time.perf_counter()
            iou = calculate_iou(pred_mask_input, gt_input)
            crop_ious.append(iou)
            timers['9_IoU_Math'].append((time.perf_counter() - t0)*1000)

            # --- 10. TRUE MIC (Maximum Inscribed Circle) computation ---
            t0 = time.perf_counter()
            mic_center, mic_radius = calculate_mic(pred_mask_input)
            timers['10_MIC_Calc'].append((time.perf_counter() - t0)*1000)

            # --- 11. Image Saving ---
            t0 = time.perf_counter()
            if SAVE_IMAGES:
                overlay = img_input.copy()
                overlay[gt_input > 0] = [0, 255, 0] # Ground truth (green)
                overlay[pred_mask_input > 0] = [0, 0, 255] # Prediction (red)
                overlap = np.logical_and(gt_input > 0, pred_mask_input > 0)
                overlay[overlap] = [0, 255, 255] # Overlap (yellow)
                
                cv2.addWeighted(overlay, 0.4, img_input, 0.6, 0, img_input)
                
                # Draw the TRUE MIC point (circle and center)
                if mic_center and mic_radius > 0:
                    cx, cy = mic_center
                    r = int(mic_radius)
                    # Draw the inscribed circle
                    cv2.circle(img_input, (cx, cy), r, (0, 0, 255), 2)
                    # Center point (plus sign)
                    cv2.line(img_input, (cx - 5, cy), (cx + 5, cy), (0, 0, 255), 2)
                    cv2.line(img_input, (cx, cy - 5), (cx, cy + 5), (0, 0, 255), 2)
                    # Label
                    cv2.putText(img_input, f"MIC(r={r})", (cx + int(r) + 5, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

                cv2.imwrite(os.path.join(OUTPUT_DIR, f"iou_{iou*100:.0f}_MIC_p{idx+1}_{img_name.split('.')[0]}.jpg"), img_input)
            timers['11_Image_Save'].append((time.perf_counter() - t0)*1000)

        avg_img_iou = np.mean(crop_ious) if crop_ious else 0
        global_ious.append(avg_img_iou)
        
        img_total_t = sum([t[-1] for k, t in timers.items() if t and k.startswith(('4','5','6','7','8','9','10'))]) 
        print(f"{img_name[:25]:<25} | {len(valid_cnts):<4} | {img_total_t:>13.1f} | %{avg_img_iou*100:.1f}")

    print("\n" + "="*75)
    print("MICRO-TIMING ANALYSIS (mean time per step)")
    print("="*75)
    
    total_pipeline_time = 0
    for key, times_list in sorted(timers.items()):
        if not times_list: continue
        avg_time = np.mean(times_list)
        total_pipeline_time += avg_time
        step_name = key.split('_', 1)[1].replace('_', ' ')
        print(f"  {step_name:<25} : {avg_time:>6.2f} ms")
        
    print("-" * 75)
    print(f" TOTAL LATENCY (per part): {total_pipeline_time:>6.2f} ms")
    print("=" * 75)
    print(f"OVERALL TEST IoU: %{np.mean(global_ious)*100:.2f}")

if __name__ == "__main__":
    run_benchmark()