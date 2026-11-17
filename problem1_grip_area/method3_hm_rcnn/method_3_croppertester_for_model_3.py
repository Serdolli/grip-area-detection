import cv2
import numpy as np
import torch
import time
import os
import glob
import torchvision
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

PROJECT_ROOT = os.environ.get("GRIP_PROJECT_ROOT", os.path.abspath("project_root"))   # see README

# =====================================================================================================
# [REALLIFE] real-life mode: camera calibration + transformation from unifier_3 (no heatmaps)
#   step 1  undistort the photo                      cv2.undistort(img, mtx, dist, None, mtx)   (as in unifier_3 maskrcnn_v4.py)
#   step 2  the method below finds the MIC
#   step 3  MIC centre -> robot mm                   p = H @ [x, y, 1]; X = p[0]/p[2]; Y = p[1]/p[2]   (as in unifier_v6.py)
# =====================================================================================================
import sys as _sys, pickle as _pickle, csv as _csv
RL_CALIB_PKL = os.path.join(PROJECT_ROOT, r"codes_results\grippointfinder\unified_point_finder algorithm\unifier_3\camera_calibration\calibration_outputs\Camera_Calibration_Params.pkl")      # written by unifier_3: camera_calibration_heatmap_v6.py
RL_H_NPY     = os.path.join(PROJECT_ROOT, r"codes_results\grippointfinder\unified_point_finder algorithm\unifier_3\transformation_matrix\transformation_matrix_error_heatmaps\H.npy")       # written by unifier_3: transformation_matrix_residual_heatmap_realworlduse_v3.py
RL_INPUT     = os.path.join(PROJECT_ROOT, r"codes_results\conforance_codes\use_ready_codes\input")      # <- put the part photos here
RL_OUTPUT    = os.path.join(PROJECT_ROOT, r"codes_results\conforance_codes\use_ready_codes\output\method_3")
os.makedirs(RL_INPUT, exist_ok=True); os.makedirs(RL_OUTPUT, exist_ok=True)
print("=" * 100)
print(" REAL-LIFE MODE | Method 3 - HM R-CNN (croppertester_for_model_3, model_3)")
_rl_missing = [p for p in (RL_CALIB_PKL, RL_H_NPY) if not os.path.exists(p)]
if _rl_missing:
    print(" NOT READY - unifier_3 has not produced these files yet:")
    for _p in _rl_missing: print("    missing:", _p)
    print(" Run unifier_3's camera calibration and transformation first, then run this code again.")
    print("=" * 100); _sys.exit(1)
with open(RL_CALIB_PKL, "rb") as _f: _rl_cal = _pickle.load(_f)
RL_MTX, RL_DIST = np.asarray(_rl_cal["mtx"], np.float64), np.asarray(_rl_cal["dist"], np.float64)
RL_H = np.load(RL_H_NPY)
_rl_points, _rl_state = {}, {"image": None, "csv_started": False}

def rl_undistort(img):                                   # step 1
    return cv2.undistort(img, RL_MTX, RL_DIST, None, RL_MTX)

def rl_pixel_to_robot(x, y):                             # step 3
    p = RL_H @ np.array([x, y, 1.0], dtype=np.float64)
    return float(p[0] / p[2]), float(p[1] / p[2])

def rl_mic(mask):                                        # distance-transform MIC (same as calculate_mic in ..._cropper_mic.py)
    m = (np.asarray(mask) > 0).astype(np.uint8)
    if cv2.countNonZero(m) == 0: return None, 0.0
    _, r, _, loc = cv2.minMaxLoc(cv2.distanceTransform(m, cv2.DIST_L2, 5))
    return loc, float(r)

def rl_report(image, part, x, y, r, from_K=None):
    """prints + saves one grip point. from_K: camera matrix of the image the pixel was found in, if it is not RL_MTX"""
    image = image if image is not None else _rl_state["image"]
    xb, yb = float(x), float(y)
    if from_K is not None:                               # move the pixel into the frame H was made in
        q = RL_MTX @ np.linalg.inv(np.asarray(from_K, np.float64)) @ np.array([x, y, 1.0])
        xb, yb = float(q[0] / q[2]), float(q[1] / q[2])
    X, Y = rl_pixel_to_robot(xb, yb)
    print(f"GRIP | {image} | part {part} | pixel=({float(x):.1f}, {float(y):.1f}) | r={float(r):.1f} px | robot X={X:.2f} mm, Y={Y:.2f} mm", flush=True)
    _rl_points.setdefault(image, []).append((float(x), float(y), float(r), X, Y, part))
    with open(os.path.join(RL_OUTPUT, "grip_points.csv"), "a" if _rl_state["csv_started"] else "w", newline="", encoding="utf-8") as f:
        w = _csv.writer(f)
        if not _rl_state["csv_started"]: w.writerow(["image", "part", "x_px", "y_px", "radius_px", "robot_X_mm", "robot_Y_mm"])
        w.writerow([image, part, round(float(x), 2), round(float(y), 2), round(float(r), 2), round(X, 3), round(Y, 3)])
    _rl_state["csv_started"] = True

def rl_save_overlay(img, image):
    image = image if image is not None else _rl_state["image"]
    vis = img.copy() if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    k = max(1.0, vis.shape[1] / 1280.0)
    for x, y, r, X, Y, part in _rl_points.get(image, []):
        c = (int(round(x)), int(round(y)))
        cv2.circle(vis, c, int(round(r)), (0, 0, 255), int(2 * k)); cv2.drawMarker(vis, c, (0, 255, 0), cv2.MARKER_CROSS, int(20 * k), int(2 * k))
        cv2.putText(vis, f"P{part}: X={X:.1f} Y={Y:.1f} mm", (c[0] + int(12 * k), c[1] - int(12 * k)), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * k, (0, 255, 255), int(max(1, 1.5 * k)))
    cv2.imwrite(os.path.join(RL_OUTPUT, f"{os.path.splitext(os.path.basename(image))[0]}_GRIP_overlay.jpg"), vis)

_s = float(np.hypot(*(np.array(rl_pixel_to_robot(RL_MTX[0, 2] + 1, RL_MTX[1, 2])) - np.array(rl_pixel_to_robot(RL_MTX[0, 2], RL_MTX[1, 2])))))
print(f"   camera calibration : {RL_CALIB_PKL}")
print(f"                        fx={RL_MTX[0, 0]:.1f} fy={RL_MTX[1, 1]:.1f} cx={RL_MTX[0, 2]:.1f} cy={RL_MTX[1, 2]:.1f}")
print(f"   transformation H   : {RL_H_NPY}")
print(f"                        scale {_s:.4f} mm/px  (a 1280 px wide photo covers about {1280 * _s:.0f} mm)")
print(f"   photos read from   : {RL_INPUT}")
print(f"   results saved to   : {RL_OUTPUT}")
print("=" * 100, flush=True)
# ============================================ end of the real-life setup ============================================
def _crop_point_to_full(img, contour, pt, r):   # [REALLIFE] one point: 160x1600 crop -> full image (inverse of crop_and_rotate + center_pad_to_target)
    rect = cv2.minAreaRect(contour); center, size, angle = rect
    if size[0] > size[1]:
        angle -= 90; size = (size[1], size[0])
    diagonal = int(np.sqrt(size[0]**2 + size[1]**2)) + 10
    x1, y1 = int(max(0, center[0] - diagonal//2)), int(max(0, center[1] - diagonal//2))
    M = cv2.getRotationMatrix2D((center[0]-x1, center[1]-y1), angle, 1.0)
    w_i, h_i = int(size[0]), int(size[1])
    m = int(w_i * PARAMS['safety_crop_percent'])
    w_c, h_c = (w_i - 2 * m if m > 0 else w_i), h_i
    s = min(PARAMS['target_w'] / w_c, PARAMS['target_h'] / h_c)
    new_w, new_h = int(w_c * s), int(h_c * s)
    left, top = (PARAMS['target_w'] - new_w) // 2, (PARAMS['target_h'] - new_h) // 2
    u = (pt[0] - left + 0.5) * w_c / new_w - 0.5 + m
    v = (pt[1] - top + 0.5) * h_c / new_h - 0.5
    xr = (center[0]-x1) - (w_i - 1) / 2.0 + u
    yr = (center[1]-y1) - (h_i - 1) / 2.0 + v
    Mi = cv2.invertAffineTransform(M)
    return Mi[0, 0]*xr + Mi[0, 1]*yr + Mi[0, 2] + x1, Mi[1, 0]*xr + Mi[1, 1]*yr + Mi[1, 2] + y1, r / s

# ============================================================================
# HYPERPARAMETERS AND PATHS
# ============================================================================
PATHS = {
    # Path of the trained vertical model
    'model_path': os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models", "sametkalip_maskrcnn_160x1600_2_master.pth"),
    'input_folder': os.path.join(PROJECT_ROOT, r"codes_results\conforance_codes\use_ready_codes\input"),                                  
    'output_folder': os.path.join(PROJECT_ROOT, r"codes_results\conforance_codes\use_ready_codes\output\method_3")                           
}

PARAMS = {
    'brightness_beta': 220,          
    'min_area': 5000,               
    'max_area': 30000000,             
    'edge_tolerance': 1,            
    'rect_ratio_limit': 1.325,       
    'safety_crop_percent': 0.02,    
    
    # VERTICAL SIZES (same as in training)
    'target_w': 160,               
    'target_h': 1600,                
    
    'mask_threshold': 0.5,          # 0.1 was too low, 0.5 gives a cleaner result
    'mask_color': [0, 0, 125],      
    'mask_alpha': 0.5,
    'grip_area_label_id': 2         
}

# --- HELPER FUNCTIONS ---

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
        rect_area = w * h
        if area == 0: continue
        if (rect_area / area) >= PARAMS['rect_ratio_limit']: continue 
        kept.append(c)
    return kept

def crop_and_rotate(img, contour):
    rect = cv2.minAreaRect(contour)
    center, size, angle = rect
    
    # Always orient the part vertically (long edge along the Y axis)
    if size[0] > size[1]:
        angle -= 90
        size = (size[1], size[0])
        
    diagonal = int(np.sqrt(size[0]**2 + size[1]**2)) + 10
    h_img, w_img = img.shape[:2]
    x1, y1 = int(max(0, center[0] - diagonal//2)), int(max(0, center[1] - diagonal//2))
    x2, y2 = int(min(w_img, center[0] + diagonal//2)), int(min(h_img, center[1] + diagonal//2))
    img_roi = img[y1:y2, x1:x2]
    mask_roi = np.zeros((y2-y1, x2-x1), dtype=np.uint8)
    cv2.drawContours(mask_roi, [contour - np.array([x1, y1])], -1, 255, -1)
    M = cv2.getRotationMatrix2D((center[0]-x1, center[1]-y1), angle, 1.0)
    img_rot = cv2.warpAffine(img_roi, M, (x2-x1, y2-y1), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
    mask_rot = cv2.warpAffine(mask_roi, M, (x2-x1, y2-y1), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    img_crop = cv2.getRectSubPix(img_rot, (int(size[0]), int(size[1])), (center[0]-x1, center[1]-y1))
    
    m = int(img_crop.shape[1] * PARAMS['safety_crop_percent'])
    if m > 0:
        img_crop = img_crop[:, m:-m]
        
    return img_crop

def center_pad_to_target(img, target_w=PARAMS['target_w'], target_h=PARAMS['target_h']):
    h_crop, w_crop = img.shape[:2]
    scale = min(target_w / w_crop, target_h / h_crop)
    
    new_w = int(w_crop * scale)
    new_h = int(h_crop * scale)
    
    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    pad_w = target_w - new_w
    pad_h = target_h - new_h
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    padded_img = cv2.copyMakeBorder(resized_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return padded_img

# ==============================================================================
# MODEL BUILDER (same anchors as in training)
# ==============================================================================
def get_custom_maskrcnn(num_classes):
    print("Loading vertical thin, elongated anchors (1.0, 2.0, 5.0, 10.0)...")
    
    # same setting as in training
    anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
    aspect_ratios = ((1.0, 2.0, 5.0, 10.0),) * len(anchor_sizes) 
    rpn_anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)

    mask_roi_pool = MultiScaleRoIAlign(
        featmap_names=['0', '1', '2', '3'], 
        output_size=28, 
        sampling_ratio=2
    )

    model = torchvision.models.detection.maskrcnn_resnet50_fpn(
        weights=None,
        min_size=160,
        max_size=1600,
        rpn_anchor_generator=rpn_anchor_generator,
        mask_roi_pool=mask_roi_pool,
    )

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    return model

# --- MAIN PIPELINE ---

def run_inference(img_orig, model, device):
    global_t0 = time.perf_counter()
    
    t0 = time.perf_counter()
    img_bright = cv2.convertScaleAbs(img_orig, alpha=1.0, beta=PARAMS['brightness_beta'])
    gray_coarse = cv2.cvtColor(img_bright, cv2.COLOR_BGR2GRAY)
    _, otsu = cv2.threshold(gray_coarse, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_cnts = shed_filter(contours, img_orig.shape[1], img_orig.shape[0])
    processed_crops = [] 

    if not valid_cnts:
        return processed_crops
    
    for i, cnt in enumerate(valid_cnts):
        img_crop = crop_and_rotate(img_orig, cnt) 
        img_padded = center_pad_to_target(img_crop)
        
        img_padded_rgb = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)
        img_tensor = img_padded_rgb.astype(np.float32) / 255.0  
        img_tensor = img_tensor.transpose((2, 0, 1)) 
        img_tensor = torch.from_numpy(img_tensor).unsqueeze(0).to(device) 

        with torch.no_grad():
            prediction = model(img_tensor)[0]

        if len(prediction['boxes']) > 0:
            for j in range(len(prediction['labels'])):
                label_id = prediction['labels'][j].item()
                score = prediction['scores'][j].item()
                
                # Draw only Grip Area with high scores
                if label_id == PARAMS['grip_area_label_id'] and score > 0.3:
                    pred_mask = prediction['masks'][j, 0].cpu().numpy()
                    pred_mask = (pred_mask > PARAMS['mask_threshold']).astype(np.uint8) * 255
                    _mc, _mr = rl_mic(pred_mask)   # [REALLIFE] step 2b: MIC of the predicted grip mask
                    if _mc is not None:
                        _fx, _fy, _fr = _crop_point_to_full(img_orig, cnt, _mc, _mr)
                        rl_report(None, i + 1, _fx, _fy, _fr)   # step 3: MIC centre -> robot mm
                    
                    colored_mask = np.zeros_like(img_padded)
                    colored_mask[pred_mask == 255] = PARAMS['mask_color']
                    cv2.addWeighted(colored_mask, PARAMS['mask_alpha'], img_padded, 1.0, 0, img_padded)
                    print(f"  [OK] Part {i+1}: Grip Area found! Score: %{score*100:.1f}")
                    break 
        
        processed_crops.append(img_padded)

    return processed_crops

if __name__ == "__main__":
    os.makedirs(PATHS['output_folder'], exist_ok=True)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # Build the model with the vertical architecture
    model = get_custom_maskrcnn(num_classes=4)
    model.load_state_dict(torch.load(PATHS['model_path'], map_location=device))
    model.to(device)
    model.eval() 

    image_files = glob.glob(os.path.join(PATHS['input_folder'], '*.*')) 
    
    for img_path in image_files:
        img_name = os.path.basename(img_path)
        img_basename, img_ext = os.path.splitext(img_name)
        img = cv2.imread(img_path)
        if img is None: continue
        img = rl_undistort(img)   # [REALLIFE] step 1: undistort first
        _rl_state["image"] = img_name
            
        result_crops = run_inference(img, model, device)
        
        for idx, crop_img in enumerate(result_crops):
            save_path = os.path.join(PATHS['output_folder'], f"{img_basename}_part{idx+1}{img_ext}")
            cv2.imwrite(save_path, crop_img)
        
        print(f"Processed: {img_name}")
        rl_save_overlay(img, img_name)   # [REALLIFE]
        
    print("\nTEST FINISHED! Results are in the folder.")