import cv2
import numpy as np
import torch
import os
import glob
import pickle
import json
import torchvision
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

PROJECT_ROOT = os.environ.get("GRIP_PROJECT_ROOT", os.path.abspath("project_root"))   # see README

# ============================================================================
# HYPERPARAMETERS AND PATHS
# ============================================================================
PATHS = {
    'model_path': os.path.join(PROJECT_ROOT, r"codes_results\gripareafinder\deep_learning_method\maskrcnn\model_3\model\sametkalip_maskrcnn_160x1600_2_master.pth"),
    'input_folder': os.path.join(PROJECT_ROOT, r"codes_results\grippointfinder\unified_point_finder algorithm\unifier_3\computer_vision\photos\photo_10"),                                  
    'output_folder': os.path.join(PROJECT_ROOT, r"codes_results\grippointfinder\unified_point_finder algorithm\unifier_3\computer_vision\outputs\photo_10"),
    'master_output_folder': os.path.join(PROJECT_ROOT, r"codes_results\grippointfinder\unified_point_finder algorithm\unifier_3\computer_vision\debug_photos"),
    'calib_path': os.path.join(PROJECT_ROOT, r"codes_results\grippointfinder\unified_point_finder algorithm\unifier_2\test_2\other_inputs\Camera_Calibration_Params.pkl")
}

PARAMS = {
    'brightness_beta': 220,          
    'min_area': 5000,                
    'max_area': 30000000,              
    'edge_tolerance': 1,            
    'rect_ratio_limit': 1.325,       
    'safety_crop_percent': 0.02,    
    'target_w': 160,                
    'target_h': 1600,                
    'mask_threshold': 0.45,          
    'mask_color': [0, 0, 125],      
    'mask_alpha': 0.6, 
    'grip_area_label_id': 2,
    'trim_percent': 0.02
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def shed_filter(contours, width, height):
    kept = []
    for c in contours:
        area = cv2.contourArea(c)
        if not (PARAMS['min_area'] <= area <= PARAMS['max_area']): continue
        rect = cv2.minAreaRect(c)
        (_, _), (w, h), _ = rect
        rect_area = w * h
        if area == 0: continue
        if (rect_area / area) >= PARAMS['rect_ratio_limit']: continue 
        kept.append(c)
    return kept

def crop_and_rotate(img, contour):
    rect = list(cv2.minAreaRect(contour))
    center, size, angle = rect[0], rect[1], rect[2]
    angle_used = angle
    if size[0] > size[1]:
        angle_used -= 90
        size = (size[1], size[0])
    diagonal = int(np.sqrt(size[0]**2 + size[1]**2)) + 10
    h_img, w_img = img.shape[:2]
    x1, y1 = int(center[0] - diagonal//2), int(center[1] - diagonal//2)
    x2, y2 = int(center[0] + diagonal//2), int(center[1] + diagonal//2)
    x1_safe, y1_safe = max(0, x1), max(0, y1)
    x2_safe, y2_safe = min(w_img, x2), min(h_img, y2)
    img_roi = img[y1_safe:y2_safe, x1_safe:x2_safe]
    top, bottom, left, right = y1_safe - y1, y2 - y2_safe, x1_safe - x1, x2 - x2_safe
    img_roi_padded = cv2.copyMakeBorder(img_roi, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    M = cv2.getRotationMatrix2D((diagonal//2, diagonal//2), angle_used, 1.0)
    img_rot = cv2.warpAffine(img_roi_padded, M, (diagonal, diagonal), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=[0,0,0])
    img_crop = cv2.getRectSubPix(img_rot, (int(size[0]), int(size[1])), (diagonal//2, diagonal//2))
    m = int(img_crop.shape[1] * PARAMS['safety_crop_percent'])
    if m > 0: img_crop = img_crop[:, m:-m]
    return img_crop, {'center': center, 'size': size, 'angle': angle_used, 'm': m, 'orig_shape': img.shape[:2]}

def center_pad_to_target(img):
    h_c, w_c = img.shape[:2]
    scale = min(PARAMS['target_w'] / w_c, PARAMS['target_h'] / h_c)
    new_w, new_h = int(w_c * scale), int(h_c * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    pad_w, pad_h = PARAMS['target_w'] - new_w, PARAMS['target_h'] - new_h
    t, l = pad_h // 2, pad_w // 2
    padded = cv2.copyMakeBorder(resized, t, pad_h-t, l, pad_w-l, cv2.BORDER_CONSTANT, value=0)
    return padded, {'scale': scale, 'top': t, 'left': l, 'w_c': w_c, 'h_c': h_c}

def inverse_mask(pred_bin, meta_c, meta_p):
    unpad = pred_bin[meta_p['top']:meta_p['top']+int(meta_p['h_c']*meta_p['scale']), 
                     meta_p['left']:meta_p['left']+int(meta_p['w_c']*meta_p['scale'])]
    unscale = cv2.resize(unpad, (meta_p['w_c'], meta_p['h_c']), interpolation=cv2.INTER_NEAREST)
    if meta_c['m'] > 0: unscale = cv2.copyMakeBorder(unscale, 0, 0, meta_c['m'], meta_c['m'], cv2.BORDER_CONSTANT, value=0)
    h_o, w_o = meta_c['orig_shape']
    M_inv = cv2.getRotationMatrix2D((unscale.shape[1]/2, unscale.shape[0]/2), -meta_c['angle'], 1.0)
    M_inv[0, 2] += (meta_c['center'][0] - unscale.shape[1]/2)
    M_inv[1, 2] += (meta_c['center'][1] - unscale.shape[0]/2)
    return cv2.warpAffine(unscale, M_inv, (w_o, h_o), flags=cv2.INTER_NEAREST)

def get_valid_region_mask(meta_c, shape):
    w, h = int(meta_c['size'][0]), int(meta_c['size'][1])
    valid_box = np.ones((h, w), dtype=np.uint8)
    trim_h = int(h * PARAMS['trim_percent'])
    valid_box[:trim_h, :] = 0
    valid_box[-trim_h:, :] = 0
    M_inv = cv2.getRotationMatrix2D((w/2, h/2), -meta_c['angle'], 1.0)
    M_inv[0, 2] += (meta_c['center'][0] - w/2)
    M_inv[1, 2] += (meta_c['center'][1] - h/2)
    return cv2.warpAffine(valid_box, M_inv, (shape[1], shape[0]), flags=cv2.INTER_NEAREST)

def overlay_heatmap(img, dist):
    if not np.any(dist > 0): return img
    norm = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    colored = cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)
    blended = cv2.addWeighted(img, 0.3, colored, 0.7, 0)
    mask = dist > 0
    mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2) 
    res = np.where(mask_3d, blended, img)
    return res

def get_model():
    m = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=None, min_size=160, max_size=1600,
        rpn_anchor_generator=AnchorGenerator(((32,),(64,),(128,),(256,),(512,)), ((1.0, 2.0, 5.0, 10.0),)*5),
        mask_roi_pool=MultiScaleRoIAlign(['0','1','2','3'], 28, 2))
    m.roi_heads.box_predictor = FastRCNNPredictor(m.roi_heads.box_predictor.cls_score.in_features, 4)
    m.roi_heads.mask_predictor = MaskRCNNPredictor(m.roi_heads.mask_predictor.conv5_mask.in_channels, 256, 4)
    return m

# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    os.makedirs(PATHS['output_folder'], exist_ok=True)
    os.makedirs(PATHS['master_output_folder'], exist_ok=True)
    
    print("Loading the camera calibration data...")
    with open(PATHS['calib_path'], 'rb') as f:
        calib_data = pickle.load(f)
        mtx, dist = calib_data['mtx'], calib_data['dist']
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model().to(device)
    model.load_state_dict(torch.load(PATHS['model_path'], map_location=device))
    model.eval()

    for img_path in glob.glob(os.path.join(PATHS['input_folder'], '*.*')):
        name = os.path.splitext(os.path.basename(img_path))[0]
        ext = os.path.splitext(img_path)[1]
        raw_img = cv2.imread(img_path)
        if raw_img is None: continue
        
        # Dynamic sizing
        img = cv2.undistort(raw_img, mtx, dist, None, mtx)
        h, w = img.shape[:2]
        
        bright = cv2.convertScaleAbs(img, alpha=1.0, beta=PARAMS['brightness_beta'])
        _, otsu = cv2.threshold(cv2.cvtColor(bright, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cnts = shed_filter(cv2.findContours(otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], w, h)

        master_heatmap = img.copy()

        for i, c in enumerate(cnts):
            crop, m_c = crop_and_rotate(img, c)
            pad, m_p = center_pad_to_target(crop)
            
            tensor = torch.from_numpy(pad.transpose(2,0,1)).float().div(255).unsqueeze(0).to(device)
            with torch.no_grad():
                pred = model(tensor)[0]
            
            for j in range(len(pred['labels'])):
                if pred['labels'][j] == PARAMS['grip_area_label_id'] and pred['scores'][j] > 0.3:
                    mask_bin = (pred['masks'][j, 0].cpu().numpy() > PARAMS['mask_threshold']).astype(np.uint8) * 255
                    
                    full_m = inverse_mask(mask_bin, m_c, m_p)
                    dist_transform = cv2.distanceTransform(full_m.astype(np.uint8), cv2.DIST_L2, 5)
                    
                    valid_mask = get_valid_region_mask(m_c, (h, w))
                    dist_transform = dist_transform * valid_mask  
                    
                    p_path = os.path.join(PATHS['output_folder'], f"{name}_part{i+1}_distance_transform.pkl")
                    with open(p_path, "wb") as f: pickle.dump(dist_transform.astype(np.float32), f)
                    
                    master_heatmap = overlay_heatmap(master_heatmap, dist_transform)
                    print(f"[OK] {name} part {i+1} processed. Size: {dist_transform.shape}")
                    break
            
        master_output_path = os.path.join(PATHS['master_output_folder'], f"{name}_MASTER_TOPOGRAPHY{ext}")
        cv2.imwrite(master_output_path, master_heatmap)
        print(f"{name} MASTER file saved.\n")

    print("Done!")