import cv2
import numpy as np
import os
import json

PROJECT_ROOT = os.environ.get("GRIP_PROJECT_ROOT", os.path.abspath("project_root"))   # see README

# =============================================================================
# --- 1. CONFIGURATION ---
# =============================================================================
INPUT_FOLDER = os.path.join(PROJECT_ROOT, r"datasets\derivative_datasets\blackbackground_withlabels\train")
JSON_PATH = os.path.join(INPUT_FOLDER, "_annotations.coco.json") 
OUTPUT_MAIN_FOLDER = os.path.join(PROJECT_ROOT, r"codes_results\gripareafinder\clahe_method\tolerance_finder\test_37")

CONFIG = {
    'alternatif_3': True,       
    'alternatif': False,
    'alternatif_2': False,
    'eat': False,               
    'horizontal_kernel': True,  
    'rect_closing_after_invert': True, 
    'saveall': False,             
    'savefinal': True             
}

PARAMS = {
    'target_class_id': 2.0,  
    'brightness_beta': 220,  
    'clahe_clip': 5.0,
    'clahe_grid': (5, 8),
    'morph_kernel': (21, 21),
    'horz_closing_len': 9,
    'rect_closing_size': (1, 1),
    'gauss_ksize': (5, 5),
    'gauss_sigma': 0,
    'min_area': 10000,
    'max_area': 7000000,
    'edge_tolerance': 0,
    'rect_ratio_limit': 1.325,
    'safety_crop_percent': 4,
    'eat_px': 0 
}

# =============================================================================
# --- 2. HELPER FUNCTIONS ---
# =============================================================================

def get_l_channel(img):
    if len(img.shape) == 3:
        bgr = img[:, :, :3] if img.shape[2] == 4 else img
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)
        return lab[:, :, 0]
    return img

def shed_filter(contours, width, height):
    kept = []
    for c in contours:
        A = cv2.contourArea(c)
        if not (PARAMS['min_area'] <= A <= PARAMS['max_area']): continue
        pts = c.reshape(-1, 2)
        touches_edge = (pts[:,0].min() <= PARAMS['edge_tolerance'] or 
                        pts[:,0].max() >= width - 1 - PARAMS['edge_tolerance'] or 
                        pts[:,1].min() <= PARAMS['edge_tolerance'] or 
                        pts[:,1].max() >= height - 1 - PARAMS['edge_tolerance'])
        if touches_edge: continue
        rect = cv2.minAreaRect(c)
        (_, _), (w, h), _ = rect
        if A == 0 or (w * h) / A >= PARAMS['rect_ratio_limit']: continue 
        kept.append(c)
    return kept

def crop_and_rotate(img, contour, eat=False):
    rect = cv2.minAreaRect(contour)
    center, size, angle = rect
    center = tuple(map(float, center))
    size = tuple(map(float, size))
    if size[0] < size[1]:
        angle += 90
        size = (size[1], size[0])
    diagonal = int(np.sqrt(size[0]**2 + size[1]**2)) + 10
    rx = int(center[0] - diagonal // 2)
    ry = int(center[1] - diagonal // 2)
    rw = diagonal
    rh = diagonal
    h_img, w_img = img.shape[:2]
    x1, y1 = max(0, rx), max(0, ry)
    x2, y2 = min(w_img, rx + rw), min(h_img, ry + rh)
    
    img_roi = img[y1:y2, x1:x2]
    mask_roi = np.zeros((y2-y1, x2-x1), dtype=np.uint8)
    contour_roi = contour - np.array([x1, y1])
    cv2.drawContours(mask_roi, [contour_roi], -1, 255, -1)
    
    center_roi = (center[0] - x1, center[1] - y1)
    M = cv2.getRotationMatrix2D(center_roi, angle, 1.0)
    img_rot = cv2.warpAffine(img_roi, M, (x2-x1, y2-y1), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
    mask_rot = cv2.warpAffine(mask_roi, M, (x2-x1, y2-y1), flags=cv2.INTER_NEAREST, borderValue=255)
    
    size_int = tuple(map(int, size))
    img_crop = cv2.getRectSubPix(img_rot, size_int, center_roi)
    mask_crop = cv2.getRectSubPix(mask_rot, size_int, center_roi)
    
    h_c, w_c = img_crop.shape[:2]
    crop_margin = int(w_c * PARAMS['safety_crop_percent'] / 100.0)
    
    if crop_margin > 0 and w_c > (2 * crop_margin):
        img_crop = img_crop[:, crop_margin : w_c - crop_margin]
        mask_crop = mask_crop[:, crop_margin : w_c - crop_margin]
        
    return img_crop, mask_crop, M, center_roi, size_int, x1, y1, crop_margin

def get_coco_instances_by_filename(coco_data, filename, target_id):
    """
    Finds the image in the JSON file and returns each polygon of the target class (griparea)
    SEPARATELY as a list.
    """
    img_id = None
    for img in coco_data['images']:
        if img['file_name'] == filename:
            img_id = img['id']
            break
            
    instances = []
    if img_id is None:
        return instances

    for ann in coco_data['annotations']:
        if ann['image_id'] == img_id and int(ann['category_id']) == int(target_id):
            for polygon in ann['segmentation']:
                pts = np.array(polygon, np.int32).reshape((-1, 1, 2))
                instances.append(pts)
                
    return instances

def calculate_single_mask_iou(mask_pred, mask_gt):
    pred_bool = mask_pred > 127
    gt_bool = mask_gt > 127
    
    intersection = np.logical_and(pred_bool, gt_bool).sum()
    union = np.logical_or(pred_bool, gt_bool).sum()
    
    if union == 0:
        return 0.0
    return float(intersection) / float(union)

def match_and_calculate_ious(pred_polys, gt_polys, img_shape):
    """
    Matches every predicted part (polygon) with the ground-truth parts.
    The highest IoU is taken as the score of that part.
    """
    h, w = img_shape
    piece_ious = []
    
    # 1. Turn the ground-truth (GT) polygons into masks (one mask per part)
    gt_masks = []
    for gt_p in gt_polys:
        m = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(m, [gt_p], 255)
        gt_masks.append(m)
        
    # 2. For every predicted part, find the best matching ground-truth part
    for i, p_poly in enumerate(pred_polys):
        p_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(p_mask, [p_poly], 255)
        
        best_iou = 0.0
        # Which ground-truth part overlaps this predicted part the most?
        for gt_mask in gt_masks:
            iou = calculate_single_mask_iou(p_mask, gt_mask)
            if iou > best_iou:
                best_iou = iou
                
        piece_ious.append(best_iou)
        
    return piece_ious

# =============================================================================
# --- 3. MAIN PIPELINE ---
# =============================================================================

def process_pipeline():
    if not os.path.exists(OUTPUT_MAIN_FOLDER): os.makedirs(OUTPUT_MAIN_FOLDER)
    
    if not os.path.exists(JSON_PATH):
        print(f"ERROR: COCO JSON file not found: {JSON_PATH}")
        return
        
    with open(JSON_PATH, 'r') as f:
        coco_data = json.load(f)
        
    files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    clahe = cv2.createCLAHE(clipLimit=PARAMS['clahe_clip'], tileGridSize=PARAMS['clahe_grid'])
    kernel_morph = cv2.getStructuringElement(cv2.MORPH_RECT, PARAMS['morph_kernel'])
    kernel_horz = cv2.getStructuringElement(cv2.MORPH_RECT, (PARAMS['horz_closing_len'], 1))
    kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, PARAMS['rect_closing_size'])

    global_ious = [] # Keeps the IoU of the parts over the whole dataset

    print(f"Processing {len(files)} files...\n" + "-"*40)

    for filename in files:
        img_path = os.path.join(INPUT_FOLDER, filename)
        img_orig = cv2.imread(img_path)
        if img_orig is None: continue
        h_orig, w_orig = img_orig.shape[:2]
        base_name = os.path.splitext(filename)[0]
        
        bright_img = cv2.convertScaleAbs(img_orig, alpha=1.0, beta=PARAMS['brightness_beta'])
        gray_coarse = cv2.cvtColor(bright_img, cv2.COLOR_BGR2GRAY)
        _, otsu_coarse = cv2.threshold(gray_coarse, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours_ext, _ = cv2.findContours(otsu_coarse, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = shed_filter(contours_ext, w_orig, h_orig)

        # Predicted polygons are collected in this list (part by part)
        pred_polygons = []
        
        # Overall mask and image for drawing (kept only for saving)
        prediction_mask_full = np.zeros((h_orig, w_orig), dtype=np.uint8)
        vis = img_orig.copy()

        for main_contour in valid_contours:
            img_cropped, mask_cropped, M, center_roi, size_int, x1, y1, crop_margin = crop_and_rotate(img_orig, main_contour, eat=CONFIG['eat'])
            
            l_clahe = clahe.apply(get_l_channel(img_cropped))
            tophat_t1 = cv2.morphologyEx(l_clahe, cv2.MORPH_TOPHAT, kernel_morph)
            enhanced = cv2.add(l_clahe, tophat_t1)
            blurred = cv2.GaussianBlur(enhanced, PARAMS['gauss_ksize'], PARAMS['gauss_sigma'])

            _, binary_final = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            binary_final = cv2.bitwise_and(binary_final, binary_final, mask=mask_cropped)
            if CONFIG['horizontal_kernel']:
                binary_final = cv2.morphologyEx(binary_final, cv2.MORPH_CLOSE, kernel_horz)
            binary_final = cv2.bitwise_not(binary_final)
            if CONFIG['rect_closing_after_invert']:
                binary_final = cv2.morphologyEx(binary_final, cv2.MORPH_CLOSE, kernel_rect)
            binary_final = cv2.bitwise_and(binary_final, binary_final, mask=mask_cropped)

            contours_final, _ = cv2.findContours(binary_final, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours_final:
                largest_cnt = max(contours_final, key=cv2.contourArea)
                
                pts = largest_cnt.reshape(-1, 2).astype(np.float32)
                if crop_margin > 0 and size_int[0] > (2 * crop_margin): pts[:, 0] += crop_margin
                pts[:, 0] += (center_roi[0] - size_int[0] / 2.0)
                pts[:, 1] += (center_roi[1] - size_int[1] / 2.0)
                
                M_inv = cv2.invertAffineTransform(M)
                pts_roi = cv2.transform(pts.reshape(-1, 1, 2), M_inv).reshape(-1, 2)
                pts_roi[:, 0] += x1
                pts_roi[:, 1] += y1
                
                # FOR PART-BY-PART ANALYSIS: add the polygon of every part to the list
                poly_int32 = pts_roi.astype(np.int32)
                pred_polygons.append(poly_int32)
                
                # Draw only for visualization
                cv2.fillPoly(prediction_mask_full, [poly_int32], 255)

        # Get the ground-truth polygons from the JSON PART BY PART
        gt_polygons = get_coco_instances_by_filename(coco_data, filename, PARAMS['target_class_id'])
        
        # PART-BY-PART IoU
        piece_ious = match_and_calculate_ious(pred_polygons, gt_polygons, (h_orig, w_orig))
        
        # Add the IoU values of the parts in this image to the global list
        global_ious.extend(piece_ious)
        
        # Print the result of every part to the console
        print(f"[{filename}] Parts found: {len(pred_polygons)} | Ground-truth parts: {len(gt_polygons)}")
        for idx, iou in enumerate(piece_ious):
            print(f"   -> Part {idx+1} IoU: {iou:.4f}")

        # Saving the visualization
        if CONFIG['savefinal']:
            cv2.drawContours(vis, pred_polygons, -1, (0, 255, 0), 2) # Predictions (green)
            cv2.drawContours(vis, gt_polygons, -1, (0, 0, 255), 1)   # Ground truth (red)
            
            # Write the mean IoU on the image
            img_mean_iou = np.mean(piece_ious) if piece_ious else 0.0
            cv2.putText(vis, f"Image mean IoU: {img_mean_iou:.2f}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imwrite(os.path.join(OUTPUT_MAIN_FOLDER, f"COMPARE_{base_name}.jpg"), vis)

    if global_ious:
        print("-" * 40)
        print(f">>> Mean IoU per part over the dataset: {np.mean(global_ious):.4f}")

if __name__ == "__main__":
    process_pipeline()