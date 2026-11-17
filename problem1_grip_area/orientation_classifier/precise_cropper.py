import cv2
import numpy as np
import os
import warnings
from tqdm import tqdm

PROJECT_ROOT = os.environ.get("GRIP_PROJECT_ROOT", os.path.abspath("project_root"))   # see README

# Silence unnecessary warnings
warnings.filterwarnings("ignore")

# =============================================================================
# --- 1. SETTINGS (CONFIG) ---
# =============================================================================

# 1. RAW PHOTOS
INPUT_RAW_FOLDER = os.path.join(PROJECT_ROOT, r"datasets\derivative_datasets\blackback_ground_alldatasets")

# --- OUTPUT FOLDERS ---
BASE_OUTPUT = os.path.join(PROJECT_ROOT, r"datasets\derivative_datasets\cropped_datasets\cropped_for_rf_better")

PATH_OUTPUT_TXT = os.path.join(BASE_OUTPUT, "txt_data")
PATH_OUTPUT_IMG = os.path.join(BASE_OUTPUT, "images_processed") # processed / annotated images
PATH_OUTPUT_RAW = os.path.join(BASE_OUTPUT, "images_raw")       # raw crops

# --- PARAMETERS ---
PARAMS = {
    # brightness (for detection)
    'brightness_beta': 220,

    'clahe_clip': 8.0,
    'clahe_grid': (8, 8),
    'min_area': 10000,
    'max_area': 7000000,
    'edge_tolerance': 2,
    'rect_ratio_limit': 1.325,
    'safety_crop_percent': 0.1,
    'eat_px': 0,
    
    'morph_kernel_size': (51, 51) 
}

LX_PARAMS = {
    'target_nodes': 1000,
    'offset_start': 0.10,
    'offset_end': 0.2      
}

# =============================================================================
# --- 2. HELPER FUNCTIONS ---
# =============================================================================

def get_l_channel(img):
    if len(img.shape) == 3:
        if img.shape[2] == 4:
            bgr = img[:, :, :3]
        else:
            bgr = img
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
        (x, y), (w, h), angle = rect
        rect_area = w * h
        if A == 0: continue
        ratio = rect_area / A
        if ratio >= PARAMS['rect_ratio_limit']: continue 
        kept.append(c)
    return kept

def crop_and_rotate(img, contour):
    rect = cv2.minAreaRect(contour)
    center, size, angle = rect
    center = tuple(map(float, center))
    size = tuple(map(float, size))
    
    if size[0] < size[1]:
        angle += 90
        size = (size[1], size[0])
    
    diagonal = int(np.sqrt(size[0]**2 + size[1]**2)) + 4
    rx = int(center[0] - diagonal // 2)
    ry = int(center[1] - diagonal // 2)
    rw, rh = diagonal, diagonal
    
    h_img, w_img = img.shape[:2]
    x1 = max(0, rx)
    y1 = max(0, ry)
    x2 = min(w_img, rx + rw)
    y2 = min(h_img, ry + rh)
    
    img_roi = img[y1:y2, x1:x2]
    mask_roi = np.zeros((y2-y1, x2-x1), dtype=np.uint8)
    contour_roi = contour - np.array([x1, y1])
    cv2.drawContours(mask_roi, [contour_roi], -1, 255, -1)
    
    top_pad = max(0, y1 - ry)
    left_pad = max(0, x1 - rx)
    bottom_pad = max(0, rh - (y2 - ry) - top_pad)
    right_pad = max(0, rw - (x2 - rx) - left_pad)
    
    if top_pad > 0 or bottom_pad > 0 or left_pad > 0 or right_pad > 0:
        img_padded = cv2.copyMakeBorder(img_roi, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        mask_padded = cv2.copyMakeBorder(mask_roi, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT, value=0)
    else:
        img_padded = img_roi
        mask_padded = mask_roi
    
    h_p, w_p = img_padded.shape[:2]
    center_padded = (w_p / 2.0, h_p / 2.0)
    
    M = cv2.getRotationMatrix2D(center_padded, angle, 1.0)
    
    img_rot = cv2.warpAffine(img_padded, M, (w_p, h_p), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    mask_rot = cv2.warpAffine(mask_padded, M, (w_p, h_p), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    
    size_int = tuple(map(int, size))
    img_crop = cv2.getRectSubPix(img_rot, size_int, center_padded)
    mask_crop = cv2.getRectSubPix(mask_rot, size_int, center_padded)
    
    h_c_raw, w_c_raw = img_crop.shape[:2]
    width_A = w_c_raw 
    
    crop_margin = int(w_c_raw * PARAMS['safety_crop_percent'])
    if crop_margin > 0 and w_c_raw > (2 * crop_margin):
        img_crop = img_crop[:, crop_margin : w_c_raw - crop_margin]
        mask_crop = mask_crop[:, crop_margin : w_c_raw - crop_margin]
    
    return img_crop, mask_crop, width_A

def get_trimmed_signal_and_coords(img_strip, mask_strip, target_len):
    row_has_data = np.max(mask_strip, axis=1) > 0
    if not np.any(row_has_data):
        return np.zeros(target_len, dtype=np.float32), -1, -1

    indices = np.where(row_has_data)[0]
    y_start, y_end = indices[0], indices[-1] + 1 

    cropped_strip = img_strip[y_start:y_end, :]
    if cropped_strip.size == 0:
        return np.zeros(target_len, dtype=np.float32), -1, -1
        
    row_means = np.mean(cropped_strip, axis=1)
    
    signal_reshaped = row_means.reshape(-1, 1).astype(np.float32)
    resized = cv2.resize(signal_reshaped, (1, target_len), interpolation=cv2.INTER_LINEAR)
    
    return resized.flatten(), y_start, y_end

# =============================================================================
# --- 3. PROCESSING AND SAVING ---
# =============================================================================

def process_and_save_all(img_cropped, mask_cropped, width_A, clahe, current_id, kernel_morph):
    
    # 1. IMAGE PROCESSING (for the signal and the debug image)
    l_channel = get_l_channel(img_cropped)
    cl1 = clahe.apply(l_channel)
    tophat = cv2.morphologyEx(cl1, cv2.MORPH_TOPHAT, kernel_morph)
    img_processed = cv2.add(cl1, tophat)
    
    # 2. Signal extraction
    h, w = img_processed.shape
    cx = w // 2 
    
    offset_start_px = int(width_A * LX_PARAMS['offset_start'])
    offset_end_px = int(width_A * LX_PARAMS['offset_end'])
    
    rx1 = min(w, max(0, cx + offset_start_px))
    rx2 = min(w, max(0, cx + offset_end_px))
    lx1 = min(w, max(0, cx - offset_end_px))
    lx2 = min(w, max(0, cx - offset_start_px))
    
    img_strip_left = img_processed[:, lx1:lx2]
    img_strip_right = img_processed[:, rx1:rx2]
    mask_strip_left = mask_cropped[:, lx1:lx2]
    mask_strip_right = mask_cropped[:, rx1:rx2]
    
    sig_l, y1_l, y2_l = get_trimmed_signal_and_coords(img_strip_left, mask_strip_left, LX_PARAMS['target_nodes'])
    sig_r, y1_r, y2_r = get_trimmed_signal_and_coords(img_strip_right, mask_strip_right, LX_PARAMS['target_nodes'])
        
    # take the longer one
    len_l = 0 if y1_l == -1 else (y2_l - y1_l)
    len_r = 0 if y1_r == -1 else (y2_r - y1_r)

    if len_l >= len_r:
        final_signal = sig_l
    else:
        final_signal = sig_r
    
    # --- 3. SAVING ---
    
    # A) save TXT
    os.makedirs(PATH_OUTPUT_TXT, exist_ok=True)
    with open(os.path.join(PATH_OUTPUT_TXT, f"{current_id}.txt"), "w") as f:
        f.write(",".join(map(str, final_signal)))
    
    final_signal_flip = final_signal[::-1]
    with open(os.path.join(PATH_OUTPUT_TXT, f"{current_id}_flip.txt"), "w") as f:
        f.write(",".join(map(str, final_signal_flip)))

    # B) save RAW IMAGES (raw, color, no drawings)
    os.makedirs(PATH_OUTPUT_RAW, exist_ok=True)
    cv2.imwrite(os.path.join(PATH_OUTPUT_RAW, f"{current_id}.jpg"), img_cropped)

    # C) save PROCESSED (processed + drawings)
    os.makedirs(PATH_OUTPUT_IMG, exist_ok=True)
    debug_img = cv2.cvtColor(img_processed, cv2.COLOR_GRAY2BGR)

    # draw the rectangles
    cv2.rectangle(debug_img, (lx1, 0), (lx2, h), (255, 0, 0), 1)
    if y1_l != -1:
        cv2.rectangle(debug_img, (lx1, y1_l), (lx2, y2_l), (0, 0, 255), 2)
    
    cv2.rectangle(debug_img, (rx1, 0), (rx2, h), (255, 0, 0), 1)
    if y1_r != -1:
        cv2.rectangle(debug_img, (rx1, y1_r), (rx2, y2_r), (0, 0, 255), 2)

    cv2.imwrite(os.path.join(PATH_OUTPUT_IMG, f"{current_id}.jpg"), debug_img)
        
    return True

# =============================================================================
# --- 4. MAIN LOOP ---
# =============================================================================

def main():
    files = sorted([f for f in os.listdir(INPUT_RAW_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    print(f"{len(files)} raw files will be processed.")
    print(f"Mode: SAVE ALL + RAW OUTPUT")
    print(f"Method: CLAHE + TopHat (brightness increase: +{PARAMS['brightness_beta']})")
    print(f"Outputs:\n  1. TXT: {PATH_OUTPUT_TXT}\n  2. IMG (Raw): {PATH_OUTPUT_RAW}\n  3. IMG (Proc): {PATH_OUTPUT_IMG}")
    
    clahe = cv2.createCLAHE(clipLimit=PARAMS['clahe_clip'], tileGridSize=PARAMS['clahe_grid'])
    kernel_morph = cv2.getStructuringElement(cv2.MORPH_RECT, PARAMS['morph_kernel_size'])
    
    GLOBAL_COUNTER = 1 
    
    for filename in tqdm(files, desc="Processing the data"):
        img_path = os.path.join(INPUT_RAW_FOLDER, filename)
        img_orig = cv2.imread(img_path)
        if img_orig is None: continue
        
        # 1. INCREASE THE BRIGHTNESS (for segmentation)
        bright_img = cv2.convertScaleAbs(img_orig, alpha=1.0, beta=PARAMS['brightness_beta'])

        # 2. FIND THE CONTOURS
        gray_coarse = cv2.cvtColor(bright_img, cv2.COLOR_BGR2GRAY)
        _, otsu_coarse = cv2.threshold(gray_coarse, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours_ext, _ = cv2.findContours(otsu_coarse, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        h, w = img_orig.shape[:2]
        valid_contours = shed_filter(contours_ext, w, h)
        
        # 3. PROCESS AND SAVE
        for cnt in valid_contours:
            img_cropped, mask_cropped, width_A = crop_and_rotate(img_orig, cnt)
            
            process_and_save_all(img_cropped, mask_cropped, width_A, clahe, GLOBAL_COUNTER, kernel_morph)
            
            GLOBAL_COUNTER += 1

    print("\n--- DONE ---")
    print(f"Total parts extracted: {GLOBAL_COUNTER - 1}")

if __name__ == "__main__":
    main()