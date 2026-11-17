import cv2
import numpy as np
import os
import time
import joblib
import warnings

PROJECT_ROOT = os.environ.get("GRIP_PROJECT_ROOT", os.path.abspath("project_root"))   # see README

# Silence unnecessary warnings
warnings.filterwarnings("ignore", category=UserWarning)

# =============================================================================
# --- 1. CONFIGURATION ---
# =============================================================================

INPUT_FOLDER = os.path.join(PROJECT_ROOT, r"datasets\main_datasets\main_products_mixed_controlledlighting_18_11_25\2lights_blackbackground")
OUTPUT_MAIN_FOLDER = os.path.join(PROJECT_ROOT, r"codes_results\gripareafinder\test_31")

# --- MODEL AND SCALER PATHS ---
MODEL_PATH = os.path.join(PROJECT_ROOT, r"codes_results\sidecheckermethodes\1d_adaptive_lightx\final_model\lgbm_model.pkl")
SCALER_PATH = os.path.join(PROJECT_ROOT, r"datasets\derivative_datasets\cropped_datasets\cropped_for_rf\READY_FOR_TRAIN\model\scaler.pkl")

CONFIG = {
    'brightness': False,           
    'alternatif_3': True,        
    'alternatif': False,
    'alternatif_2': False,
    'eat': False,                 
    'horizontal_kernel': True,    
    'rect_closing_after_invert': True, 
    
    'saveall': True,               
    'savefinal': True,
    
    'use_lightx_check': True
}

PARAMS = {
    'brightness_factor': 17,     
    'clahe_clip': 8.0,
    'clahe_grid': (8, 8),
    'morph_kernel': (51, 51),
    'horz_closing_len': 25,
    'rect_closing_size': (7, 7),
    'gauss_ksize': (15, 15),
    'gauss_sigma': 0,
    'min_area': 10000,
    'max_area': 7000000,
    'edge_tolerance': 25,
    'rect_ratio_limit': 1.325,
    'safety_crop_percent': 0.1,
    'eat_px': 0 
}

# LightX signal parameters
LX_PARAMS = {
    'target_nodes': 300,
    'adaptive_block': 35,
    'adaptive_c': 1,
    'offset_start': 0.10, # 10% of the cropped width
    'offset_end': 0.15    # 15% of the cropped width
}

# =============================================================================
# --- 2. HELPER FUNCTIONS ---
# =============================================================================

def adjust_brightness(img, factor):
    return cv2.convertScaleAbs(img, alpha=factor, beta=0)

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

def crop_and_rotate(img, contour, eat=False):
    """
    Crop with white padding
    RETURN: img_crop, mask_crop, width_A (width before trimming)
    """
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
    
    # Image padded with WHITE
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
    
    # --- keep the A value (width before trimming) ---
    h_c_raw, w_c_raw = img_crop.shape[:2]
    width_A = w_c_raw # reference A value
    
    # Safety crop (trimming)
    crop_margin = int(w_c_raw * PARAMS['safety_crop_percent'])
    if crop_margin > 0 and w_c_raw > (2 * crop_margin):
        img_crop = img_crop[:, crop_margin : w_c_raw - crop_margin]
        mask_crop = mask_crop[:, crop_margin : w_c_raw - crop_margin]
    
    return img_crop, mask_crop, width_A

def get_max_inscribed_circle_info(binary_mask):
    dist_transform = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
    _, max_val, _, max_loc = cv2.minMaxLoc(dist_transform)
    return max_loc, max_val

# =============================================================================
# --- 3. LIGHTX FUNCTIONS ---
# =============================================================================

def get_trimmed_signal(img_strip, mask_strip, target_len):
    """GET THE DATA INSIDE THE CONTOUR (TRIM)"""
    row_has_data = np.max(mask_strip, axis=1) > 0
    if not np.any(row_has_data):
        return np.zeros(target_len, dtype=np.float32)

    indices = np.where(row_has_data)[0]
    y_start, y_end = indices[0], indices[-1] + 1 

    cropped_strip = img_strip[y_start:y_end, :]
    if cropped_strip.size == 0:
        return np.zeros(target_len, dtype=np.float32)
        
    row_means = np.mean(cropped_strip, axis=1)
    signal_reshaped = row_means.reshape(-1, 1).astype(np.float32)
    resized = cv2.resize(signal_reshaped, (1, target_len), interpolation=cv2.INTER_LINEAR)
    
    return resized.flatten()

def check_part_with_lightx(img_cropped, mask_cropped, width_A, model, scaler, clahe):
    """
    Uses the width (w) of img_cropped as the reference
    (not width_A, the width before trimming).
    """
    l_channel = get_l_channel(img_cropped)
    cl1 = clahe.apply(l_channel)
    th_adaptive = cv2.adaptiveThreshold(cl1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY, LX_PARAMS['adaptive_block'], LX_PARAMS['adaptive_c'])
    
    # Size of the current (trimmed) image
    h, w = th_adaptive.shape # w: cropped width
    cx = w // 2 
    
    # --- offset from the cropped width (w) ---
    offset_start_px = int(w * LX_PARAMS['offset_start'])
    offset_end_px = int(w * LX_PARAMS['offset_end'])
    
    # Shift the coordinates relative to the center
    rx1 = cx + offset_start_px
    rx2 = cx + offset_end_px
    lx1 = cx - offset_end_px
    lx2 = cx - offset_start_px
    
    # Make sure the coordinates do not go outside the image (safety)
    rx1 = min(w, max(0, rx1))
    rx2 = min(w, max(0, rx2))
    lx1 = min(w, max(0, lx1))
    lx2 = min(w, max(0, lx2))
    
    # Cut the strips
    img_strip_left = th_adaptive[:, lx1:lx2]
    img_strip_right = th_adaptive[:, rx1:rx2]
    mask_strip_left = mask_cropped[:, lx1:lx2]
    mask_strip_right = mask_cropped[:, rx1:rx2]
    
    # Extract the signals
    sig_l = get_trimmed_signal(img_strip_left, mask_strip_left, LX_PARAMS['target_nodes'])
    sig_r = get_trimmed_signal(img_strip_right, mask_strip_right, LX_PARAMS['target_nodes'])
        
    final_signal_raw = (sig_l + sig_r) / 2.0
    final_signal_reshaped = final_signal_raw.reshape(1, -1)
    
    # Scaler & Predict
    final_signal_scaled = scaler.transform(final_signal_reshaped)
    prediction = model.predict(final_signal_scaled)[0]
    
    return int(prediction) == 1

# =============================================================================
# --- 4. MAIN PIPELINE ---
# =============================================================================

def process_pipeline():
    lx_model = None
    lx_scaler = None
    
    if CONFIG['use_lightx_check']:
        print(">>> Loading the LightX model and scaler...")
        try:
            lx_model = joblib.load(MODEL_PATH)
            lx_scaler = joblib.load(SCALER_PATH)
            print(">>> Model and scaler loaded SUCCESSFULLY.")
        except Exception as e:
            print(f"!!! LOADING ERROR: {e}")
            CONFIG['use_lightx_check'] = False

    if not os.path.exists(OUTPUT_MAIN_FOLDER): os.makedirs(OUTPUT_MAIN_FOLDER)
    
    files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Number of files to process: {len(files)}")
    print("-" * 50)

    clahe = cv2.createCLAHE(clipLimit=PARAMS['clahe_clip'], tileGridSize=PARAMS['clahe_grid'])
    kernel_morph = cv2.getStructuringElement(cv2.MORPH_RECT, PARAMS['morph_kernel'])
    kernel_horz = cv2.getStructuringElement(cv2.MORPH_RECT, (PARAMS['horz_closing_len'], 1))
    kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, PARAMS['rect_closing_size'])

    for filename in files:
        print(f"\n>>> Processing file: {filename}")
        img_path = os.path.join(INPUT_FOLDER, filename)
        img_orig = cv2.imread(img_path)
        if img_orig is None: continue
        
        base_name = os.path.splitext(filename)[0]
        t_start_global = time.time()

        img_global_vis = None
        if CONFIG['savefinal']:
            img_global_vis = img_orig.copy()

        if CONFIG['brightness']:
            img_bright = adjust_brightness(img_orig, PARAMS['brightness_factor'])
        else:
            img_bright = img_orig

        gray_coarse = cv2.cvtColor(img_bright, cv2.COLOR_BGR2GRAY)
        _, otsu_coarse = cv2.threshold(gray_coarse, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        contours_ext, _ = cv2.findContours(otsu_coarse, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = img_orig.shape[:2]
        valid_contours = shed_filter(contours_ext, w, h)

        if not valid_contours:
            print("! WARNING: No part found.")
            continue
        
        print(f"  >>> {len(valid_contours)} parts found.")

        for i, main_contour in enumerate(valid_contours):
            print(f"    --- Part {i+1} ---")
            t_part_start = time.time()
            part_suffix = f"_part_{i}"
            
            t5 = time.time()
            # width_A is read but not used
            img_cropped, mask_cropped, width_A = crop_and_rotate(img_orig, main_contour, eat=CONFIG['eat'])
            print(f"      [Time] Crop & Rotate: {(time.time() - t5)*1000:.2f} ms")

            # --- LIGHTX CHECK ---
            if CONFIG['use_lightx_check'] and lx_model is not None and lx_scaler is not None:
                t_lx = time.time()
                # width_A is passed to the function, but the function uses w
                is_valid_part = check_part_with_lightx(img_cropped, mask_cropped, width_A, lx_model, lx_scaler, clahe)
                lx_time = (time.time() - t_lx) * 1000
                
                if not is_valid_part:
                    print(f"      [LightX]  BOTTOM CLASS. Skipping. ({lx_time:.2f} ms)")
                    if CONFIG['savefinal'] and img_global_vis is not None:
                         rect_g = cv2.minAreaRect(main_contour)
                         box = cv2.boxPoints(rect_g)
                         box = np.int32(box) 
                         cv2.drawContours(img_global_vis, [box], 0, (0, 0, 255), 2)
                    continue 
                else:
                    print(f"      [LightX]  TOP CLASS. Continuing... ({lx_time:.2f} ms)")

            # --- ENHANCE & MIC (only if it is the top class) ---
            t6 = time.time()
            l_channel = get_l_channel(img_cropped)
            l_clahe = clahe.apply(l_channel)
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

            t_cnt = time.time()
            contours_final, _ = cv2.findContours(binary_final, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            result_vis = img_cropped.copy()
            
            if contours_final:
                largest_cnt = max(contours_final, key=cv2.contourArea)
                mask_for_mic = np.zeros(binary_final.shape, dtype=np.uint8)
                cv2.drawContours(mask_for_mic, [largest_cnt], -1, 255, -1)
                
                (cx_local, cy_local), radius = get_max_inscribed_circle_info(mask_for_mic)
                
                rect_global = cv2.minAreaRect(main_contour)
                center_g, size_g, angle_g = rect_global
                if size_g[0] < size_g[1]: angle_g += 90
                
                h_c, w_c = img_cropped.shape[:2]
                vec_x = cx_local - (w_c / 2.0)
                vec_y = cy_local - (h_c / 2.0)
                rad = np.radians(angle_g)
                x_global = center_g[0] + (vec_x * np.cos(rad) - vec_y * np.sin(rad))
                y_global = center_g[1] + (vec_x * np.sin(rad) + vec_y * np.cos(rad))
                
                print(f"      >> MIC: R={radius:.2f} | Global: ({x_global:.2f}, {y_global:.2f})")
                
                cv2.drawContours(result_vis, [largest_cnt], -1, (255,0,0), 1)
                cv2.circle(result_vis, (int(cx_local), int(cy_local)), int(radius), (0,0,255), 2)

                if CONFIG['savefinal'] and img_global_vis is not None:
                    cv2.circle(img_global_vis, (int(x_global), int(y_global)), int(radius), (0, 0, 255), 2)
                    cv2.circle(img_global_vis, (int(x_global), int(y_global)), 3, (0, 255, 0), -1)
                    cv2.putText(img_global_vis, f"R{int(radius)}", (int(x_global)+10, int(y_global)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            print(f"      [Time] Calc & Vis: {(time.time() - t_cnt)*1000:.2f} ms")
            print(f"    --- End of part: {(time.time() - t_part_start)*1000:.2f} ms ---")

            if CONFIG['saveall']:
                cv2.imwrite(os.path.join(OUTPUT_MAIN_FOLDER, f"{base_name}{part_suffix}_5_BinaryFinal.jpg"), binary_final)
                cv2.imwrite(os.path.join(OUTPUT_MAIN_FOLDER, f"RESULT_PART_{base_name}{part_suffix}.jpg"), result_vis)

        if CONFIG['savefinal'] and img_global_vis is not None:
            t_save_global = time.time()
            cv2.imwrite(os.path.join(OUTPUT_MAIN_FOLDER, f"FULL_RESULT_{base_name}.jpg"), img_global_vis)
            print(f"  [Time] Global Image Save: {(time.time() - t_save_global)*1000:.2f} ms")
        
        print(f"--> FILE TOTAL: {(time.time() - t_start_global)*1000:.2f} ms")

if __name__ == "__main__":
    process_pipeline()