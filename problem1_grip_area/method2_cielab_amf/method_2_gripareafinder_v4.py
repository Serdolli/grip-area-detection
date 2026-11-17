import cv2
import numpy as np
import os
import time
from skimage.filters import threshold_multiotsu

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
RL_OUTPUT    = os.path.join(PROJECT_ROOT, r"codes_results\conforance_codes\use_ready_codes\output\method_2")
os.makedirs(RL_INPUT, exist_ok=True); os.makedirs(RL_OUTPUT, exist_ok=True)
print("=" * 100)
print(" REAL-LIFE MODE | Method 2 - CIELAB-AMF (gripareafinder_v4)")
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

# =============================================================================
# --- 1. CONFIGURATION ---
# =============================================================================

INPUT_FOLDER = os.path.join(PROJECT_ROOT, r"codes_results\conforance_codes\use_ready_codes\input")
OUTPUT_MAIN_FOLDER = os.path.join(PROJECT_ROOT, r"codes_results\conforance_codes\use_ready_codes\output\method_2")
CONFIG = {
    # 'brightness': False,  (set with PARAMS['brightness_beta'] instead)
    
    'alternatif_3': True,       
    'alternatif': False,
    'alternatif_2': False,
    
    'eat': False,               
    
    'horizontal_kernel': True,  
    'rect_closing_after_invert': True, 
    
    # --- SAVING ENABLED ---
    'saveall': True,             
    'savefinal': True            
}

PARAMS = {
    # --- BRIGHTNESS SETTING ---
    'brightness_beta': 220,  # Brightness increase used only while finding contours
    
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

def adjust_brightness(img, factor):
    return cv2.convertScaleAbs(img, alpha=1.0, beta=factor)

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
    x1 = max(0, rx)
    y1 = max(0, ry)
    x2 = min(w_img, rx + rw)
    y2 = min(h_img, ry + rh)
    img_roi = img[y1:y2, x1:x2]
    mask_roi = np.zeros((y2-y1, x2-x1), dtype=np.uint8)
    contour_roi = contour - np.array([x1, y1])
    cv2.drawContours(mask_roi, [contour_roi], -1, 255, -1)
    center_roi = (center[0] - x1, center[1] - y1)
    M = cv2.getRotationMatrix2D(center_roi, angle, 1.0)
    img_rot = cv2.warpAffine(img_roi, M, (x2-x1, y2-y1), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    mask_rot = cv2.warpAffine(mask_roi, M, (x2-x1, y2-y1), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    size_int = tuple(map(int, size))
    img_crop = cv2.getRectSubPix(img_rot, size_int, center_roi)
    mask_crop = cv2.getRectSubPix(mask_rot, size_int, center_roi)
    h_c, w_c = img_crop.shape[:2]
    crop_margin = int(w_c * PARAMS['safety_crop_percent'])
    if crop_margin > 0 and w_c > (2 * crop_margin):
        img_crop = img_crop[:, crop_margin : w_c - crop_margin]
        mask_crop = mask_crop[:, crop_margin : w_c - crop_margin]
    return img_crop, mask_crop

def get_max_inscribed_circle_info(binary_mask):
    dist_transform = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
    _, max_val, _, max_loc = cv2.minMaxLoc(dist_transform)
    return max_loc, max_val

# =============================================================================
# --- 3. MAIN PIPELINE ---
# =============================================================================

def process_pipeline():
    if not os.path.exists(OUTPUT_MAIN_FOLDER): os.makedirs(OUTPUT_MAIN_FOLDER)
    
    files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg','.bmp'))]
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
        img_orig = rl_undistort(img_orig)   # [REALLIFE] step 1: undistort first
        
        base_name = os.path.splitext(filename)[0]
        t_start_global = time.time()

        # [Global Vis Init]
        img_global_vis = None
        if CONFIG['savefinal']:
            t_mem = time.time()
            img_global_vis = img_orig.copy()

        # =================================================================
        # --- 1. DETECTION STAGE ---
        # =================================================================
        t1 = time.time()
        
        # A. INCREASE BRIGHTNESS (temporary copy, only for finding contours)
        # alpha=1.0 (contrast unchanged), beta=PARAMS['brightness_beta'] (add brightness)
        bright_img = cv2.convertScaleAbs(img_orig, alpha=1.0, beta=PARAMS['brightness_beta'])
        
        # B. COARSE OTSU & CONTOUR (on the brightened image)
        gray_coarse = cv2.cvtColor(bright_img, cv2.COLOR_BGR2GRAY)
        _, otsu_coarse = cv2.threshold(gray_coarse, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours_ext, _ = cv2.findContours(otsu_coarse, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        print(f"  [Time] Detection (Bright+Otsu+Cnt): {(time.time() - t1)*1000:.2f} ms")

        # C. FILTERING
        t4 = time.time()
        h, w = img_orig.shape[:2]
        valid_contours = shed_filter(contours_ext, w, h)
        print(f"  [Time] Shed Filter: {(time.time() - t4)*1000:.2f} ms")

        if not valid_contours:
            print("! WARNING: No part found.")
            continue
        
        print(f"  >>> {len(valid_contours)} parts found.")

        # =================================================================
        # --- 2. EXTRACTION AND ANALYSIS STAGE ---
        # =================================================================
        for i, main_contour in enumerate(valid_contours):
            print(f"    --- Part {i+1} ---")
            t_part_start = time.time()
            part_suffix = f"_part_{i}"
            
            # A. CROP & ROTATE (uses the original image 'img_orig')
            t5 = time.time()
            img_cropped, mask_cropped = crop_and_rotate(img_orig, main_contour, eat=CONFIG['eat'])
            print(f"      [Time] Crop & Rotate: {(time.time() - t5)*1000:.2f} ms")
            
            # B. ENHANCE (continue on the original data)
            t6 = time.time()
            l_channel = get_l_channel(img_cropped)
            l_clahe = clahe.apply(l_channel)
            tophat_t1 = cv2.morphologyEx(l_clahe, cv2.MORPH_TOPHAT, kernel_morph)
            enhanced = cv2.add(l_clahe, tophat_t1)
            blurred = cv2.GaussianBlur(enhanced, PARAMS['gauss_ksize'], PARAMS['gauss_sigma'])
            print(f"      [Time] Enhance: {(time.time() - t6)*1000:.2f} ms")

            # C. THRESHOLD & MIC
            t7 = time.time()
            _, binary_final = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            binary_final = cv2.bitwise_and(binary_final, binary_final, mask=mask_cropped)
            if CONFIG['horizontal_kernel']:
                binary_final = cv2.morphologyEx(binary_final, cv2.MORPH_CLOSE, kernel_horz)
            binary_final = cv2.bitwise_not(binary_final)
            if CONFIG['rect_closing_after_invert']:
                binary_final = cv2.morphologyEx(binary_final, cv2.MORPH_CLOSE, kernel_rect)
            binary_final = cv2.bitwise_and(binary_final, binary_final, mask=mask_cropped)
            print(f"      [Time] Threshold & Morph: {(time.time() - t7)*1000:.2f} ms")

            # D. MIC CALCULATION
            t_cnt = time.time()
            contours_final, _ = cv2.findContours(binary_final, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            result_vis = img_cropped.copy() 
            
            if contours_final:
                largest_cnt = max(contours_final, key=cv2.contourArea)
                mask_for_mic = np.zeros(binary_final.shape, dtype=np.uint8)
                cv2.drawContours(mask_for_mic, [largest_cnt], -1, 255, -1)
                
                (cx_local, cy_local), radius = get_max_inscribed_circle_info(mask_for_mic)
                
                # Global coordinates
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
                rl_report(filename, i + 1, x_global, y_global, radius)   # [REALLIFE] step 3: MIC centre -> robot mm
                
                # Drawing
                cv2.drawContours(result_vis, [largest_cnt], -1, (255,0,0), 1)
                cv2.circle(result_vis, (int(cx_local), int(cy_local)), int(radius), (0,0,255), 2)

                if CONFIG['savefinal'] and img_global_vis is not None:
                    cv2.circle(img_global_vis, (int(x_global), int(y_global)), int(radius), (0, 0, 255), 2)
                    cv2.circle(img_global_vis, (int(x_global), int(y_global)), 3, (0, 255, 0), -1)
                    cv2.putText(img_global_vis, f"R{int(radius)}", (int(x_global)+10, int(y_global)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            print(f"      [Time] Calc & Vis: {(time.time() - t_cnt)*1000:.2f} ms")
            print(f"    --- End of part: {(time.time() - t_part_start)*1000:.2f} ms ---")

            # Intermediate save
            if CONFIG['saveall']:
                cv2.imwrite(os.path.join(OUTPUT_MAIN_FOLDER, f"{base_name}{part_suffix}_5_BinaryFinal.jpg"), binary_final)
                cv2.imwrite(os.path.join(OUTPUT_MAIN_FOLDER, f"RESULT_PART_{base_name}{part_suffix}.jpg"), result_vis)

        # SAVE THE GLOBAL IMAGE
        if CONFIG['savefinal'] and img_global_vis is not None:
            t_save_global = time.time()
            cv2.imwrite(os.path.join(OUTPUT_MAIN_FOLDER, f"FULL_RESULT_{base_name}.jpg"), img_global_vis)
            print(f"  [Time] Global Image Save: {(time.time() - t_save_global)*1000:.2f} ms")
        
        print(f"--> FILE TOTAL: {(time.time() - t_start_global)*1000:.2f} ms")
        rl_save_overlay(img_orig, filename)   # [REALLIFE]

if __name__ == "__main__":
    process_pipeline()