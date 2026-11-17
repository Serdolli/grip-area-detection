#Librarys
#______________________________________________________________________________
import os
import time
import numpy as np
import cv2
import argparse

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
RL_OUTPUT    = os.path.join(PROJECT_ROOT, r"codes_results\conforance_codes\use_ready_codes\output\method_1")
os.makedirs(RL_INPUT, exist_ok=True); os.makedirs(RL_OUTPUT, exist_ok=True)
print("=" * 100)
print(" REAL-LIFE MODE | Method 1 - MAT-GOF (sidechecker_debugged_v3)")
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
#______________________________________________________________________________

#Folder paths
#______________________________________________________________________________
INPUT_FOLDER              = os.path.join(PROJECT_ROOT, r"codes_results\conforance_codes\use_ready_codes\input")
CONTOUR_OUTPUT_FOLDER     = os.path.join(PROJECT_ROOT, r"codes_results\conforance_codes\use_ready_codes\output\method_1\contourdetection_output")
CONTOUR_COORDS_TXT        = os.path.join(os.path.join(PROJECT_ROOT, r"codes_results\conforance_codes\use_ready_codes\output\method_1\contourcoordinates"), "all_coords.txt")
ADAPTIVE_THRESHOLD_OUTPUT = os.path.join(PROJECT_ROOT, r"codes_results\conforance_codes\use_ready_codes\output\method_1\test_adapt_thresh_output")
MASKED_RESULTS_FOLDER     = os.path.join(PROJECT_ROOT, r"codes_results\conforance_codes\use_ready_codes\output\method_1\masked_results")
OUTPUT_FOLDER             = os.path.join(PROJECT_ROOT, r"codes_results\conforance_codes\use_ready_codes\output\method_1\output")
UNDISTORTED_FOLDER        = os.path.join(PROJECT_ROOT, r"codes_results\conforance_codes\use_ready_codes\output\method_1\undistorted")
CALIB_NPZ                 = os.path.join(PROJECT_ROOT, r"codes_results\grippointfinder\unified_point_finder algorithm\unifier_3\camera_calibration\calibration_outputs\Camera_Calibration_Params.pkl")
H_NPY_PATH                = os.path.join(PROJECT_ROOT, r"codes_results\conforance_codes\use_ready_codes\needed_files\H_img2world_cm.npy")
H                         = np.load(H_NPY_PATH)
#______________________________________________________________________________

#Settings
#______________________________________________________________________________
#Saves halfputs when True
SAVE_INTERMEDIATE          = True
SAVE_FINAL                 = True
#Uses fast png method when true
USE_FAST_PNG               = True
#Compresses png files without loss
PNG_COMPRESSION            = 1
#Uses jpg instead of png when true
USE_JPEG                   = False
#Compresses jpeg files in percent
JPEG_QUALITY               = 80

STEP1_ALTERNATIVE          = True
#______________________________________________________________________________
#Parameters for proccessing
#______________________________________________________________________________
# Contour shedding parameters
MIN_CONTOUR_AREA_PX        = 15000  # exclude contours smaller than this
MAX_CONTOUR_AREA_PX        = 500000# exclude contours larger than this
EDGE_TOUCH_TOLERANCE       = 1     # exclude contours that are near x px to edge         
RECT_AREA_RATIO_LIMIT      = 1.325 # exclude contours that are too different than a rectangle

COLOR                      = 15
MASK_BEFORE_TRESHOLD_COLOR = (COLOR, COLOR, COLOR) # The RGB(x,x,x) value of the first mask

CONTRAST                   = 1  # < 1.0 lowers contrast, > 1.0 increases
BRIGHTNESS                 = 15   # positive increases brightness, negative darkens

CONTRAST_2                 = 1.4   #second contrast value for Adaptive Treshold
BRIGHTNESS_2               = 1.5   #second brightness value for Adaptive Treshold

ADAPTIVE_BLOCKSIZE         = 23    #Blocksize parameter of Adaptive Treshold
ADAPTIVE_C                 = 0.4     #C value for Adaptive Treshold

MED_K                      = 13    # median blur kernel size (odd)
OPEN_K                     = 1     # morphology open kernel size
OPEN_ITERS                 = 0     # open iterations
DILATE_ITERS               = 2     # dilate iterations after open
MIN_SPECKLE_AREA           = 3     # remove connected components smaller than this
MIN_MAIN_AREA              = 25    # keep contours >= this area
MERGE_GAP_PX               = 10     # ~max gap to bridge when merging nearby contours


MIN_RECT_AREA_PX           = 400   # minimum area of candidate rectangle
MAX_RECT_AREA_PX           = 40000 # maximum area of candidate rectangle
ANGLE_TOLERANCE_DEG        = 7.5     # allowed deviation from parent angle (+/-)
MIN_ASPECT_RATIO           = 10.0  # expected long/short ratio

# Step 1 Alternative (Hybrid LAB + BG Sub + Adaptive + Canny)
KERNEL_BG_MIN_1             = 31
KERNEL_BG_SCALE_1           = 7   # the smaller the number, the stronger the blur

CLAHE_CLIP_LIMIT_1          = 1
CLAHE_TILE_GRID_1           = (3, 3)

ADAPTIVE_BLOCKSIZE_LARGE_1  = 51   # for large images
ADAPTIVE_BLOCKSIZE_SMALL_1  = 35   # for small images
ADAPTIVE_BLOCKSIZE_SWITCH_1 = 300  # use LARGE if min(h, w) is larger than this
ADAPTIVE_C_1                = -2  # more negative -> more white area

CANNY_LOW_1                 = 5
CANNY_HIGH_1                = 15
CANNY_BLUR_KSIZE_1          = 7    # (7,7)

MORPH_KERNEL_SIZE_1         = 7
MORPH_CLOSE_ITER_1          = 0
MORPH_OPEN_ITER_1           = 1

AREA_EDGE_MASK_MIN_1        = 10000
AREA_FINAL_MASK_MIN_1       = 12800

# Trim the shadow protrusion
SPIKE_KSIZE_1               = 3    # 3x3 erode
SPIKE_ERODE_ITERS_1         = 7


 
#______________________________________________________________________________
# === CLI overrides ===
def _odd(v):
    v = int(round(float(v)))
    return v if v % 2 == 1 else v + 1

def _tuple3(s):
    parts = [p.strip() for p in str(s).split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("r,g,b expected")
    return tuple(int(float(x)) for x in parts)

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--COLOR", type=float)
parser.add_argument("--MASK_BEFORE_TRESHOLD_COLOR", type=_tuple3)
parser.add_argument("--CONTRAST_2", type=float)
parser.add_argument("--BRIGHTNESS_2", type=float)
parser.add_argument("--ADAPTIVE_BLOCKSIZE", type=float)
parser.add_argument("--ADAPTIVE_C", type=float)
parser.add_argument("--MED_K", type=float)
parser.add_argument("--OPEN_K", type=float)
parser.add_argument("--OPEN_ITERS", type=int)
parser.add_argument("--DILATE_ITERS", type=int)
parser.add_argument("--MIN_SPECKLE_AREA", type=int)
parser.add_argument("--MIN_MAIN_AREA", type=int)
parser.add_argument("--MERGE_GAP_PX", type=int)
args, _ = parser.parse_known_args()

g = globals()
if args.COLOR is not None:
    g["COLOR"] = int(round(args.COLOR))
    g["MASK_BEFORE_TRESHOLD_COLOR"] = (g["COLOR"], g["COLOR"], g["COLOR"])
if args.MASK_BEFORE_TRESHOLD_COLOR is not None:
    g["MASK_BEFORE_TRESHOLD_COLOR"] = args.MASK_BEFORE_TRESHOLD_COLOR

if args.CONTRAST_2 is not None:      g["CONTRAST_2"] = float(args.CONTRAST_2)
if args.BRIGHTNESS_2 is not None:    g["BRIGHTNESS_2"] = float(args.BRIGHTNESS_2)
if args.ADAPTIVE_BLOCKSIZE is not None: g["ADAPTIVE_BLOCKSIZE"] = _odd(args.ADAPTIVE_BLOCKSIZE)
if args.ADAPTIVE_C is not None:      g["ADAPTIVE_C"] = float(args.ADAPTIVE_C)
if args.MED_K is not None:           g["MED_K"] = _odd(args.MED_K)
if args.OPEN_K is not None:          g["OPEN_K"] = _odd(args.OPEN_K)
if args.OPEN_ITERS is not None:      g["OPEN_ITERS"] = int(args.OPEN_ITERS)
if args.DILATE_ITERS is not None:    g["DILATE_ITERS"] = int(args.DILATE_ITERS)
if args.MIN_SPECKLE_AREA is not None:g["MIN_SPECKLE_AREA"] = int(args.MIN_SPECKLE_AREA)
if args.MIN_MAIN_AREA is not None:   g["MIN_MAIN_AREA"] = int(args.MIN_MAIN_AREA)
if args.MERGE_GAP_PX is not None:    g["MERGE_GAP_PX"] = int(args.MERGE_GAP_PX)
g["ADAPTIVE_BLOCKSIZE"] = _odd(g["ADAPTIVE_BLOCKSIZE"])
g["MED_K"] = _odd(g["MED_K"])
g["OPEN_K"] = _odd(g["OPEN_K"])
if g["OPEN_K"] <= 1:
    g["OPEN_ITERS"] = 0
# ______________________________________________________________________________

#Step -3 Makes code not crash when folders already exist
#______________________________________________________________________________
os.makedirs(CONTOUR_OUTPUT_FOLDER, exist_ok=True)
os.makedirs(ADAPTIVE_THRESHOLD_OUTPUT, exist_ok=True)
os.makedirs(MASKED_RESULTS_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(UNDISTORTED_FOLDER, exist_ok=True)
#______________________________________________________________________________

#Step -2 Renames
#______________________________________________________________________________
def renamer(path, new_ext):
    base, _ = os.path.splitext(path)
    return base + new_ext
#______________________________________________________________________________

#Step -1 Changes to jpg or png
#______________________________________________________________________________
def changer(path, img):
    if USE_JPEG:
        path = renamer(path, ".jpg")
        cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    else:
        if path.lower().endswith(".png") and USE_FAST_PNG:
            cv2.imwrite(path, img, [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION])
        else:
            cv2.imwrite(path, img)
#______________________________________________________________________________

#Step 0 Undistorts the image
#_______________________________________________________________________________
def undistorter(INPUT_FOLDER, CALIB_NPZ, SAVE_INTERMEDIATE, UNDISTORTED_FOLDER):
    K = RL_MTX; dist = RL_DIST   # [REALLIFE] step 1: Method 1 keeps its own undistortion, but with the camera calibration of unifier_3
    if SAVE_INTERMEDIATE:
        os.makedirs(UNDISTORTED_FOLDER, exist_ok=True)
    files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith((".png", ".jpg", ".bmp"))]   # [REALLIFE] the Cognex saves .bmp
    files.sort()
    if not files:
        print("No images found.")
        return
    prev_size = None
    prev_newK = None
    for fname in files:
        input_path = os.path.join(INPUT_FOLDER, fname)
        img = cv2.imread(input_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f" Skipping unreadable file: {fname}")
            continue
        h, w = img.shape[:2]
        size_key = (w, h)
        if size_key != prev_size:
            prev_newK, _ = cv2.getOptimalNewCameraMatrix(K, dist, size_key, alpha=0)
            prev_size = size_key
        undistorted = cv2.undistort(img, K, dist, None, prev_newK)
        if SAVE_INTERMEDIATE:
            output_path = os.path.join(UNDISTORTED_FOLDER, f"{os.path.splitext(fname)[0]}_undistorted.png")
            cv2.imwrite(output_path, undistorted)
        yield fname, undistorted
#_______________________________________________________________________________

#Step 1 Adjusts contrast and brightness using global parameters,performs Otsu thresholding, detects contours,and appends all contour coordinates to the global text file.
#_______________________________________________________________________________
def otsu_treshold_filtering(fname, img_bgr, save=False, out_dir=None):
    # Adjusts brightness and contrast
    x = img_bgr.astype(np.float32) / 255.0
    y = (x - 0.5) * CONTRAST + 0.5
    y = y * BRIGHTNESS
    y = np.clip(y, 0.0, 1.0)
    adjusted = (y * 255.0).astype(np.uint8)
    # Converts to grayscale
    gray = cv2.cvtColor(adjusted, cv2.COLOR_BGR2GRAY)
    #Applys Otsu thresholding
    _, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+ cv2.THRESH_OTSU)
    #Finds external contours
    contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Draws contours for visualization
    vis = adjusted.copy()
    cv2.drawContours(vis, contours, -1, (0, 255, 0), 3)
    #Optionally saves images
    if save and out_dir:
        os.makedirs(out_dir, exist_ok=True)
        base = os.path.splitext(fname)[0]
        changer(os.path.join(out_dir, f"{base}_adjusted.png"), adjusted)
        changer(os.path.join(out_dir, f"{base}_contours.png"), vis)
    #Returns outputs for further processing
    return contours, thr, vis, adjusted
#________________________________________________________________________________

def step1_alternative_hybrid(fname, img_bgr, save=False, out_dir=None):
  
    h, w = img_bgr.shape[:2]

    # --- 1) LAB + CLAHE ---
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT_1,
        tileGridSize=CLAHE_TILE_GRID_1
    )
    L_eq = clahe.apply(L)
    lab_eq = cv2.merge((L_eq, a, b))
    adjusted = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    # --- 2) Gray + background subtraction ---
    gray = cv2.cvtColor(adjusted, cv2.COLOR_BGR2GRAY)

    dyn_ksize = int(max(KERNEL_BG_MIN_1, min(h, w) / KERNEL_BG_SCALE_1))
    if dyn_ksize % 2 == 0:
        dyn_ksize += 1

    bg = cv2.GaussianBlur(gray, (dyn_ksize, dyn_ksize), 0)

    diff = cv2.subtract(gray, bg)
    diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    diff = diff.astype(np.uint8)

    # --- 3) adaptive threshold on diff -> mask1 ---
    if min(h, w) > ADAPTIVE_BLOCKSIZE_SWITCH_1:
        blockSize = ADAPTIVE_BLOCKSIZE_LARGE_1
    else:
        blockSize = ADAPTIVE_BLOCKSIZE_SMALL_1
    if blockSize % 2 == 0:
        blockSize += 1

    mask1 = cv2.adaptiveThreshold(
        diff,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize,
        ADAPTIVE_C_1
    )

    # --- 4) Canny + contour fill -> mask2 ---
    blur_gray = cv2.GaussianBlur(
        gray,
        (CANNY_BLUR_KSIZE_1, CANNY_BLUR_KSIZE_1),
        0
    )
    edges = cv2.Canny(blur_gray, CANNY_LOW_1, CANNY_HIGH_1)

    morph_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (MORPH_KERNEL_SIZE_1, MORPH_KERNEL_SIZE_1)
    )
    edges_dil = cv2.dilate(edges, morph_kernel, iterations=1)

    contours_edge, _ = cv2.findContours(edges_dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    edge_mask = np.zeros_like(gray)
    for cnt in contours_edge:
        if cv2.contourArea(cnt) < AREA_EDGE_MASK_MIN_1:
            continue
        cv2.drawContours(edge_mask, [cnt], -1, 255, thickness=cv2.FILLED)

    # --- 5) Merge the masks ---
    combined = cv2.bitwise_or(mask1, edge_mask)

    # --- 6) Clean up with morphology ---
    combined = cv2.morphologyEx(
        combined,
        cv2.MORPH_CLOSE,
        morph_kernel,
        iterations=MORPH_CLOSE_ITER_1
    )
    combined = cv2.morphologyEx(
        combined,
        cv2.MORPH_OPEN,
        morph_kernel,
        iterations=MORPH_OPEN_ITER_1
    )

    # --- 7) Area filter (first contour set) ---
    contours_all, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_mask = np.zeros_like(combined)
    for cnt in contours_all:
        if cv2.contourArea(cnt) < AREA_FINAL_MASK_MIN_1:
            continue
        cv2.drawContours(final_mask, [cnt], -1, 255, thickness=cv2.FILLED)

    # ---trimming
    spike_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (SPIKE_KSIZE_1, SPIKE_KSIZE_1)
    )
    final_mask = cv2.erode(final_mask, spike_kernel, iterations=SPIKE_ERODE_ITERS_1)

    # contours after trimming
    contours_final, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean_contours = [c for c in contours_final if cv2.contourArea(c) >= AREA_FINAL_MASK_MIN_1]

    # --- 8) Visualization ---
    vis = img_bgr.copy()
    if clean_contours:
        cv2.drawContours(vis, clean_contours, -1, (0, 255, 0), 3)

    # --- 9) Save if requested ---
    if save and out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        base = os.path.splitext(fname)[0]
        changer(os.path.join(out_dir, f"{base}_step1hyb_mask.png"), final_mask)
        changer(os.path.join(out_dir, f"{base}_step1hyb_contours.png"), vis)
        changer(os.path.join(out_dir, f"{base}_step1hyb_adjusted.png"), adjusted)

    # pipeline format:
    # contours, thr, vis, adjusted
    return clean_contours, final_mask, vis, adjusted


#Step 2 Filters the contours and creates 2 classes(kept and rejected)
#____________________________________________________________________________________
def shed_filter(contours, image_shape):
    kept, rejected, kept_info,trash = [], [], [],[]
    h, w = image_shape[:2]
    for c in contours:
        # --- Contour area ---
        A = cv2.contourArea(c)
        if not (MIN_CONTOUR_AREA_PX <= A <= MAX_CONTOUR_AREA_PX):
            trash.append((c, "area", A))
            continue
        # --- Edge touch check (fast min/max + tolerance) ---
        pts = c.reshape(-1, 2)
        minx, miny = pts.min(axis=0)
        maxx, maxy = pts.max(axis=0)
        touches_edge = (
            minx <= EDGE_TOUCH_TOLERANCE or
            maxx >= w - 1 - EDGE_TOUCH_TOLERANCE or
            miny <= EDGE_TOUCH_TOLERANCE or
            maxy >= h - 1 - EDGE_TOUCH_TOLERANCE)
        if touches_edge:
            rejected.append((c, "edge", A))
            continue
        # --- minAreaRect, rectangle area, and ratio check ---
        rect = cv2.minAreaRect(c)                 # ((cx,cy), (rw,rh), angle_raw)
        (_, _), (rw, rh), angle_raw = rect
        B = (rw * rh) if (rw > 0 and rh > 0) else 0.0
        if B <= 0.0:
            rejected.append((c, "invalid_rect", A))
            continue
        ratio = B / A
        if ratio > RECT_AREA_RATIO_LIMIT:
            rejected.append((c, "rect_ratio", ratio))
            continue
        # --- Stable angles ---
        # Tie angle to the long edge so it doesn't jump by ~90 deg
        angle_long = angle_raw + 90.0 if rw < rh else angle_raw
        # Wrap to [0..180) for consistent comparisons
        angle_0_180 = (angle_long + 180.0) % 180.0
        # --- Passed all checks ---
        kept.append(c)
        kept_info.append({
            "A": float(A),
            "B": float(B),
            "ratio": float(ratio),
            "angle_raw": float(angle_raw),
            "angle_long": float(angle_long),
            "angle_0_180": float(angle_0_180),})
    return kept, rejected, kept_info
#____________________________________________________________________________________

#Step 2_1 Converts a contour's area from image pixels to world-space cm^2 using homography
#______________________________________________________________________________
def contour_area_in_world_cm2(cnt, H_img2world):
    pts = cnt.reshape(-1, 2).astype(np.float32)
    ones = np.ones((pts.shape[0], 1), dtype=np.float32)
    ph = np.hstack([pts, ones])                # [x, y, 1]
    Pw = (H_img2world @ ph.T).T                # project into world coords
    XY = Pw[:, :2] / Pw[:, 2:3]                # normalize
    XYc = XY.reshape(-1, 1, 2).astype(np.float32)
    area_cm2 = float(cv2.contourArea(XYc))     # compute real area
    return area_cm2
#___________________________________________________________________________

#Step 3 Masks the background for adaptive treshold usage
#____________________________________________________________________________________
def backgroundremover_1(img, kept_contours, fname, save_dir, MASK_BEFORE_TRESHOLD_COLOR):
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask, kept_contours, -1, 255, thickness=-1)  # fill inside contours
    # Create background with chosen color
    background = np.full_like(img, MASK_BEFORE_TRESHOLD_COLOR, dtype=np.uint8)
    # Combine image and background
    masked_result = np.where(mask[:, :, None] == 255, img, background)
    # Save only if SAVE_INTERMEDIATE is True
    if SAVE_INTERMEDIATE:
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, f"{os.path.splitext(fname)[0]}_masked.png")
        changer(out_path, masked_result)
    return masked_result
#_______________________________________________________________________________________

#Step 4 Applies adaptive treshold and then masks background as white
#_____________________________________________________________________________________
def adaptivethreshold(gray_img, kept_contours, fname, save_dir, contour_thickness=1):
    # 0) Brightness/contrast adjust (same logic as Step 1)
    x = gray_img.astype(np.float32) / 255.0
    y = (x - 0.5) * CONTRAST_2 + 0.5
    y = y * BRIGHTNESS_2
    y = np.clip(y, 0.0, 1.0)
    adjusted = (y * 255.0).astype(np.uint8)
    thr = cv2.adaptiveThreshold(adjusted, 255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,ADAPTIVE_BLOCKSIZE, ADAPTIVE_C)
    # --- Save if enabled ---
    if SAVE_INTERMEDIATE:
        os.makedirs(save_dir, exist_ok=True)
        base = os.path.splitext(fname)[0]
        changer(os.path.join(save_dir, f"{base}_adaptive_thresh.png"), thr)
    return thr
#_____________________________________________________________________________________

#Step 5 Masks the background with white
#__________________________________________________________________________________
def backgroundremover_2(img_gray, kept_contours, fname, save_dir):
    h, w = img_gray.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask, kept_contours, -1, 255, thickness=-1)  # fill inside contours
    # Create white background
    background = np.full_like(img_gray, 255, dtype=np.uint8)
    # Combine: keep inside contours, set outside to white
    masked_white = np.where(mask == 255, img_gray, background)
    # Save intermediate if enabled
    if SAVE_INTERMEDIATE:
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, f"{os.path.splitext(fname)[0]}_masked_white.png")
        changer(out_path, masked_white)
    return masked_white
#______________________________________________________________________________________

#Step 6 grip area filtering
#_____________________________________________________________________________________
def segment_and_merge_from_gray(img_gray, fname, save_dir):
    
    h, w = img_gray.shape[:2]
    # 1) Median blur
    blur = cv2.medianBlur(img_gray, MED_K)
    # 2) Otsu (inverted) threshold
    _, thr = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # 3) Morphology: open -> dilate
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (OPEN_K, OPEN_K))
    opened = cv2.morphologyEx(thr, cv2.MORPH_OPEN, k, iterations=OPEN_ITERS)
    dilated = cv2.dilate(opened, k, iterations=DILATE_ITERS)
    # 4) Remove micro components (connected components filter)
    nb, labels, stats, _ = cv2.connectedComponentsWithStats(dilated, connectivity=8)
    cleaned = np.zeros_like(dilated)
    for lab in range(1, nb):
        if stats[lab, cv2.CC_STAT_AREA] >= MIN_SPECKLE_AREA:
            cleaned[labels == lab] = 255
    # 5) Merge close contours via morphological close on a drawn mask
    cnts_pre, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Draw prelim contours (area pre-filter)
    prelim = [c for c in cnts_pre if cv2.contourArea(c) >= MIN_MAIN_AREA]
    merge_mask = np.zeros_like(cleaned)
    if prelim:
        cv2.drawContours(merge_mask, prelim, -1, 255, thickness=-1)
        k_merge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MERGE_GAP_PX + 1, MERGE_GAP_PX + 1))
        merged_mask = cv2.morphologyEx(merge_mask, cv2.MORPH_CLOSE, k_merge, iterations=1)
    else:
        merged_mask = merge_mask
    merged_contours, _ = cv2.findContours(merged_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 6) Final area filter
    final_contours = [c for c in merged_contours if cv2.contourArea(c) >= MIN_MAIN_AREA]
    # Visualization
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    if final_contours:
        cv2.drawContours(vis, final_contours, -1, (255, 255, 255), thickness=-1)# thixkness=-1 so it fills inside the contours
    return final_contours, vis, thr, cleaned
#___________________________________________________________________________________

#Step 7 Most isolated point finder.
#_____________________________________________________________________________________
def largest_inscribed_circles_per_parent(final_contours, parent_contours, image_shape, fname=None, save_dir=None):
    h, w = image_shape[:2]
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    if not final_contours or not parent_contours:
        return {}, vis
    # Helper: build parent polygons from their minAreaRect boxes
    parent_polys = []
    for pc in parent_contours:
        rect = cv2.minAreaRect(pc)
        box = cv2.boxPoints(rect).astype(np.int32).reshape(-1, 1, 2)
        parent_polys.append(box)
    # Draw final contours as white fill for context
    cv2.drawContours(vis, final_contours, -1, (255, 255, 255), thickness=-1)
    best_circles = {}  # parent_idx -> (r, (x,y))
    # For each final (child) contour, compute its largest inscribed circle via distance transform
    for c in final_contours:
        # Make a binary mask of just this contour
        c_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(c_mask, [c], -1, 255, thickness=-1)
        # Distance transform to get farthest point from boundary
        dist = cv2.distanceTransform(c_mask, cv2.DIST_L2, 5)
        _, maxVal, _, maxLoc = cv2.minMaxLoc(dist)
        r = int(maxVal)
        if r <= 0:
            continue
        # Assign this circle to a parent if its center is inside that parent's polygon
        parent_idx = -1
        for i, poly in enumerate(parent_polys):
            if cv2.pointPolygonTest(poly, maxLoc, False) >= 0:
                parent_idx = i
                break
        if parent_idx < 0:
            continue
        prev = best_circles.get(parent_idx)
        if (prev is None) or (r > prev["r"]):
            best_circles[parent_idx] = {"center": (int(maxLoc[0]), int(maxLoc[1])), "r": r}
    # Draw best circles per parent
    for pidx, info in best_circles.items():
        cx, cy = info["center"]
        r = info["r"]
        cv2.circle(vis, (cx, cy), r, (0, 0, 255), 2)   # red circle
        cv2.circle(vis, (cx, cy), 2, (0, 255, 0), -1)  # green center dot
        # (Optional) outline parent box for reference:
        # cv2.polylines(vis, [parent_polys[pidx]], True, (255, 0, 0), 1)
    if SAVE_INTERMEDIATE and fname is not None and save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        base = os.path.splitext(fname)[0]
        changer(os.path.join(save_dir, f"{base}_best_circles.png"), vis)
    return best_circles, vis
#_____________________________________________________________________________________
#Step 8 Orientation classification 
#_____________________________________________________________________________________
# Step 8: Orientation classification (spatially assigns child to its parent)
#_____________________________________________________________________________________
def orientation_filter_by_rectangles(step6_contours, parent_contours, parent_infos, fname, save_dir, base_image):
    vis = base_image.copy()
    if vis.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    # build polygons for parents from their minAreaRect boxes
    parent_polys = []
    for pc in parent_contours:
        rect = cv2.minAreaRect(pc)
        box = cv2.boxPoints(rect).astype(np.int32).reshape(-1, 1, 2)
        parent_polys.append(box)

    results = []
    for i, (parent, poly) in enumerate(zip(parent_infos, parent_polys)):
        parent_angle = parent["angle_0_180"]
        parent_status = "UP"
        found_rect = None

        for c in step6_contours:
            A = cv2.contourArea(c)
            if not (MIN_RECT_AREA_PX <= A <= MAX_RECT_AREA_PX): 
                continue

            rect = cv2.minAreaRect(c)
            (cx, cy), (rw, rh), ang_raw = rect
            if rw <= 0 or rh <= 0: 
                continue

            long_side, short_side = max(rw, rh), min(rw, rh)
            aspect = long_side / short_side if short_side > 0 else 0
            rect_angle = (ang_raw + 90.0) if rw < rh else ang_raw
            rect_angle = (rect_angle + 180.0) % 180.0
            angle_diff = abs(rect_angle - parent_angle)
            if angle_diff > 90:
                angle_diff = 180 - angle_diff

            # check if rectangle center lies *inside this parent*
            if cv2.pointPolygonTest(poly, (cx, cy), False) < 0:
                continue

            if angle_diff <= ANGLE_TOLERANCE_DEG and aspect >= MIN_ASPECT_RATIO:
                found_rect = rect
                parent_status = "DOWN"
                break

        if found_rect is not None:
            box = cv2.boxPoints(found_rect).astype(int)
            cv2.drawContours(vis, [box], 0, (0, 255, 0), 2)  # green rect for DOWN
            # optional: draw parent outline
            # cv2.polylines(vis, [poly], True, (255, 0, 0), 1)

        results.append({"parent_index": i, "status": parent_status, "angle": parent_angle})
    if SAVE_FINAL:
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, f"{os.path.splitext(fname)[0]}.png")
        changer(out_path, vis)
    return results, vis

# _____________________________________________________________________________________

#Step 9 Accuracy calculation per photo + total (with image_x_ rule)
#_____________________________________________________________________________________
def accuracy_evaluator(fname, orientation_results):
    base = os.path.splitext(fname)[0]
    parts = base.split("_")
    mid = parts[1].lower() if len(parts) >= 2 else ""
    suffix = parts[2].lower() if len(parts) >= 3 else ""

    # Expected counts / "none" rule
    # - image_x_  -> mid == "x" and suffix == ""  => expect none
    # - image_2_xx (or any suffix only 'x')       => expect none
    if (suffix == "" and mid == "x") or (suffix != "" and set(suffix) == {"x"}):
        exp_down, exp_up, expected_none = 0, 0, True
    else:
        exp_down = suffix.count("a")
        exp_up   = suffix.count("u")
        expected_none = False

    # Predictions
    pred_down = sum(1 for r in orientation_results if r.get("status") == "DOWN")
    pred_up   = sum(1 for r in orientation_results if r.get("status") == "UP")
    predicted_none = (pred_down + pred_up) == 0

    # Scoring
    if expected_none:
        photo_acc = 100.0 if predicted_none else 0.0
    else:
        total_expected = exp_down + exp_up
        total_pred = pred_down + pred_up
        total_for_score = max(total_expected, total_pred, 1)  # avoid /0
        correct = min(pred_down, exp_down) + min(pred_up, exp_up)
        photo_acc = (correct / total_for_score) * 100.0

    return {
        "file": base,
        "expected": {"down": exp_down, "up": exp_up, "none": expected_none},
        "pred": {"down": pred_down, "up": pred_up, "none": predicted_none},
        "photo_acc": photo_acc
    }

#____________________________________________________________________________
# Main Execution Pipeline
# _____________________________________________________________________________________
def main():
    print("=== Grip Area Finder Pipeline Started ===")
    processed = 0
    correct_count = 0
    zero_acc_files = []

    for fname, undistorted in undistorter(INPUT_FOLDER, CALIB_NPZ, SAVE_INTERMEDIATE, UNDISTORTED_FOLDER):
        print(f"\nProcessing file: {fname}")
        processed += 1

        if STEP1_ALTERNATIVE:
            contours, thr, vis, adjusted = step1_alternative_hybrid(fname, undistorted, save=SAVE_INTERMEDIATE, out_dir=CONTOUR_OUTPUT_FOLDER)
        else:
            contours, thr, vis, adjusted = otsu_treshold_filtering(fname, undistorted, save=SAVE_INTERMEDIATE, out_dir=CONTOUR_OUTPUT_FOLDER)

        # Step 2: Filter contours (kept/rejected)
        kept_contours, rejected, kept_info = shed_filter(contours, undistorted.shape)
        print(f"  Step 2 - Kept contours: {len(kept_contours)}, Rejected: {len(rejected)}")

        # Step 3: Mask background with gray before adaptive threshold
        masked_gray = backgroundremover_1(undistorted, kept_contours, fname, MASKED_RESULTS_FOLDER, MASK_BEFORE_TRESHOLD_COLOR)

        # Step 4: Adaptive threshold (operate on grayscale)
        gray_for_adaptive = cv2.cvtColor(masked_gray, cv2.COLOR_BGR2GRAY)
        adaptive_thr = adaptivethreshold(gray_for_adaptive, kept_contours, fname, ADAPTIVE_THRESHOLD_OUTPUT)

        # Step 5: Mask background as white
        masked_white = backgroundremover_2(adaptive_thr, kept_contours, fname, MASKED_RESULTS_FOLDER)

        # Step 6: Segment and merge from gray (Grip area filtering)
        final_contours, vis_final, thr_final, cleaned = segment_and_merge_from_gray(masked_white, fname, OUTPUT_FOLDER)
        print(f"  Step 6 - Final merged contours: {len(final_contours)}")

        # Step 7: Largest inscribed circle per parent
        best_circles, vis_circles = largest_inscribed_circles_per_parent(final_contours, kept_contours, undistorted.shape, fname, OUTPUT_FOLDER)
        print(f"  Step 7 - Found {len(best_circles)} best circles for {len(kept_contours)} parents")
        # [REALLIFE] step 3: MIC centre -> robot mm. Method 1 undistorts with its own camera matrix (getOptimalNewCameraMatrix),
        # so that matrix is handed over and the pixel is moved into the frame of H before H is applied.
        _newK, _ = cv2.getOptimalNewCameraMatrix(RL_MTX, RL_DIST, (undistorted.shape[1], undistorted.shape[0]), alpha=0)
        for _pidx in sorted(best_circles):
            _c = best_circles[_pidx]
            rl_report(fname, _pidx + 1, _c["center"][0], _c["center"][1], _c["r"], from_K=_newK)
        rl_save_overlay(undistorted, fname)

        # Step 8: Orientation classification (draws on same image as circles)
        orientation_results, vis_orient = orientation_filter_by_rectangles(
    final_contours, kept_contours, kept_info, fname, OUTPUT_FOLDER, vis_circles)

        print(f"  Step 8 - Orientation results: {[r['status'] for r in orientation_results]}")

         # Step 9: Simple filename-based exact-check
        base = os.path.splitext(fname)[0]
        parts = base.split("_", 2)
        suffix = parts[2].lower() if len(parts) >= 3 else ""
        # treat pure-x suffix (image_x_ or image_xx) as "none"
        if suffix != "" and all(ch == "x" for ch in suffix):
            exp_down, exp_up = 0, 0
        else:
            exp_down = suffix.count("a")
            exp_up   = suffix.count("u")

        pred_down = sum(1 for r in orientation_results if r.get("status") == "DOWN")
        pred_up = sum(1 for r in orientation_results if r.get("status") == "UP")

        is_correct = (pred_down == exp_down and pred_up == exp_up)
        photo_acc = 100.0 if is_correct else 0.0
        if is_correct: correct_count += 1
        else: 
            zero_acc_files.append(fname)

        print(f"  Step 9 - Expected down={exp_down} up={exp_up} | Pred down={pred_down} up={pred_up} | Acc={photo_acc:.2f}%")

    print("\n=== Processing Complete ===")
    if processed > 0:
        overall = (100.0 * correct_count / processed)
        print(f"Processed {processed} images, Exact-match correct: {correct_count}, Overall exact accuracy: {overall:.2f}%")
        if zero_acc_files:
            print("\nImages with 0% accuracy:")
            for zf in zero_acc_files:
                print(f" - {zf}")

    print("\n=== Processing Complete ===")
    if processed > 0:
        overall = (100.0 * correct_count / processed)
        print(f"Processed {processed} images, Exact-match correct: {correct_count}, Overall exact accuracy: {overall:.2f}%")
    else:
        print("No images processed.")

    print("\n=== Processing Complete ===")
if __name__ == "__main__":
    main()

# _____________________________________________________________________________________