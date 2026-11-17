import os
import pickle
import numpy as np
import pandas as pd
import warnings
import cv2
import glob
import sys
import matplotlib.pyplot as plt
import json    

PROJECT_ROOT = os.environ.get("GRIP_PROJECT_ROOT", os.path.abspath("project_root"))   # see README

warnings.filterwarnings("ignore")

# ==============================================================================
# 1. Parameters
# ==============================================================================

CALIBRATION_IMG_DIR = os.path.join(PROJECT_ROOT, r"codes_results\grippointfinder\unified_point_finder algorithm\unifier_3\camera_calibration\calibration_photos")
TEST_IMG_DIR        = os.path.join(PROJECT_ROOT, r"codes_results\grippointfinder\unified_point_finder algorithm\unifier_3\camera_calibration\heatmap_photos")
OUTPUT_DIR          = os.path.join(PROJECT_ROOT, r"codes_results\grippointfinder\unified_point_finder algorithm\unifier_3\camera_calibration\calibration_outputs")

BOARD_W, BOARD_H = 9, 6
SQUARE_SIZE = 1.0 
GRID_OPTION = "adaptive"  # "adaptive" or "manual"
TARGET_H = 4              
MANUAL_GRID_SIZE = 40            
DELTA_THRESHOLD_Z = 2.0  
MAX_FILL_ITERATIONS = 3    
MERGE_SIZES = [2, 3, 4, 5, 6]      

TERMINATION_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# ==============================================================================
# 2. Control
# ==============================================================================
if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
ERR_DIR = os.path.join(OUTPUT_DIR, 'errors')
if not os.path.exists(ERR_DIR): os.makedirs(ERR_DIR)

def get_common_divisors(w, h):
    divs_w = [i for i in range(1, w + 1) if w % i == 0]
    divs_h = [i for i in range(1, h + 1) if h % i == 0]
    return sorted(list(set(divs_w) & set(divs_h)))

# ==============================================================================
# 3. Calibration
# ==============================================================================
print("Starting the calibration process...")
calib_images = []
for ext in ('*.jpg', '*.jpeg', '*.bmp', '*.JPG', '*.BMP'):
    calib_images.extend(glob.glob(os.path.join(CALIBRATION_IMG_DIR, ext)))

img_shape = None
objpoints, imgpoints = [], []
objp = np.zeros((BOARD_W * BOARD_H, 3), np.float32)
objp[:, :2] = np.mgrid[0:BOARD_W, 0:BOARD_H].T.reshape(-1, 2) * SQUARE_SIZE

for fname in calib_images:
    img = cv2.imread(fname)
    if img is None: continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img_shape is None: 
        img_shape = gray.shape[::-1] # (width, height)
    ret, corners = cv2.findChessboardCorners(gray, (BOARD_W, BOARD_H), None)
    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), TERMINATION_CRITERIA)
        imgpoints.append(corners2)

if not objpoints:
    print("Error: calibration photos could not be read!"); sys.exit()

width, height = img_shape
ret_val, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (width, height), None, None)


calib_params_json = {"rms_error": ret_val, "mtx": mtx.tolist(), "dist": dist.tolist(), "width": width, "height": height}
with open(os.path.join(OUTPUT_DIR, "Camera_Calibration_Params.json"), "w") as f: json.dump(calib_params_json, f, indent=4)
with open(os.path.join(OUTPUT_DIR, "Camera_Calibration_Params.pkl"), "wb") as f: pickle.dump({"mtx": mtx, "dist": dist}, f)

# ==============================================================================
# 4. TEST
# ==============================================================================
test_images = []
for ext in ('*.jpg', '*.jpeg', '*.bmp', '*.JPG', '*.BMP'):
    test_images.extend(glob.glob(os.path.join(TEST_IMG_DIR, ext)))

raw_point_data = []
for fname in test_images:
    img = cv2.imread(fname)
    if img is None: continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (BOARD_W, BOARD_H), None)
    if ret:
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), TERMINATION_CRITERIA)
        _, rvec, tvec = cv2.solvePnP(objp, corners2, mtx, dist)
        imgpoints2, _ = cv2.projectPoints(objp, rvec, tvec, mtx, dist)
        for i in range(len(objp)):
            error = np.linalg.norm(corners2[i][0] - imgpoints2[i][0])
            raw_point_data.append({'x': corners2[i][0][0], 'y': corners2[i][0][1], 'err': error, 'file': os.path.basename(fname)})

f = min(get_common_divisors(width, height), key=lambda x: abs(x - np.sqrt((TARGET_H * width * height) / len(raw_point_data)))) if GRID_OPTION == "adaptive" else MANUAL_GRID_SIZE

gx_count, gy_count = width // f, height // f
consensus_grid = np.zeros((gy_count, gx_count))
consensus_counts = np.zeros((gy_count, gx_count))

for p in raw_point_data:
    gx, gy = int(p['x'] // f), int(p['y'] // f)
    if gx < gx_count and gy < gy_count:
        consensus_grid[gy, gx] += p['err']
        consensus_counts[gy, gx] += 1

with np.errstate(divide='ignore', invalid='ignore'):
    consensus_means = np.where(consensus_counts > 0, consensus_grid / consensus_counts, 0)

# ==============================================================================
# 5. Elimination
# ==============================================================================
image_deltas = {}
for base in set(p['file'] for p in raw_point_data):
    pts = [p for p in raw_point_data if p['file'] == base]
    deltas = [p['err'] - consensus_means[int(p['y'] // f), int(p['x'] // f)] for p in pts]
    image_deltas[base] = np.mean(deltas)

d_mean, d_std = np.mean(list(image_deltas.values())), np.std(list(image_deltas.values()))
valid_photos = [name for name, d in image_deltas.items() if abs(d - d_mean) <= DELTA_THRESHOLD_Z * d_std]

final_err_sum = np.zeros((gy_count, gx_count))
final_cnt = np.zeros((gy_count, gx_count))
for p in raw_point_data:
    if p['file'] in valid_photos:
        gx, gy = int(p['x'] // f), int(p['y'] // f)
        if gx < gx_count and gy < gy_count:
            final_err_sum[gy, gx] += p['err']
            final_cnt[gy, gx] += 1

augmented_means = np.full((gy_count, gx_count), np.nan)
threshold_h = TARGET_H / 2
for gy in range(gy_count):
    for gx in range(gx_count):
        if final_cnt[gy, gx] >= threshold_h:
            augmented_means[gy, gx] = final_err_sum[gy, gx] / final_cnt[gy, gx]


for iteration in range(MAX_FILL_ITERATIONS):
    temp_means = augmented_means.copy()
    for gy in range(gy_count):
        for gx in range(gx_count):
            if np.isnan(augmented_means[gy, gx]):
                valid_neighbors = [augmented_means[ny, nx] for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]
                                   for ny, nx in [(gy+dy, gx+dx)] if 0 <= ny < gy_count and 0 <= nx < gx_count and not np.isnan(augmented_means[ny, nx])]
                if len(valid_neighbors) >= 2: temp_means[gy, gx] = np.mean(valid_neighbors)
    augmented_means = temp_means.copy()


locked_mask = np.zeros((gy_count, gx_count), dtype=bool)
cy, cx = gy_count // 2, gx_count // 2
for K in MERGE_SIZES:
    corners = sorted([(y, x) for y in range(gy_count - K + 1) for x in range(gx_count - K + 1)], key=lambda p: (p[0] + K/2 - cy)**2 + (p[1] + K/2 - cx)**2)
    for y, x in corners:
        if not np.any(locked_mask[y:y+K, x:x+K]) and not np.isnan(augmented_means[y:y+K, x:x+K]).any():
            augmented_means[y:y+K, x:x+K] = np.mean(augmented_means[y:y+K, x:x+K])
            locked_mask[y:y+K, x:x+K] = True


valid_coords = np.argwhere(~np.isnan(augmented_means))
valid_values = augmented_means[~np.isnan(augmented_means)]
if len(valid_values) > 0:
    global_max_err = np.nanmax(augmented_means)
    max_dist = np.sqrt(max(cy, gy_count - cy - 1)**2 + max(cx, gx_count - cx - 1)**2)
    for gy in range(gy_count):
        for gx in range(gx_count):
            if np.isnan(augmented_means[gy, gx]):
                dists = np.sqrt((valid_coords[:, 0] - gy)**2 + (valid_coords[:, 1] - gx)**2)
                nearest = valid_coords[np.argmin(dists)]
                dist_curr = np.sqrt((gy - cy)**2 + (gx - cx)**2)
                f_ratio = np.clip((dist_curr - np.sqrt((nearest[0] - cy)**2 + (nearest[1] - cx)**2)) / (max_dist + 1e-9), 0, 1)
                augmented_means[gy, gx] = valid_values[np.argmin(dists)] + f_ratio * (global_max_err - valid_values[np.argmin(dists)])

# ==============================================================================
# Save
# ==============================================================================
pixel_error_map = cv2.resize(augmented_means.astype(np.float32), (width, height), interpolation=cv2.INTER_NEAREST)

with open(os.path.join(OUTPUT_DIR, "camera_calibration_error_heatmap.pkl"), "wb") as f: pickle.dump(pixel_error_map, f)


pure_heatmap_color = cv2.applyColorMap(cv2.resize(((np.clip(augmented_means, 0, 2) / 2.0) * 255).astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST), cv2.COLORMAP_INFERNO)
cv2.imwrite(os.path.join(OUTPUT_DIR, 'Camera_Calibration_debug_photo.png'), pure_heatmap_color)

plt.figure(figsize=(16, 9), dpi=150)
im = plt.imshow(augmented_means, cmap='inferno', interpolation='nearest', extent=[0, width, height, 0], vmin=0, vmax=2.0)
plt.colorbar(im, label='Mean Reprojection Error (Pixels)')
plt.savefig(os.path.join(OUTPUT_DIR, 'camera_calibration_error_heatmap.png'))
print("done")