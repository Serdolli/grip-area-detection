import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import cv2
import json
import pickle
from scipy.interpolate import griddata

PROJECT_ROOT = os.environ.get("GRIP_PROJECT_ROOT", os.path.abspath("project_root"))   # see README

# =====================================================================
# 1. SETTINGS
# =====================================================================

OUTPUT_DIR = os.path.join(PROJECT_ROOT, r"codes_results\grippointfinder\unified_point_finder algorithm\test_5")

# -- Gripper parameter --------------------------------------------------
GRIPPER_LENGTH = 0.10   # meters
# -----------------------------------------------------------------------

# -- Camera and transformation parameters -------------------------------
CAMERA_WIDTH  = 1280
CAMERA_HEIGHT = 1024
GRID_SIZE_PX  = 40
TRANSFORM_MATRIX_PATH = os.path.join(PROJECT_ROOT, r"codes_results\grippointfinder\unified_point_finder algorithm\test_2\transformation_matrix\H.npy")
# -----------------------------------------------------------------------

Z_TCP_TARGET = 0.30                         
Z_TARGET     = Z_TCP_TARGET + GRIPPER_LENGTH  

JOINT_ERRORS = np.array([0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001])
GRID_SIZE    = 0.003

DH_A     = [0, -0.24355, -0.2132, 0,       0,       0     ]
DH_D     = [0.15185, 0,  0,       0.13105, 0.08535, 0.0921]
DH_ALPHA = [np.pi/2, 0,  0,       np.pi/2, -np.pi/2, 0    ]

MIN_REACH_RADIUS = 0.112
MAX_REACH_RADIUS = 0.511
JOINT_LIMITS_DEG = 180

REACHABILITY_NPY      = None
REACHABILITY_META_NPY = None
FORCE_ELBOW_SIGN = None

# =====================================================================
# 2. PREPARATION AND AUTOMATIC ROI (X-Y LIMITS)
# =====================================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Load the matrix
if not os.path.exists(TRANSFORM_MATRIX_PATH):
    raise FileNotFoundError(f"Transformation matrix not found: {TRANSFORM_MATRIX_PATH}")

if TRANSFORM_MATRIX_PATH.endswith('.npy'):
    TRANSFORM_MATRIX = np.load(TRANSFORM_MATRIX_PATH)
else:
    with open(TRANSFORM_MATRIX_PATH, "rb") as f:
        TRANSFORM_MATRIX = pickle.load(f)

# 2. Extract the ROI by converting the 4 outer corners of the camera to the physical world (meters)
corners_px = np.array([
    [0, 0, 1.0],
    [CAMERA_WIDTH, 0, 1.0],
    [0, CAMERA_HEIGHT, 1.0],
    [CAMERA_WIDTH, CAMERA_HEIGHT, 1.0]
])

transformed_corners = TRANSFORM_MATRIX.dot(corners_px.T).T
x_mm = transformed_corners[:, 0] / transformed_corners[:, 2]
y_mm = transformed_corners[:, 1] / transformed_corners[:, 2]

x_m = x_mm / 1000.0
y_m = y_mm / 1000.0

# 3. Determine the X and Y limits automatically
X_LIMITS = [np.min(x_m), np.max(x_m)]
Y_LIMITS = [np.min(y_m), np.max(y_m)]

print(f"\n[OK] AUTOMATIC ROI (camera field of view) LOCKED:")
print(f"   X axis limits: {X_LIMITS[0]*1000:.1f} mm  ->  {X_LIMITS[1]*1000:.1f} mm")
print(f"   Y axis limits: {Y_LIMITS[0]*1000:.1f} mm  ->  {Y_LIMITS[1]*1000:.1f} mm\n")

# Reachability check
reach_mask = reach_range_limit = reach_resolution = None
if REACHABILITY_NPY and os.path.exists(REACHABILITY_NPY):
    reach_mask = np.load(REACHABILITY_NPY)
    if REACHABILITY_META_NPY and os.path.exists(REACHABILITY_META_NPY):
        meta = np.load(REACHABILITY_META_NPY)
        reach_range_limit, reach_resolution = float(meta[0]), float(meta[1])
    print(f"[OK] Reachability mask loaded.")
else:
    print("[INFO] No reachability mask - using the radial filter.")

def is_reachable(x, y):
    if reach_mask is not None and reach_range_limit is not None:
        rl, res = reach_range_limit, reach_resolution
        col = int(round((x + rl) / res))
        row = int(round((y + rl) / res))
        n_y, n_x = reach_mask.shape
        if 0 <= row < n_y and 0 <= col < n_x:
            val = reach_mask[row, col]
            return (not np.isnan(val)) and (val >= 1.0)
        return False
    r = np.hypot(x, y)
    return MIN_REACH_RADIUS <= r <= MAX_REACH_RADIUS

# =====================================================================
# 3. KINEMATICS
# =====================================================================

def get_matrix(theta, a, d, alpha):
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [0,   sa,     ca,     d   ],
        [0,   0,      0,      1   ]
    ])

def forward_kin(q):
    T = np.eye(4)
    for i in range(6):
        T = T @ get_matrix(q[i], DH_A[i], DH_D[i], DH_ALPHA[i])
    return T[:3, 3]

def get_jacobian_full(q):
    pos0 = forward_kin(q)
    J = np.zeros((3, 6))
    dq = 1e-6
    for i in range(6):
        qt = q.copy(); qt[i] += dq
        J[:, i] = (forward_kin(qt) - pos0) / dq
    return J

def test_elbow_sign():
    x_t, y_t, z_t = 0.30, 0.0, Z_TARGET
    r     = np.hypot(x_t, y_t)
    z_rel = z_t - DH_D[0]
    L2, L3 = abs(DH_A[1]), abs(DH_A[2])
    D = np.clip((r**2 + z_rel**2 - L2**2 - L3**2) / (2*L2*L3), -1, 1)

    results = {}
    for sign in [+1, -1]:
        q3 = np.arctan2(sign * np.sqrt(max(0, 1 - D**2)), D)
        q2 = np.arctan2(z_rel, r) - np.arctan2(L3*np.sin(q3), L2 + L3*np.cos(q3))
        q  = np.array([np.arctan2(y_t, x_t), q2, q3, 0, -np.pi/2, 0])

        T = np.eye(4)
        for i in range(3):
            T = T @ get_matrix(q[i], DH_A[i], DH_D[i], DH_ALPHA[i])
            elbow_z = T[2, 3]
            results[sign] = elbow_z

    return max(results, key=results.get)

detected_sign = test_elbow_sign()
CONF_ELBOW = FORCE_ELBOW_SIGN if FORCE_ELBOW_SIGN is not None else detected_sign

def inverse_kin(x, y, z, elbow):
    if not is_reachable(x, y): return None

    r     = np.hypot(x, y)
    q1    = np.arctan2(y, x)
    L2, L3 = abs(DH_A[1]), abs(DH_A[2])
    z_rel = z - DH_D[0]

    D_raw = (r**2 + z_rel**2 - L2**2 - L3**2) / (2*L2*L3)
    if abs(D_raw) > 1.0: return None
    D  = np.clip(D_raw, -1.0, 1.0)
    q3 = np.arctan2(elbow * np.sqrt(max(0.0, 1.0 - D**2)), D)

    if detected_sign == +1 and q3 <= 0.05: return None
    if detected_sign == -1 and q3 >= -0.05: return None

    q2    = np.arctan2(z_rel, r) - np.arctan2(L3*np.sin(q3), L2 + L3*np.cos(q3))
    q_sol = np.array([q1, q2, q3, 0.0, -np.pi/2, 0.0])

    if np.any(np.abs(np.degrees(q_sol)) > JOINT_LIMITS_DEG): return None
    return q_sol

# =====================================================================
# 4. METHOD D
# =====================================================================

COV_Q = np.diag(JOINT_ERRORS**2)

def compute_method_D(J_full, err_mm):
    Sigma_ee = J_full @ COV_Q @ J_full.T
    sigma_ee_mm = np.sqrt(np.trace(Sigma_ee)) * 1000.0
    kappa = np.linalg.cond(J_full)

    if kappa > 10: sigma_ee_mm *= 10
    elif kappa > 5: sigma_ee_mm *= 2

    return sigma_ee_mm

# =====================================================================
# 5. GRID SCAN AND DATABASE CREATION
# =====================================================================

GLOBAL_LIMIT = 0.60
x_global = np.arange(-GLOBAL_LIMIT, GLOBAL_LIMIT, GRID_SIZE)
y_global = np.arange(-GLOBAL_LIMIT, GLOBAL_LIMIT, GRID_SIZE)

heatmap_D = np.full((len(y_global), len(x_global)), np.nan)
lookup_rows = []

for i, y in enumerate(y_global):
    for j, x in enumerate(x_global):
        q = inverse_kin(x, y, Z_TARGET, CONF_ELBOW)
        if q is None: continue

        J_full = get_jacobian_full(q)
        vD = compute_method_D(J_full, 0)
        heatmap_D[i, j] = vD

        # Save the points inside the automatically determined ROI limits
        if X_LIMITS[0] <= x <= X_LIMITS[1] and Y_LIMITS[0] <= y <= Y_LIMITS[1]:
            lookup_rows.append([round(x*1000,2), round(y*1000,2), round(vD,6)])

pd.DataFrame(lookup_rows, columns=["X_mm","Y_mm","D_sigma"]).to_csv(
    os.path.join(OUTPUT_DIR, "Kombine_Hata_Veritabani.csv"), index=False, sep=';')

grid_dict_pkl = {(row[0], row[1]): row[2] for row in lookup_rows}
with open(os.path.join(OUTPUT_DIR, "robot_error_heatmap.pkl"), "wb") as f:
    pickle.dump(grid_dict_pkl, f)

grid_dict_json = {}
for x, y, d in lookup_rows:
    if x not in grid_dict_json: grid_dict_json[x] = {}
    grid_dict_json[x][y] = d

with open(os.path.join(OUTPUT_DIR, "robot_error_heatmap.json"), "w") as f:
    json.dump(grid_dict_json, f, indent=4)

# =====================================================================
# 6. VISUALIZATION
# =====================================================================

cmap = plt.cm.viridis.copy()
cmap.set_bad(color='black', alpha=0.9)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

valid = heatmap_D[~np.isnan(heatmap_D)]
vmax = np.percentile(valid, 95) if len(valid) > 0 else 1.0

ext = [-GLOBAL_LIMIT*1000, GLOBAL_LIMIT*1000, -GLOBAL_LIMIT*1000, GLOBAL_LIMIT*1000]

im1 = axes[0].imshow(heatmap_D, extent=ext, origin='lower', cmap=cmap, vmax=vmax)
axes[0].set_title(f"Method D + Condition [Gripper={GRIPPER_LENGTH*1000:.0f} mm, Z_flange={Z_TARGET*1000:.0f} mm]")

# Automatically extracted red rectangle (camera ROI)
axes[0].add_patch(patches.Rectangle(
    (X_LIMITS[0]*1000, Y_LIMITS[0]*1000),
    (X_LIMITS[1]-X_LIMITS[0])*1000,
    (Y_LIMITS[1]-Y_LIMITS[0])*1000,
    edgecolor='red', facecolor='none'))

fig.colorbar(im1, ax=axes[0])

xs = np.searchsorted(x_global, X_LIMITS[0])
xe = np.searchsorted(x_global, X_LIMITS[1])
ys = np.searchsorted(y_global, Y_LIMITS[0])
ye = np.searchsorted(y_global, Y_LIMITS[1])

roi = heatmap_D[ys:ye, xs:xe]

im2 = axes[1].imshow(roi, extent=[X_LIMITS[0]*1000, X_LIMITS[1]*1000, Y_LIMITS[0]*1000, Y_LIMITS[1]*1000], origin='lower', cmap=cmap)
axes[1].set_title("Automatically detected camera field of view (ROI)")
fig.colorbar(im2, ax=axes[1])

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "Risk_Analizi_D.png"), dpi=300)

# =====================================================================
# 7. PIXEL-BASED ROBOT HEATMAP (1280x1024 CAMERA SPACE)
# =====================================================================

def generate_pixel_based_robot_heatmap(output_dir, img_width, img_height, grid_size=40):
    print(f"\n[Pixel Heatmap] Computing the robot kinematic error in the {img_width}x{img_height} camera space...")

    xi = np.arange(0, img_width, grid_size)
    yi = np.arange(0, img_height, grid_size)

    grid_dict_pkl = {}
    grid_dict_json = {}
    
    grid_matrix = np.full((len(yi), len(xi)), np.nan)

    for r_idx, y_px in enumerate(yi):
        for c_idx, x_px in enumerate(xi):
            
            pt_vector = np.array([x_px, y_px, 1.0])
            transformed = TRANSFORM_MATRIX.dot(pt_vector)
            x_m = (transformed[0] / transformed[2]) / 1000.0
            y_m = (transformed[1] / transformed[2]) / 1000.0
            
            q = inverse_kin(x_m, y_m, Z_TARGET, CONF_ELBOW)
            
            if q is not None:
                J_full = get_jacobian_full(q)
                vD = compute_method_D(J_full, 0)
                grid_matrix[r_idx, c_idx] = vD
                
                grid_dict_pkl[(int(x_px), int(y_px))] = float(vD)
                
                x_str, y_str = str(int(x_px)), str(int(y_px))
                if x_str not in grid_dict_json: grid_dict_json[x_str] = {}
                grid_dict_json[x_str][y_str] = float(vD)

    with open(os.path.join(output_dir, "robot_error_pixel_space.pkl"), "wb") as f: pickle.dump(grid_dict_pkl, f)
    with open(os.path.join(output_dir, "robot_error_pixel_space.json"), "w", encoding="utf-8") as f: json.dump(grid_dict_json, f, indent=4)

    print("[OK] robot_error_pixel_space files saved successfully!")

    plt.figure(figsize=(10, 8), dpi=150)
    cmap_px = plt.cm.viridis.copy()
    cmap_px.set_bad(color='black', alpha=0.9) 
    
    valid_data = grid_matrix[~np.isnan(grid_matrix)]
    vmax_px = np.percentile(valid_data, 95) if len(valid_data) > 0 else 1.0
    
    im_px = plt.imshow(grid_matrix, extent=[0, img_width, img_height, 0], origin="upper", cmap=cmap_px, vmax=vmax_px)
    plt.colorbar(im_px, label="Kinematic manipulation error (D_sigma)")
    plt.title(f"Robot error map from the {img_width}x{img_height} camera perspective")
    plt.xlabel("Camera X (px)")
    plt.ylabel("Camera Y (px)")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"Robot_Heatmap_{img_width}x{img_height}.png"))
    plt.close()

    # -- PLAIN CONTINUOUS HEATMAP OUTPUT (1280x1020 PLAIN PNG) --
    print("\n[Pure Continuous Robot Heatmap] Creating the plain continuous 1280x1020 robot kinematics map...")
    # Fill the outer blind spots the robot cannot reach (NaN) with the nearest border value (extrapolation)
    valid_mask = ~np.isnan(grid_matrix)
    if np.any(valid_mask):
        grid_r, grid_c = np.where(valid_mask)
        grid_vals = grid_matrix[valid_mask]
        all_r, all_c = np.indices(grid_matrix.shape)
        pure_grid_filled = griddata((grid_r, grid_c), grid_vals, (all_r, all_c), method='nearest')
    else:
        pure_grid_filled = np.zeros_like(grid_matrix)

    # Normalize the data to [0, 255] using the 95th-percentile upper limit (vmax_px)
    pure_grid_clip = np.clip(pure_grid_filled, 0.0, vmax_px)
    pure_grid_norm = ((pure_grid_clip / vmax_px) * 255).astype(np.uint8)

    # Resize to the final 1280x1020 target size with smooth cubic interpolation
    pure_grid_resized = cv2.resize(pure_grid_norm, (1280, 1020), interpolation=cv2.INTER_CUBIC)

    # Write to disk with the INFERNO palette (same as the other maps)
    pure_robot_heatmap_color = cv2.applyColorMap(pure_grid_resized, cv2.COLORMAP_INFERNO)

    pure_robot_map_path = os.path.join(output_dir, 'Robot_Heatmap_1280x1020_Pure.png')
    cv2.imwrite(pure_robot_map_path, pure_robot_heatmap_color)
    print(f"  [OK] Plain continuous robot heatmap input saved successfully: {pure_robot_map_path}")


# Run the dynamic pixel map function
generate_pixel_based_robot_heatmap(OUTPUT_DIR, CAMERA_WIDTH, CAMERA_HEIGHT, GRID_SIZE_PX)
plt.show()