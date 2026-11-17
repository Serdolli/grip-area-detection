import os
import numpy as np
import pickle
import cv2
from scipy.interpolate import griddata

PROJECT_ROOT = os.environ.get("GRIP_PROJECT_ROOT", os.path.abspath("project_root"))   # see README

# =====================================================================
# 1. PARAMETERS AND PATHS
# =====================================================================
OUTPUT_DIR = os.path.join(PROJECT_ROOT, r"codes_results\grippointfinder\unified_point_finder algorithm\unifier_3\robot_theoric_error_heatmaps")
TRANSFORM_MATRIX_PATH = os.path.join(PROJECT_ROOT, r"codes_results\grippointfinder\unified_point_finder algorithm\unifier_2\test_1\transformation_matrix_residual_heatmap\H.npy")

GRIPPER_LENGTH = 0.2
Z_TCP_TARGET = 0                          
Z_TARGET     = Z_TCP_TARGET + GRIPPER_LENGTH  
JOINT_ERRORS = np.array([0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001])
GRID_SIZE_PX = 20

# DH parameters
DH_A     = [0, -0.24355, -0.2132, 0,       0,       0       ]
DH_D     = [0.15185, 0,  0,        0.13105, 0.08535, 0.0921]
DH_ALPHA = [np.pi/2, 0,  0,        np.pi/2, -np.pi/2, 0    ]
JOINT_LIMITS_DEG = 180

# =====================================================================
# 2. KINEMATICS
# =====================================================================
def get_matrix(theta, a, d, alpha):
    ct, st, ca, sa = np.cos(theta), np.sin(theta), np.cos(alpha), np.sin(alpha)
    return np.array([[ct, -st*ca, st*sa, a*ct], [st, ct*ca, -ct*sa, a*st], [0, sa, ca, d], [0, 0, 0, 1]])

def forward_kin(q):
    T = np.eye(4)
    for i in range(6): T = T @ get_matrix(q[i], DH_A[i], DH_D[i], DH_ALPHA[i])
    return T[:3, 3]

def get_jacobian_full(q):
    pos0 = forward_kin(q)
    J = np.zeros((3, 6))
    dq = 1e-6
    for i in range(6):
        qt = q.copy(); qt[i] += dq
        J[:, i] = (forward_kin(qt) - pos0) / dq
    return J

def inverse_kin(x, y, z, elbow):
    r = np.hypot(x, y); z_rel = z - DH_D[0]
    D = np.clip((r**2 + z_rel**2 - DH_A[1]**2 - DH_A[2]**2) / (2*abs(DH_A[1])*abs(DH_A[2])), -1, 1)
    q3 = np.arctan2(elbow * np.sqrt(max(0.0, 1.0 - D**2)), D)
    q2 = np.arctan2(z_rel, r) - np.arctan2(abs(DH_A[2])*np.sin(q3), abs(DH_A[1]) + abs(DH_A[2])*np.cos(q3))
    q_sol = np.array([np.arctan2(y, x), q2, q3, 0.0, -np.pi/2, 0.0])
    return q_sol if np.all(np.abs(np.degrees(q_sol)) <= JOINT_LIMITS_DEG) else None

def compute_method_D(J_full):
    Sigma_ee = J_full @ np.diag(JOINT_ERRORS**2) @ J_full.T
    sigma_ee_mm = np.sqrt(np.trace(Sigma_ee)) * 1000.0
    kappa = np.linalg.cond(J_full)
    return sigma_ee_mm * (10 if kappa > 10 else (2 if kappa > 5 else 1))

# =====================================================================
# 3.    Heatmap
# =====================================================================
def run_dynamic_analysis(img_w, img_h):
    print(f"\n[Dynamic analysis] Target resolution: {img_w}x{img_h}")
    TRANSFORM_MATRIX = np.load(TRANSFORM_MATRIX_PATH)
    
    xi = np.arange(0, img_w, GRID_SIZE_PX)
    yi = np.arange(0, img_h, GRID_SIZE_PX)
    grid_d, grid_m, grid_c = [np.full((len(yi), len(xi)), np.nan) for _ in range(3)]

    for r_idx, y_px in enumerate(yi):
        for c_idx, x_px in enumerate(xi):
            pt = TRANSFORM_MATRIX.dot([x_px, y_px, 1.0])
            x_m, y_m = (pt[0]/pt[2])/1000.0, (pt[1]/pt[2])/1000.0
            
            # Elbow is taken as 1 by default
            q = inverse_kin(x_m, y_m, Z_TARGET, 1) 
            if q is not None:
                J = get_jacobian_full(q)
                grid_d[r_idx, c_idx] = compute_method_D(J)
                grid_m[r_idx, c_idx] = np.sqrt(max(0, np.linalg.det(J @ J.T)))
                _, S, _ = np.linalg.svd(J)
                grid_c[r_idx, c_idx] = S[0] / (S[-1] + 1e-9)

    full_xi, full_yi = np.meshgrid(np.arange(img_w), np.arange(img_h))
    
    def process_and_save(grid_mat, name):
        valid = ~np.isnan(grid_mat)
        if np.any(valid):
            pts = np.vstack((np.meshgrid(xi, yi)[0][valid], np.meshgrid(xi, yi)[1][valid])).T
            full_map = griddata(pts, grid_mat[valid], (full_xi, full_yi), method='linear')
            full_map[np.isnan(full_map)] = griddata(pts, grid_mat[valid], (full_xi, full_yi), method='nearest')[np.isnan(full_map)]
        else:
            full_map = np.zeros((img_h, img_w))
        
        with open(os.path.join(OUTPUT_DIR, f"{name}.pkl"), "wb") as f:
            pickle.dump(full_map.astype(np.float32), f)
        print(f"[OK] {name}.pkl saved. Size: {full_map.shape}")

    process_and_save(grid_d, "robot_therotic_error_pixel_space")
    process_and_save(grid_m, "robot_manipulability_pixel_space")
    process_and_save(grid_c, "robot_condition_number_pixel_space")

# =====================================================================
# 4. Run
# =====================================================================
if __name__ == "__main__":
    run_dynamic_analysis(1280, 1024)