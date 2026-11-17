

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import pickle
from scipy.interpolate import griddata

PROJECT_ROOT = os.environ.get("GRIP_PROJECT_ROOT", os.path.abspath("project_root"))   # see README

# ---------------------------------------------
# 0.Paths
# ---------------------------------------------
IMAGE_PATH           = os.path.join(PROJECT_ROOT, r"codes_results\grippointfinder\unified_point_finder algorithm\unifier_3\transformation_matrix\calibration_photo\transformation_photo.bmp")
CALIB_PATH           = os.path.join(PROJECT_ROOT, r"codes_results\grippointfinder\unified_point_finder algorithm\unifier_3\camera_calibration\calibration_outputs\Camera_Calibration_Params.pkl")
OUTPUT_DIR           = os.path.join(PROJECT_ROOT, r"codes_results\grippointfinder\unified_point_finder algorithm\unifier_3\transformation_matrix\transformation_matrix_error_heatmaps")
DETECTED_POINTS_PATH = os.path.join(OUTPUT_DIR, "detected_points.png")

# -- PHASE 2: REAL-WORLD (SHEET) DIMENSIONS --
# spacing of the circles on the printed calibration sheet (mm)
CIRCLE_SPACING_X = 20.0  # Center distance of two circles side by side (mm)
CIRCLE_SPACING_Y = 20.0  # Center distance of two circles one below the other (mm)

ROW_THRESHOLD = 20

# ---------------------------------------------
# 1. IMAGE AND POINT PROCESSING
# ---------------------------------------------
def load_and_undistort_image(img_path, calib_path):
    raw_img = cv2.imread(img_path)
    if raw_img is None:
        raise FileNotFoundError(f"Could not load image: {img_path}")
    with open(calib_path, "rb") as f:
        calib_data = pickle.load(f)
        mtx  = calib_data["mtx"]
        dist = calib_data["dist"]
    return cv2.undistort(raw_img, mtx, dist, None, mtx), mtx, dist


def detect_dots(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    dots = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 40:
            continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        # Circularity: 1.0 = perfect circle, 0.0 = completely irregular shape
        circularity = (4 * np.pi * area) / (perimeter ** 2)
        if circularity < 0.7:
            continue
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            dots.append((int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])))

    if not dots:
        raise RuntimeError("No circles/points could be detected in the image!")
    return dots


def sort_points(dots, row_threshold=ROW_THRESHOLD):
    points_sorted_y = sorted(dots, key=lambda p: (p[1], p[0]))
    rows = []
    for p in points_sorted_y:
        placed = False
        for row in rows:
            row_mean_y = np.mean([r[1] for r in row])
            if abs(row_mean_y - p[1]) < row_threshold:
                row.append(p)
                placed = True
                break
        if not placed:
            rows.append([p])

    for row in rows:
        row.sort(key=lambda p: p[0])
    rows.sort(key=lambda row: np.mean([p[1] for p in row]))

    sorted_points = []
    for row in rows:
        sorted_points.extend(row)

    return sorted_points, len(rows), max(len(r) for r in rows)


def generate_object_points(num_rows, num_cols):
    """Builds the exact 3D (X, Y, Z=0) map of the points on the sheet."""
    obj_points = []
    for r in range(num_rows):
        for c in range(num_cols):
            x = c * CIRCLE_SPACING_X
            y = r * CIRCLE_SPACING_Y
            obj_points.append([x, y, 0.0])
    return np.array(obj_points, dtype=np.float32)


# ---------------------------------------------
# 2. ROBOT 3-POINT INPUT & AXIS COMPUTATION
# ---------------------------------------------
def get_robot_reference_frame():
    print(f"\n{'='*55}")
    print("  ROBOT-WORLD CALIBRATION (ONLY 3 POINTS)")
    print(f"{'='*55}")
    print("  [!] Please move the robot EXACTLY to the center of the given circles.")

    pts = {}
    labels = {
        "Orijin": "FIRST circle at the top left (0,0)",
        "X_Axis": "Any circle TO THE RIGHT in the same row",
        "Y_Axis": "Any circle BELOW in the same column"
    }

    for key, desc in labels.items():
        while True:
            raw = input(f"  > Robot (X Y Z) for {desc}: ").strip()
            parts = raw.split()
            if len(parts) == 3:
                try:
                    pts[key] = np.array([float(parts[0]), float(parts[1]), float(parts[2])])
                    break
                except ValueError:
                    pass
            print("     [!] Invalid input! Format: X Y Z (e.g. 300.5 150.0 50.0)")

    v_x = pts["X_Axis"] - pts["Orijin"]
    v_x = v_x / np.linalg.norm(v_x)

    v_y_temp = pts["Y_Axis"] - pts["Orijin"]
    v_z = np.cross(v_x, v_y_temp)
    v_z = v_z / np.linalg.norm(v_z)

    v_y = np.cross(v_z, v_x)

    H_board2robot = np.array([
        [v_x[0], v_y[0], pts["Orijin"][0]],
        [v_x[1], v_y[1], pts["Orijin"][1]],
        [0.0,    0.0,    1.0             ]
    ], dtype=np.float64)

    R_b2r = np.column_stack((v_x, v_y, v_z))
    t_b2r = pts["Orijin"].copy()

    return R_b2r, t_b2r, H_board2robot


# ---------------------------------------------
# 3. ERROR ANALYSIS
# ---------------------------------------------
def analyze_errors(pts_img, theoretical_robot_pts, H_pixel2robot):
    projected_robot = cv2.perspectiveTransform(
        pts_img.reshape(-1, 1, 2), H_pixel2robot
    ).reshape(-1, 2)

    errors_mm = np.linalg.norm(projected_robot - theoretical_robot_pts, axis=1)

    per_point = []
    for i in range(len(errors_mm)):
        per_point.append({
            "id":       i + 1,
            "px_x":     float(pts_img[i, 0]),
            "px_y":     float(pts_img[i, 1]),
            "world_x":  float(theoretical_robot_pts[i, 0]),
            "world_y":  float(theoretical_robot_pts[i, 1]),
            "error_mm": float(errors_mm[i])
        })
    return errors_mm, per_point


# ---------------------------------------------
# 4. IMAGE OUTPUTS
# ---------------------------------------------
def generate_pure_pixel_error_map(per_point, output_dir, img_width=1280, img_height=1024):
    px_x    = np.array([p["px_x"]     for p in per_point])
    px_y    = np.array([p["px_y"]     for p in per_point])
    errs_mm = np.array([p["error_mm"] for p in per_point])

    # Grid for 1280x1024
    xi = np.arange(0, img_width)
    yi = np.arange(0, img_height)
    grid_x, grid_y = np.meshgrid(xi, yi)

    # Spread the data with interpolation
    grid_z = griddata((px_x, px_y), errs_mm, (grid_x, grid_y), method='linear')
    nan_mask = np.isnan(grid_z)
    if np.any(nan_mask):
        grid_z_nearest = griddata((px_x, px_y), errs_mm, (grid_x, grid_y), method='nearest')
        grid_z[nan_mask] = grid_z_nearest[nan_mask]

    # --- SAVING DIRECTLY AS A NUMPY ARRAY INSTEAD OF A DICTIONARY ---
    # every pixel gets one value
    # The resulting matrix will be exactly (1024, 1280).
    with open(os.path.join(output_dir, "piksel_hata_haritasi.pkl"), "wb") as f:
        pickle.dump(grid_z.astype(np.float32), f)

    # --- PLAIN IMAGE (PNG) ---
    vmax_val = np.max(errs_mm) if np.max(errs_mm) > 0 else 1.0
    grid_z_clip = np.clip(grid_z, 0.0, vmax_val)
    grid_z_norm = ((grid_z_clip / vmax_val) * 255).astype(np.uint8)

    pure_heatmap_color = cv2.applyColorMap(grid_z_norm, cv2.COLORMAP_INFERNO)
    cv2.imwrite(
        os.path.join(output_dir, "Transformation_Residual_Continuous_Error.png"),
        pure_heatmap_color
    )


# ---------------------------------------------
# MAIN FLOW
# ---------------------------------------------
def main():
    print("\n" + "="*55)
    print("  CALIBRATION PHASE 2: REAL-WORLD (3D) INTEGRATION")
    print("="*55 + "\n")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load the camera and find the points
    img, mtx, dist = load_and_undistort_image(IMAGE_PATH, CALIB_PATH)
    circles = detect_dots(img)
    points, num_rows, num_cols = sort_points(circles, row_threshold=ROW_THRESHOLD)

    expected_count = num_rows * num_cols
    if len(points) != expected_count:
        raise RuntimeError(
            f"The grid has missing or extra points! "
            f"Beklenen: {expected_count} ({num_rows}x{num_cols}), "
            f"Detected: {len(points)}. "
            f"Check ROW_THRESHOLD or the circularity threshold."
        )

    pts_img    = np.array(points, dtype=np.float32)
    obj_points = generate_object_points(num_rows, num_cols)

    vis = img.copy()
    for idx, (x, y) in enumerate(points):
        cv2.circle(vis, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(vis, str(idx + 1), (x + 6, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    cv2.imwrite(DETECTED_POINTS_PATH, vis)

    # 2. The camera solves the world (SolvePnP)
    ret, rvec, tvec = cv2.solvePnP(obj_points, pts_img, mtx, dist)
    if not ret:
        raise RuntimeError("SolvePnP failed! The pixels or the spacing values are wrong.")

    # 3. H_pixel2board: pixel -> board (mm) transformation
    board_pts_2d = obj_points[:, :2].reshape(-1, 1, 2)   
    pixel_pts_2d = pts_img.reshape(-1, 1, 2)             

    H_pixel2board, hmask = cv2.findHomography(pixel_pts_2d, board_pts_2d, cv2.RANSAC, 3.0)
    if H_pixel2board is None:
        raise RuntimeError("findHomography failed! The detected points are insufficient or wrong.")

    # 4. The robot solves the world (hand-eye)
    _, _, H_board2robot = get_robot_reference_frame()

    # 5. BRIDGE MATH: building the H matrix
    H_pixel2robot = H_board2robot @ H_pixel2board
    H_pixel2robot = H_pixel2robot / H_pixel2robot[2, 2]

    np.save(os.path.join(OUTPUT_DIR, "H.npy"), H_pixel2robot)
    print("\n  [OK] 3D space merged. Backward-compatible 'H.npy' matrix created!")

    # 6. Theoretical robot coordinates and error computation
    theoretical_robot_pts = []
    for p_obj in obj_points:
        p_b = np.array([p_obj[0], p_obj[1], 1.0])
        p_r = H_board2robot @ p_b
        theoretical_robot_pts.append([p_r[0] / p_r[2], p_r[1] / p_r[2]])
    theoretical_robot_pts = np.array(theoretical_robot_pts, dtype=np.float32)

    errors_mm, per_point = analyze_errors(pts_img, theoretical_robot_pts, H_pixel2robot)

    print("\n  Producing the target outputs...")
    # img_height is 1024
    generate_pure_pixel_error_map(per_point, OUTPUT_DIR, img_width=1280, img_height=1024)
    print("  -> [OK] Transformation_Residual_Continuous_Error.png & piksel_hata_haritasi.pkl")

    print(f"\n  Process completed successfully. Outputs are in the folder: {OUTPUT_DIR}\n")


if __name__ == "__main__":
    main()