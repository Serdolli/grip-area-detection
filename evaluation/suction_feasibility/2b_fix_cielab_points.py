# !pip install ultralytics pycocotools opencv-python numpy torch torchvision
#
# STEP 2b: Re-measures the CIELAB-AMF points chosen in the FAKE part of the crop against the real label.
#
#   Problem : method_2 crop_and_rotate rotates the part inside a square around it. If this square goes past the photo edge,
#             the rotated part does not fit into the square and getRectSubPix fills the missing columns
#             by repeating the last pixels. If the method's point falls into this repeated (not real) region, 2_run_tests.py
#             counts it as "successful" because it measures it against the label in the same (equally distorted) crop.
#   Fix: such points are moved to the shared part crop and measured against the REAL label there
#             (scale: label width in the same crop = 9.2 mm). The method's point does not change, only its measurement.
#   Result  : the point was in the repeated region in 8 of 342 tests; the result changed for 6 of them
#             (38, 85, 171, 176, 237 and 317 on the gray background: success -> failure), 2 of them (117, 139) stayed successful.
#
#   Input/output: results\grasp_success_rate_parcalar.csv is updated in place.
#                The original is kept as results\grasp_success_rate_parcalar_duzeltme_oncesi.csv (only on the first run).
#                New columns: corrected (1 = re-measured by this script), point_to_label_edge_mm_first (first measurement).
#   The other three methods are not affected: their crop (crop_and_rotate_perfect) leaves missing areas black and keeps the geometry.

import os
import csv
import shutil
import importlib.util
import cv2
import numpy as np

# ==============================================================================
# 1. PATHS
# ==============================================================================
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
TEST_LIST_CSV = os.path.join(RESULTS_DIR, "test_listesi.csv")
PARTS_CSV = os.path.join(RESULTS_DIR, "grasp_success_rate_parcalar.csv")
FIRST_CSV = os.path.join(RESULTS_DIR, "grasp_success_rate_parcalar_duzeltme_oncesi.csv")
METHOD = "CIELAB-AMF"

_spec = importlib.util.spec_from_file_location("run_tests", os.path.join(HERE, "2_run_tests.py"))
rt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rt)

# ==============================================================================
# 2. GEOMETRY
# ==============================================================================
def crop_canvas(contour, img_shape):
    # Same numbers as method_2 crop_and_rotate: size of the rotation canvas and position of the crop's (0, 0) pixel on the canvas
    center, size, _ = cv2.minAreaRect(contour)
    if size[0] < size[1]:
        size = (size[1], size[0])
    diagonal = int(np.sqrt(size[0] ** 2 + size[1] ** 2)) + 10
    rx, ry = int(center[0] - diagonal // 2), int(center[1] - diagonal // 2)
    x1, y1 = max(0, rx), max(0, ry)
    x2, y2 = min(img_shape[1], rx + diagonal), min(img_shape[0], ry + diagonal)
    cx, cy = center[0] - x1, center[1] - y1
    return (x2 - x1, y2 - y1), (cx - (int(size[0]) - 1) / 2.0, cy - (int(size[1]) - 1) / 2.0)

def distance_to_real_label(gt_crop, point):
    # Distance (mm; negative if outside) from the point to the nearest edge of the real label in the shared part crop, and the scale
    x, y = int(round(point[0])), int(round(point[1]))
    _, gt_radius, gt_dist = rt.mic(gt_crop)
    mm_per_px = rt.GRIP_WIDTH_MM / (2.0 * gt_radius)
    inside = 0 <= x < gt_crop.shape[1] and 0 <= y < gt_crop.shape[0]
    if inside and gt_dist[y, x] > 0:
        d_px = float(gt_dist[y, x])
    else:
        outside = cv2.distanceTransform((gt_crop == 0).astype(np.uint8), cv2.DIST_L2, 5)
        xx, yy = int(np.clip(x, 0, gt_crop.shape[1] - 1)), int(np.clip(y, 0, gt_crop.shape[0] - 1))
        d_px = -(float(outside[yy, xx]) + float(np.hypot(x - xx, y - yy)))
    return d_px * mm_per_px, d_px, gt_radius, mm_per_px

# ==============================================================================
# 3. MAIN LOOP
# ==============================================================================
def run():
    if not os.path.exists(FIRST_CSV):
        shutil.copy2(PARTS_CSV, FIRST_CSV)
    with open(FIRST_CSV, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    by_key = {(int(r["test"]), r["method"]): r for r in rows}
    with open(TEST_LIST_CSV, encoding="utf-8-sig") as f:
        photos = list(csv.DictReader(f))
    for r in rows:
        r["corrected"] = 0
        r["point_to_label_edge_mm_first"] = r["point_to_label_edge_mm"]

    P, yb = rt.P, rt.yb
    test_no, changed = 0, []
    for row in photos:
        img_orig = cv2.imread(row["foto_yolu"])
        if img_orig is None: continue
        h_orig, w_orig = img_orig.shape[:2]
        full_gt_mask, _ = rt.load_labels(row)
        gray = cv2.cvtColor(cv2.convertScaleAbs(img_orig, alpha=1.0, beta=P['brightness_beta']), cv2.COLOR_BGR2GRAY)
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for part_no, cnt in enumerate(yb["shed_filter"](contours, w_orig, h_orig), start=1):
            img_crop, gt_crop = yb["crop_and_rotate_perfect"](img_orig, full_gt_mask, cnt)
            _, gt_input = yb["prepare_inputs_perfect"](img_crop, gt_crop)
            pred, gt = rt.cielab_part_masks(img_orig, full_gt_mask, cnt)
            if rt.mic(gt_input)[1] <= 0 or rt.mic(gt)[1] <= 0:      # no labeled grip area for this part (same exclusion as 2_run_tests.py)
                continue
            test_no += 1
            saved = by_key[(test_no, METHOD)]
            assert saved["photo"] == row["etiket_dosyasi"].split(".rf.")[0] and int(saved["part"]) == part_no, (test_no, saved["photo"], part_no)
            m = rt.measure(pred, gt)
            if m["center"] is None: continue
            (canvas_w, canvas_h), (left, top) = crop_canvas(cnt, img_orig.shape)
            px, py = m["center"][0] + left, m["center"][1] + top      # position of the point on the rotation canvas
            if 0 <= px <= canvas_w - 1 and 0 <= py <= canvas_h - 1:
                continue                                              # the point is inside the real image: the measurement is valid
            A = rt.cielab_crop_to_part_crop(cnt, img_orig.shape)
            point = A @ np.array([m["center"][0], m["center"][1], 1.0])
            d_mm, d_px, gt_radius, mm_per_px = distance_to_real_label(gt_crop, point)
            changed.append((test_no, saved["point_to_label_edge_mm"], d_mm))
            saved.update({"point_to_label_edge_mm": round(d_mm, 4), "point_to_label_edge_px": round(d_px, 4),
                          "label_mic_radius_px": round(gt_radius, 4), "mm_per_px": round(mm_per_px, 4), "corrected": 1})
        print(f"   photo {int(row['foto_no']):>3}/{len(photos)} | test {test_no}", flush=True)

    with open(PARTS_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("=" * 96)
    print(f"[OK] {PARTS_CSV}")
    print(f"   {METHOD} points chosen in the fake region: {len(changed)}")
    for t, old, new in changed:
        print(f"      test {t:>3} | first measurement {float(old):6.2f} mm | against the real label {new:7.2f} mm")

if __name__ == "__main__":
    run()
