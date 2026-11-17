# !pip install numpy opencv-python scipy matplotlib
#
# Check for Section 3.5 of the paper (no number of the paper comes from this script).
#
#   The paper describes the suction point as "the center of the largest inscribed circle via a Voronoi diagram".
#   All method scripts find the same point with the distance transform (cv2.distanceTransform):
#       method_1_sidechecker_debugged_v3.py   -> largest_inscribed_circles_per_parent()
#       method_2_gripareafinder_v4.py         -> get_max_inscribed_circle_info()
#       method_3_croppertester_for_model_3.py -> rl_mic()
#       2_run_tests.py (theoretical rate)     -> mic()
#   This script compares the two ways on the same labeled grip areas:
#       A) distance transform: the pixel inside the mask that is farthest from the edge
#       B) Voronoi           : among the Voronoi vertices of the edge points, the one inside the mask and farthest from the edge
#   Expected: both ways give the same circle RADIUS (difference below 1 px).
#   Note: on a straight strip the largest circle can slide along the strip, so the CENTER can
#        end up in different places, but the distance to the edge (radius) is the same. That is what matters for the theoretical rate.
#
#   Outputs: results\voronoi_kontrol.csv, results\voronoi_kontrol.png

import os
import json
import csv
import cv2
import numpy as np
from scipy.spatial import Voronoi, cKDTree
import matplotlib

PROJECT_ROOT = os.environ.get("GRIP_PROJECT_ROOT", os.path.abspath("project_root"))   # see README

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ==============================================================================
# 1. PATHS AND SETTINGS
# ==============================================================================
DATASET_DIR = os.path.join(PROJECT_ROOT, r"datasets\derivative_datasets\blackbackground_withlabels\test")
ANN_FILE = os.path.join(DATASET_DIR, "_annotations.coco.json")
GRIP_CLASS_NAME = "griparea"
BOUNDARY_STEP = 2          # edge sampling step (px); the paper's "edge is densely sampled"

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ==============================================================================
# 2. THE TWO WAYS
# ==============================================================================
def mic_distance_transform(mask):
    # Way used in the method scripts: 1 px border + distance transform + maximum value
    padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    dist = cv2.distanceTransform(padded, cv2.DIST_L2, 5)
    _, radius, _, center = cv2.minMaxLoc(dist)
    return (center[0] - 1, center[1] - 1), radius

def mic_voronoi(mask):
    # Way described in the paper: edge points -> Voronoi -> vertices inside the mask -> vertex farthest from the edge
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    boundary = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)[::BOUNDARY_STEP]
    vertices = Voronoi(boundary).vertices
    h, w = mask.shape
    xi, yi = np.round(vertices[:, 0]).astype(int), np.round(vertices[:, 1]).astype(int)
    inside = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    inside[inside] &= mask[yi[inside], xi[inside]] > 0
    vertices = vertices[inside]
    distances, _ = cKDTree(boundary).query(vertices)
    best = int(np.argmax(distances))
    return (vertices[best, 0], vertices[best, 1]), float(distances[best]), boundary, vertices

# ==============================================================================
# 3. MAIN LOOP
# ==============================================================================
def run():
    with open(ANN_FILE, encoding="utf-8") as f:
        coco = json.load(f)
    grip_id = [c["id"] for c in coco["categories"] if c["name"] == GRIP_CLASS_NAME][0]
    images = {im["id"]: im for im in coco["images"]}

    rows, example = [], None
    for ann in coco["annotations"]:
        if ann["category_id"] != grip_id or not isinstance(ann["segmentation"], list): continue
        im = images[ann["image_id"]]
        x, y, w, h = [int(v) for v in ann["bbox"]]
        x0, y0 = max(0, x - 5), max(0, y - 5)
        mask = np.zeros((h + 12, w + 12), dtype=np.uint8)        # only the perimeter of the label (for speed)
        for poly in ann["segmentation"]:
            pts = np.array(poly, dtype=np.float64).reshape(-1, 2) - [x0, y0]
            cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 1)
        if mask.sum() < 500: continue

        c_dt, r_dt = mic_distance_transform(mask)
        c_vo, r_vo, boundary, vertices = mic_voronoi(mask)
        # The edge distance of the Voronoi center is also read from the distance transform map (cross-check)
        rows.append({"photo": im["file_name"].split(".rf.")[0], "label_id": ann["id"],
                     "radius_distance_transform_px": round(r_dt, 3), "radius_voronoi_px": round(r_vo, 3),
                     "radius_difference_px": round(abs(r_dt - r_vo), 3),
                     "center_shift_px": round(float(np.hypot(c_dt[0] - c_vo[0], c_dt[1] - c_vo[1])), 1)})
        if example is None and w > 2 * h or example is None and h > 2 * w:
            example = (mask, boundary, vertices, c_dt, r_dt, c_vo, r_vo)

    with open(os.path.join(RESULTS_DIR, "voronoi_kontrol.csv"), "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    diff = np.array([r["radius_difference_px"] for r in rows])
    rad = np.array([r["radius_distance_transform_px"] for r in rows])
    print("=" * 78)
    print(f"VORONOI CHECK | labeled grip areas: {len(rows)}")
    print("=" * 78)
    print(f"Radius (distance transform) mean    : {rad.mean():.1f} px")
    print(f"Radius difference mean / max       : {diff.mean():.3f} px / {diff.max():.3f} px")
    print(f"Radius difference as % of radius    : mean %{100 * (diff / rad).mean():.2f}, max %{100 * (diff / rad).max():.2f}")
    print(f"Center shift (along the strip)       : median {np.median([r['center_shift_px'] for r in rows]):.0f} px")
    print(f"[OK] {os.path.join(RESULTS_DIR, 'voronoi_kontrol.csv')}")

    if example is not None:
        mask, boundary, vertices, c_dt, r_dt, c_vo, r_vo = example
        if mask.shape[0] > mask.shape[1]:        # make the strip horizontal
            mask = mask.T; boundary = boundary[:, ::-1]; vertices = vertices[:, ::-1]; c_dt = c_dt[::-1]; c_vo = c_vo[::-1]
        fig, ax = plt.subplots(figsize=(12, 2.6), dpi=200)
        ax.imshow(mask, cmap="gray", vmin=0, vmax=1.6)
        ax.plot(vertices[:, 0], vertices[:, 1], ".", color="#1f77b4", markersize=1.5, label="Voronoi vertices (inside the mask)")
        ax.add_patch(plt.Circle(c_dt, r_dt, fill=False, color="#d62728", linewidth=1.6, label="distance transform"))
        ax.add_patch(plt.Circle(c_vo, r_vo, fill=False, color="#2ca02c", linewidth=1.6, linestyle="--", label="Voronoi"))
        ax.plot(*c_dt, "o", color="#d62728", markersize=3); ax.plot(*c_vo, "o", color="#2ca02c", markersize=3)
        ax.set_axis_off(); ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3, frameon=False, fontsize=8)
        ax.set_title(f"Same label, two ways: radius {r_dt:.1f} px and {r_vo:.1f} px", fontsize=9)
        fig.savefig(os.path.join(RESULTS_DIR, "voronoi_kontrol.png"), bbox_inches="tight")
        print(f"[OK] {os.path.join(RESULTS_DIR, 'voronoi_kontrol.png')}")

if __name__ == "__main__":
    run()
