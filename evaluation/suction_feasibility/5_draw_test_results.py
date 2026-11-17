# !pip install ultralytics pycocotools opencv-python numpy torch torchvision pillow
#
# STEP 5: Draws every test image WITH its measurement (visual evidence for the theoretical rates in Table 2).
#
#   The image itself:
#     MAT-GOF, CIELAB-AMF : black-and-white mask (as in the paper)
#     HM R-CNN, YOLOv8-seg: part photo + mask (blue)
#   Only three things are drawn on it, all green:
#     point  : vacuum point chosen by the method (MIC center of its own mask)
#     line   : from the point to the nearest edge of the LABELED grip area
#     text   : length of this line (mm). Negative if the point is outside the label.
#   If the method found no area, the image has no point, line or text.
#
#   The measurement is redone with the functions of 2_run_tests.py and compared with grasp_success_rate_parcalar.csv.
#   The value from the CSV (used in the paper) is written on the image; if the re-measurement differs, it is printed on screen.
#   In the CIELAB-AMF tests re-measured by 2b_fix_cielab_points.py (corrected = 1 in the CSV) the line goes to the real label.
#
#   Output: results\tests\matgof|cielab|hmrcnn|yolo\<method>_test_N.png / .jpg   all tests (342)
#          Condition by test number: 1-285 D7000 black background, 286-328 D7000 gray background, 329-342 Cognex.
#   The test number is the same as in the CSV and Excel; the same number is the same part for all four methods.

import os
import csv
import importlib.util
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ==============================================================================
# 1. PATHS AND SETTINGS
# ==============================================================================
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
TEST_LIST_CSV = os.path.join(RESULTS_DIR, "test_listesi.csv")
PARTS_CSV = os.path.join(RESULTS_DIR, "grasp_success_rate_parcalar.csv")
TESTS_DIR = os.path.join(RESULTS_DIR, "tests")

IMAGE_WIDTH = 1600                 # the part lies horizontally
BINARY_METHODS = ["MAT-GOF", "CIELAB-AMF"]     # their image is a black-and-white mask
GREEN = (0, 215, 0)                # RGB: point, line and length text
MASK_TINT = (0, 170, 255)          # RGB: mask color in the HM R-CNN and YOLO images (blue)
FONT = r"C:\Windows\Fonts\arialbd.ttf"
FONT_SIZE = 30

# loads 2_run_tests.py as a module (the models and method functions come from there; run() is not executed)
_spec = importlib.util.spec_from_file_location("run_tests", os.path.join(HERE, "2_run_tests.py"))
rt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rt)

font = ImageFont.truetype(FONT, FONT_SIZE)

# ==============================================================================
# 2. GEOMETRY
# ==============================================================================
def nearest_label_edge(gt_mask, center):
    # In the measurement space, the label edge pixel nearest to the point.
    # If the point is inside the label: the nearest pixel outside the label; if outside: the nearest label pixel (same definition as measure()).
    padded = cv2.copyMakeBorder((gt_mask > 0).astype(np.uint8), 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    px, py = center[0] + 1, center[1] + 1
    ys, xs = np.nonzero(padded == 0) if padded[py, px] > 0 else np.nonzero(padded > 0)
    i = int(np.argmin((xs - px) ** 2 + (ys - py) ** 2))
    return (int(xs[i]) - 1, int(ys[i]) - 1)

def horizontal(mask_or_img, out_size, is_mask):
    # Turns the vertical crop horizontal and scales it to the output size
    turned = cv2.rotate(mask_or_img, cv2.ROTATE_90_CLOCKWISE)
    if is_mask:
        return cv2.resize((turned > 0).astype(np.uint8) * 255, out_size, interpolation=cv2.INTER_LINEAR) > 127
    return cv2.resize(turned, out_size, interpolation=cv2.INTER_CUBIC if out_size[0] >= turned.shape[1] else cv2.INTER_AREA)

# ==============================================================================
# 3. DRAWING
# ==============================================================================
def draw_test(path, method, distance_mm, img_crop, pred_crop, point_crop, edge_crop):
    # All given in the vertical crop of the part: pred_crop mask, point_crop / edge_crop (x, y) or None.
    h, w = img_crop.shape[:2]
    f = IMAGE_WIDTH / h
    out_size = (IMAGE_WIDTH, max(1, int(round(w * f))))
    to_view = lambda p: (int(round((h - 1 - p[1]) * f)), int(round(p[0] * f)))      # vertical crop -> horizontal image
    mask = horizontal(pred_crop, out_size, is_mask=True)

    if method in BINARY_METHODS:
        vis = np.zeros((out_size[1], out_size[0], 3), dtype=np.uint8)
        vis[mask] = 255
    else:
        gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
        alpha = float(np.clip(230.0 / max(np.percentile(gray, 99.5), 1.0), 1.0, 4.0))     # make dark photos visible
        vis = horizontal(cv2.convertScaleAbs(img_crop, alpha=alpha, beta=0), out_size, is_mask=False)
        vis[mask] = (0.6 * vis[mask] + 0.4 * np.array(MASK_TINT[::-1])).astype(np.uint8)
        edges, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(vis, edges, -1, MASK_TINT[::-1], 1, cv2.LINE_AA)

    if point_crop is not None:
        p, e = to_view(point_crop), to_view(edge_crop)
        cv2.line(vis, p, e, (0, 0, 0), 7, cv2.LINE_AA)            # black outline: so it is visible on a white mask too
        cv2.line(vis, p, e, GREEN[::-1], 3, cv2.LINE_AA)
        cv2.circle(vis, p, 9, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(vis, p, 7, GREEN[::-1], -1, cv2.LINE_AA)

    canvas = Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    if point_crop is not None:                                      # length of the line, next to the point
        draw = ImageDraw.Draw(canvas)
        label = "%.2f mm" % distance_mm
        tw = draw.textlength(label, font=font)
        dx, dy = e[0] - p[0], e[1] - p[1]
        if abs(dx) > abs(dy):                                       # horizontal line: text above the point (below if there is no room)
            tx = p[0] - tw / 2
            ty = p[1] - FONT_SIZE - 18 if p[1] - FONT_SIZE - 18 >= 4 else p[1] + 14
        else:                                                       # vertical line: text on the side the line does not go to
            tx = p[0] + 22 if dx <= 0 else p[0] - 22 - tw
            if tx < 4 or tx + tw + 4 > out_size[0]:
                tx = p[0] - 22 - tw if dx <= 0 else p[0] + 22
            ty = p[1] - FONT_SIZE / 2 - 2
        tx = int(np.clip(tx, 4, max(4, out_size[0] - tw - 6)))
        ty = int(np.clip(ty, 4, max(4, out_size[1] - FONT_SIZE - 10)))
        draw.text((tx, ty), label, font=font, fill=GREEN, stroke_width=3, stroke_fill=(0, 0, 0))

    if path.lower().endswith(".jpg"):
        canvas.save(path, quality=92)
    else:
        canvas.save(path)

# ==============================================================================
# 4. MAIN LOOP
# ==============================================================================
def run():
    with open(TEST_LIST_CSV, encoding="utf-8-sig") as f:
        photos = list(csv.DictReader(f))
    with open(PARTS_CSV, encoding="utf-8-sig") as f:
        measured = {(int(r["test"]), r["method"]): r for r in csv.DictReader(f)}

    for folder in rt.FOLDERS.values():
        os.makedirs(os.path.join(TESTS_DIR, folder), exist_ok=True)
        for old in os.listdir(os.path.join(TESTS_DIR, folder)):      # previous images are deleted so numbers do not get mixed up
            if "_test_" in old and old.lower().endswith((".png", ".jpg")):
                os.remove(os.path.join(TESTS_DIR, folder, old))

    P, yb = rt.P, rt.yb
    test_no, written, differences = 0, 0, []
    for row in photos:
        img_orig = cv2.imread(row["foto_yolu"])
        if img_orig is None:
            print(f"   ! could not read: {row['foto_yolu']}")
            continue
        h_orig, w_orig = img_orig.shape[:2]
        full_gt_mask, _ = rt.load_labels(row)
        photo_name = row["etiket_dosyasi"].split(".rf.")[0]

        # Parts and methods: same order as 2_run_tests.py -> run()
        img_bright = cv2.convertScaleAbs(img_orig, alpha=1.0, beta=P['brightness_beta'])
        gray = cv2.cvtColor(img_bright, cv2.COLOR_BGR2GRAY)
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_cnts = yb["shed_filter"](contours, w_orig, h_orig)
        if not valid_cnts:
            continue
        matgof_mask = rt.matgof_full_mask(img_orig)
        yolo_mask = rt.yolo_full_mask(img_orig)

        first = test_no + 1
        for part_no, cnt in enumerate(valid_cnts, start=1):
            img_crop, gt_crop = yb["crop_and_rotate_perfect"](img_orig, full_gt_mask, cnt)
            img_input, gt_input = yb["prepare_inputs_perfect"](img_crop, gt_crop)
            matgof_crop, matgof_input = rt.to_canvas(img_orig, matgof_mask, cnt, img_crop)
            yolo_crop, yolo_input = rt.to_canvas(img_orig, yolo_mask, cnt, img_crop)
            hm_input = rt.hm_rcnn_canvas_mask(img_input)
            cielab_pred, cielab_gt = rt.cielab_part_masks(img_orig, full_gt_mask, cnt)

            # (predicted mask, label mask) in the measurement space; the first three on the 160x1600 canvas, CIELAB-AMF on its own crop
            spaces = {"MAT-GOF": (matgof_input, gt_input), "CIELAB-AMF": (cielab_pred, cielab_gt),
                      "HM R-CNN": (hm_input, gt_input), "YOLOv8s-seg": (yolo_input, gt_input)}
            results = {method: rt.measure(pred, gt) for method, (pred, gt) in spaces.items()}
            if any(m is None for m in results.values()):   # no labeled grip area for this part
                continue
            test_no += 1

            A = rt.cielab_crop_to_part_crop(cnt, img_orig.shape)          # CIELAB crop -> shared crop
            size = (img_crop.shape[1], img_crop.shape[0])
            pred_crops = {"MAT-GOF": matgof_crop, "YOLOv8s-seg": yolo_crop,
                          "HM R-CNN": rt.canvas_mask_to_crop(hm_input, img_crop.shape),
                          "CIELAB-AMF": cv2.warpAffine(cielab_pred, A, size, flags=cv2.INTER_NEAREST)}

            for method in rt.METHODS:
                m = results[method]
                _, _, distance_mm = rt.to_mm(m)
                saved = measured[(test_no, method)]
                saved_mm = None if saved["point_to_label_edge_mm"] == "" else float(saved["point_to_label_edge_mm"])
                corrected = saved.get("corrected") == "1"      # re-measured against the real label by 2b_fix_cielab_points.py
                same = (distance_mm is None and saved_mm is None) or (distance_mm is not None and saved_mm is not None and abs(distance_mm - saved_mm) < 0.001)
                if (not same and not corrected) or saved["photo"] != photo_name or int(saved["part"]) != part_no:
                    differences.append((test_no, method, saved_mm, distance_mm))

                point_crop = edge_crop = None
                if m["center"] is not None:
                    edge = nearest_label_edge(spaces[method][1], m["center"])
                    if method == "CIELAB-AMF":
                        point_crop = tuple(A @ np.array([m["center"][0], m["center"][1], 1.0]))
                        edge_crop = tuple(A @ np.array([edge[0], edge[1], 1.0]))
                        if corrected:                                 # the line also goes to the nearest edge of the real label (shared crop)
                            edge_crop = nearest_label_edge(gt_crop, (int(round(point_crop[0])), int(round(point_crop[1]))))
                    else:
                        point_crop, _ = rt.canvas_point_to_crop(m["center"], 0, img_crop.shape)
                        edge_crop, _ = rt.canvas_point_to_crop(edge, 0, img_crop.shape)

                folder = rt.FOLDERS[method]
                extension = "png" if method in BINARY_METHODS else "jpg"
                draw_test(os.path.join(TESTS_DIR, folder, f"{folder}_test_{test_no}.{extension}"), method, saved_mm,
                          img_crop, pred_crops[method], point_crop, edge_crop)
                written += 1
        print(f"   photo {int(row['foto_no']):>3}/{len(photos)} | {photo_name[:34]:<34} | tests {first}-{test_no}" if test_no >= first else f"   photo {int(row['foto_no']):>3}/{len(photos)} | {photo_name[:34]:<34} | no tests", flush=True)

    print("=" * 96)
    print(f"[OK] {written} images written ({test_no} tests x {len(rt.METHODS)} methods) -> {TESTS_DIR}")
    print(f"   Re-measurements that differ from the CSV : {len(differences)}")
    for d in differences[:20]:
        print(f"      test {d[0]} | {d[1]} | CSV: {d[2]} | now: {d[3]}")

if __name__ == "__main__":
    run()
