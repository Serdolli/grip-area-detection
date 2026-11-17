# mIoU of the four methods on exactly the workpieces of the grasp-success-rate tests (same photos, same part finder,
# same masks as 2_run_tests.py). Nothing is written to the project; the result goes to miou_same_tests.csv next to this file.
#   iou_all      : whole predicted mask inside the part against the labelled grip area
#   iou_selected : only the connected region that contains the suction point (the region the method "takes as the grip area")
# A workpiece without a prediction gets IoU 0.
import os, io, csv, sys
import numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(os.path.dirname(HERE), "2_run_tests.py")
src = io.open(SCRIPT, encoding="utf-8").read()
ns = {"__name__": "defs", "__file__": SCRIPT}
exec(compile(src[:src.index("# 7. MAIN LOOP")], SCRIPT, "exec"), ns)     # definitions + models, not the main loop
globals().update({k: v for k, v in ns.items() if not k.startswith("__")})

def iou(a, b):
    a = a > 0; b = b > 0
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum()) / u if u else 0.0

def selected_region(pred):
    # connected region of the prediction that contains the centre of the largest inscribed circle
    if not np.any(pred): return pred
    center, _, _ = mic(pred)
    n, lab = cv2.connectedComponents((pred > 0).astype(np.uint8), connectivity=8)
    k = lab[int(center[1]), int(center[0])]
    return (lab == k).astype(np.uint8) if k > 0 else pred

with open(TEST_LIST_CSV, encoding="utf-8-sig") as f:
    photos = list(csv.DictReader(f))
rows, test_no = [], 0
for row in photos:
    img_orig = cv2.imread(row["foto_yolu"])
    if img_orig is None: continue
    h_orig, w_orig = img_orig.shape[:2]
    full_gt_mask, label_count = load_labels(row)
    img_bright = cv2.convertScaleAbs(img_orig, alpha=1.0, beta=P['brightness_beta'])
    gray = cv2.cvtColor(img_bright, cv2.COLOR_BGR2GRAY)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_cnts = yb["shed_filter"](contours, w_orig, h_orig)
    if not valid_cnts: continue
    matgof_mask = matgof_full_mask(img_orig)
    yolo_mask = yolo_full_mask(img_orig)
    for part_no, cnt in enumerate(valid_cnts, start=1):
        img_crop, gt_crop = yb["crop_and_rotate_perfect"](img_orig, full_gt_mask, cnt)
        img_input, gt_input = yb["prepare_inputs_perfect"](img_crop, gt_crop)
        _, matgof_input = to_canvas(img_orig, matgof_mask, cnt, img_crop)
        _, yolo_input = to_canvas(img_orig, yolo_mask, cnt, img_crop)
        hm_input = hm_rcnn_canvas_mask(img_input)
        cielab_pred, cielab_gt = cielab_part_masks(img_orig, full_gt_mask, cnt)
        pairs = {"MAT-GOF": (matgof_input, gt_input), "CIELAB-AMF": (cielab_pred, cielab_gt), "HM R-CNN": (hm_input, gt_input), "YOLOv8s-seg": (yolo_input, gt_input)}
        if any(measure(p, g) is None for p, g in pairs.values()): continue      # no labelled grip area on this part (same rule as the tests)
        test_no += 1
        for method, (p, g) in pairs.items():
            rows.append({"test": test_no, "source": row["kaynak"], "photo_no": int(row["foto_no"]), "part": part_no, "method": method,
                         "found": int(bool(np.any(p))), "iou_all": round(iou(p, g), 5), "iou_selected": round(iou(selected_region(p), g), 5)})
    print("photo %3d/%d | tests so far %d" % (int(row["foto_no"]), len(photos), test_no), flush=True)

out = os.path.join(HERE, "miou_same_tests.csv")
with open(out, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("tests:", test_no, "->", out)
for name, sel in (("all sources", lambda r: True), ("without grey background", lambda r: "gri" not in r["source"].lower() and "grey" not in r["source"].lower())):
    print("==", name)
    for method in METHODS:
        s = [r for r in rows if r["method"] == method and sel(r)]
        f_ = [r for r in s if r["found"]]
        print("  %-12s n=%3d | mIoU all %.2f | mIoU selected %.2f | found %3d | mIoU selected, found only %.2f" % (
            method, len(s), 100 * np.mean([r["iou_all"] for r in s]), 100 * np.mean([r["iou_selected"] for r in s]), len(f_), 100 * np.mean([r["iou_selected"] for r in f_]) if f_ else 0))
print("sources:", sorted(set(r["source"] for r in rows)))
