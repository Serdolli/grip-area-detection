# For every test: is the suction point of CIELAB-AMF inside the part of the method's own crop that really comes from the photograph?
# method_2 crop_and_rotate rotates inside a square around the part; when that square is cut by the photo border the rotated part
# does not fit and getRectSubPix fills the missing columns by repeating the last pixels. A point chosen there is not on the part.
import os, csv, importlib.util
import cv2, numpy as np
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # evaluation/suction_feasibility
spec = importlib.util.spec_from_file_location("draw", os.path.join(HERE, "5_draw_test_results.py"))
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
rt = mod.rt; P, yb = rt.P, rt.yb

photos = list(csv.DictReader(open(mod.TEST_LIST_CSV, encoding="utf-8-sig")))
saved = {(int(r["test"]), r["method"]): r for r in csv.DictReader(open(mod.PARTS_CSV, encoding="utf-8-sig"))}

def crop_geometry(contour, img_shape):
    # same numbers as method_2 crop_and_rotate
    center, size, angle = cv2.minAreaRect(contour)
    if size[0] < size[1]:
        size = (size[1], size[0])
    diagonal = int(np.sqrt(size[0] ** 2 + size[1] ** 2)) + 10
    rx, ry = int(center[0] - diagonal // 2), int(center[1] - diagonal // 2)
    x1, y1 = max(0, rx), max(0, ry)
    x2, y2 = min(img_shape[1], rx + diagonal), min(img_shape[0], ry + diagonal)
    cx, cy = center[0] - x1, center[1] - y1
    w, h = int(size[0]), int(size[1])
    left, top = cx - (w - 1) / 2.0, cy - (h - 1) / 2.0            # crop pixel (0, 0) in the rotated canvas
    return (x2 - x1, y2 - y1), (left, top), (w, h), (x1 > rx or y1 > ry or x2 < rx + diagonal or y2 < ry + diagonal)

test_no, cut_parts, bad = 0, 0, []
for row in photos:
    img = cv2.imread(row["foto_yolu"])
    if img is None: continue
    h_orig, w_orig = img.shape[:2]
    full_gt, _ = rt.load_labels(row)
    gray = cv2.cvtColor(cv2.convertScaleAbs(img, alpha=1.0, beta=P['brightness_beta']), cv2.COLOR_BGR2GRAY)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for part_no, cnt in enumerate(yb["shed_filter"](contours, w_orig, h_orig), start=1):
        img_crop, gt_crop = yb["crop_and_rotate_perfect"](img, full_gt, cnt)
        _, gt_input = yb["prepare_inputs_perfect"](img_crop, gt_crop)
        pred, gt = rt.cielab_part_masks(img, full_gt, cnt)
        if rt.mic(gt_input)[1] <= 0 or rt.mic(gt)[1] <= 0:      # no labelled grip area on this part
            continue
        test_no += 1
        s = saved[(test_no, "CIELAB-AMF")]
        assert s["photo"] == row["etiket_dosyasi"].split(".rf.")[0] and int(s["part"]) == part_no, (test_no, s["photo"], part_no)
        m = rt.measure(pred, gt)
        d_mm = rt.to_mm(m)[2]
        assert (s["point_to_label_edge_mm"] == "" and d_mm is None) or abs(float(s["point_to_label_edge_mm"]) - d_mm) < 0.001
        (cw, ch), (left, top), (w, h), cut = crop_geometry(cnt, img.shape)
        if cut: cut_parts += 1
        if m["center"] is None: continue
        px, py = m["center"][0] + left, m["center"][1] + top     # the point in the rotated canvas
        if not (0 <= px <= cw - 1 and 0 <= py <= ch - 1):
            bad.append((test_no, row["kaynak"], s["photo"], part_no, d_mm, round(px), round(py), cw, ch))
print("tests:", test_no, "| parts whose crop square is cut by the photo border:", cut_parts)
print("CIELAB-AMF points lying in the repeated (not real) part of the crop:", len(bad))
for b in bad:
    print("   test %d | %s | %s part %d | d = %.2f mm | canvas point (%d, %d) of %d x %d" % b)
