# For the eight CIELAB-AMF points that were chosen in the repeated part of the method's crop:
# where does the point land on the real part, and how far is it from the real label edge there?
import os, csv, importlib.util
import cv2, numpy as np
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # evaluation/suction_feasibility
spec = importlib.util.spec_from_file_location("draw", os.path.join(HERE, "5_draw_test_results.py"))
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
rt = mod.rt; P, yb = rt.P, rt.yb
CASES = [(38, "image_53_aaau_JPG", 4), (85, "image_29_auu_JPG", 3), (117, "image_56_aaaauu_JPG", 5), (139, "image_60_aaaaauu_JPG", 6),
         (171, "image_200_aauu_JPG", 5), (176, "image_201_aauu_JPG", 5), (237, "image_199_aaauu_JPG", 6), (317, "image_2_aau_JPG", 1)]
saved = {(int(r["test"]), r["method"]): r for r in csv.DictReader(open(mod.PARTS_CSV, encoding="utf-8-sig"))}
rows = list(csv.DictReader(open(mod.TEST_LIST_CSV, encoding="utf-8-sig")))
for test, photo, part in CASES:
    s = saved[(test, "CIELAB-AMF")]
    row = [r for r in rows if r["etiket_dosyasi"].split(".rf.")[0] == photo and r["veri_seti"] == s["dataset"] and r["split"] == s["split"]][0]
    img = cv2.imread(row["foto_yolu"]); h_orig, w_orig = img.shape[:2]
    full_gt, _ = rt.load_labels(row)
    gray = cv2.cvtColor(cv2.convertScaleAbs(img, alpha=1.0, beta=P['brightness_beta']), cv2.COLOR_BGR2GRAY)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = yb["shed_filter"](contours, w_orig, h_orig)[part - 1]
    img_crop, gt_crop = yb["crop_and_rotate_perfect"](img, full_gt, cnt)
    pred, gt = rt.cielab_part_masks(img, full_gt, cnt)
    m = rt.measure(pred, gt)
    assert abs(rt.to_mm(m)[2] - float(s["point_to_label_edge_mm"])) < 0.001
    A = rt.cielab_crop_to_part_crop(cnt, img.shape)
    p = A @ np.array([m["center"][0], m["center"][1], 1.0])
    x, y = int(round(p[0])), int(round(p[1]))
    _, gt_r, gt_dist = rt.mic(gt_crop)
    mm = rt.GRIP_WIDTH_MM / (2.0 * gt_r)
    inside_crop = 0 <= x < gt_crop.shape[1] and 0 <= y < gt_crop.shape[0]
    if inside_crop and gt_dist[y, x] > 0:
        d_true = float(gt_dist[y, x]) * mm
    else:
        outside = cv2.distanceTransform((gt_crop == 0).astype(np.uint8), cv2.DIST_L2, 5)
        xx, yy = int(np.clip(x, 0, gt_crop.shape[1] - 1)), int(np.clip(y, 0, gt_crop.shape[0] - 1))
        d_true = -(float(outside[yy, xx]) + float(np.hypot(x - xx, y - yy))) * mm
    print("test %3d | saved d = %5.2f mm | on the real part d = %6.2f mm | point in part crop (%d, %d) of %d x %d" %
          (test, float(s["point_to_label_edge_mm"]), d_true, x, y, gt_crop.shape[1], gt_crop.shape[0]))
