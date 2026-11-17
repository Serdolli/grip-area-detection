# Rates per raw photo set (the rows of the new Table 1): every test photograph is assigned to its set by file name + image similarity.
import os, re, csv, collections, math
import cv2, numpy as np

PROJECT_ROOT = os.environ.get("GRIP_PROJECT_ROOT", os.path.abspath("project_root"))   # see README

S = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
ROOT = os.path.join(PROJECT_ROOT, r"datasets")
SKIP = ("maskrcnn_", "fastrcnn_", "blackbackground_withlabels", "cropped_datasets", "yolo_dataset", "verified_results", "derivative_datasets")
FIX = {38: 2.49, 85: -10.09, 171: -8.82, 176: -9.50, 237: -9.90, 317: -2.00}
METHODS = ["MAT-GOF", "CIELAB-AMF", "HM R-CNN", "YOLOv8s-seg"]

def setup_of(p):
    p = p.lower(); g = "grey" if "grey" in p else "black"
    if "controlledlighting_18_11_25" in p: return "D7000 white " + g
    if "bluelight_zoomed_03_12_25" in p: return "D7000 blue " + g
    if "cognex" in p: return "Cognex"
    if "mixed_17_07_25" in p: return "Sony"
    return "other"
def small(img, size=256):
    h, w = img.shape[:2]; s = size / max(h, w)
    out = np.zeros((size, size, 3), np.uint8); nw, nh = int(round(w * s)), int(round(h * s))
    out[(size - nh) // 2:(size - nh) // 2 + nh, (size - nw) // 2:(size - nw) // 2 + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    return out.astype(np.float32)

originals = collections.defaultdict(list)
for folder, _, files in os.walk(os.path.join(ROOT, "main_datasets")):
    if any(s in folder for s in SKIP): continue
    for f in files:
        if f.lower().endswith((".jpg", ".jpeg", ".bmp")): originals[f.lower()].append(os.path.join(folder, f))

photo_set = {}
for r in csv.DictReader(open(os.path.join(R, "test_listesi.csv"), encoding="utf-8-sig")):
    path = r["foto_yolu"]
    if "main_datasets" in path:
        photo_set[r["foto_no"]] = setup_of(path); continue
    base = r["etiket_dosyasi"].split(".rf.")[0]
    m = re.match(r"(.*)_(JPG|jpg|jpeg|bmp|png)$", base)
    cands = originals.get(("%s.%s" % (m.group(1), m.group(2))).lower(), []) if m else []
    if len(cands) == 1:
        photo_set[r["foto_no"]] = setup_of(cands[0]); continue
    ref = small(cv2.imread(path)); best = None
    for c in cands:
        d = float(np.abs(small(cv2.imread(c)) - ref).mean())
        if best is None or d < best[0]: best = (d, c)
    photo_set[r["foto_no"]] = setup_of(best[1]) if best and best[0] < 6 else "unknown (%s)" % (None if best is None else round(best[0], 1))

rows = list(csv.DictReader(open(os.path.join(R, "grasp_success_rate_parcalar.csv"), encoding="utf-8-sig")))
iou = {(int(r["test"]), r["method"]): float(r["iou_all"]) for r in csv.DictReader(open(os.path.join(S, "miou_same_tests.csv"), encoding="utf-8-sig"))}
d = {}; tset = {}; tphoto = {}
for r in rows:
    t = int(r["test"]); v = r["point_to_label_edge_mm"]
    d[(t, r["method"])] = None if v == "" else float(v)
    tset[t] = photo_set[r["photo_no"]]; tphoto[t] = r["photo_no"]
for t, v in FIX.items(): d[(t, "CIELAB-AMF")] = v
def wilson(k, n, z=1.959964):
    p = k / n; den = 1 + z * z / n; c = (p + z * z / (2 * n)) / den; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * (c - h), 100 * (c + h)
groups = collections.defaultdict(list)
for t, s in tset.items(): groups[s].append(t)
for s in sorted(groups):
    sel = groups[s]
    print("%-18s workpieces %3d | photographs %3d" % (s, len(sel), len(set(tphoto[t] for t in sel))))
    for m in METHODS:
        out = []
        for tol in (0.5, 1.0):
            k = sum(d[(t, m)] is not None and d[(t, m)] >= 2.5 + tol for t in sel); lo, hi = wilson(k, len(sel))
            out.append("%5.1f (%3d) [%4.1f, %5.1f]" % (100 * k / len(sel), k, lo, hi))
        print("     %-12s t=0.5: %s | t=1: %s | mIoU %.1f | no result %d" % (m, out[0], out[1], 100 * sum(iou[(t, m)] for t in sel) / len(sel), sum(d[(t, m)] is None for t in sel)))
print(collections.Counter(photo_set.values()))
