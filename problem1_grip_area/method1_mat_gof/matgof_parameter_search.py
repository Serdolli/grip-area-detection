# MAT-GOF parameter search on a small part of the TRAIN split (test and valid photos are never touched here).
# Uses the functions of method_1_sidechecker_debugged_v3.py unchanged; only its parameter values are varied.
# Objective: share of parts whose selected point lies >= REQUIRED_MM inside the labelled grip area, then mean IoU.
import ast, os, sys, time, json, csv, itertools
import cv2, numpy as np
from pycocotools.coco import COCO

PROJECT_ROOT = os.environ.get("GRIP_PROJECT_ROOT", os.path.abspath("project_root"))   # see README

CODES = os.path.join(PROJECT_ROOT, r"codes_results\conforance_codes")
M1 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "method_1_sidechecker_debugged_v3.py")
YOLO_SCRIPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "yolov8seg_baseline", "yolov8seg_fullframe_train_test.py")
DATA = os.path.join(PROJECT_ROOT, r"datasets\derivative_datasets\blackbackground_withlabels")
OUT = os.path.dirname(os.path.abspath(__file__))
GRIP_WIDTH_MM = 9.2
REQUIRED_MM = 3.5            # 5 mm cup + 1 mm tolerance
EVERY = 6                    # every 6th train photo -> 24 photos
TIME_LIMIT_S = float(sys.argv[1]) if len(sys.argv) > 1 else 1500

def load_definitions(path):
    src = open(path, encoding="utf-8").read(); ns = {"__name__": "x", "__file__": path}
    for node in ast.parse(src).body:
        keep = isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef))
        if isinstance(node, ast.Assign):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            keep = len(names) == len(node.targets) and all(n.isupper() and not n.startswith("RL_") for n in names) and "RL_" not in ast.get_source_segment(src, node)
        if keep:
            try: exec(compile(ast.Module([node], []), path, "exec"), ns)
            except ImportError: pass
    return ns
m1 = load_definitions(M1); m1["SAVE_INTERMEDIATE"] = False
yb = load_definitions(YOLO_SCRIPT); P = yb["PARAMS"]

NAMES = ["MAX_CONTOUR_AREA_PX", "COLOR", "CONTRAST_2", "BRIGHTNESS_2", "ADAPTIVE_BLOCKSIZE", "ADAPTIVE_C", "MED_K",
         "OPEN_K", "OPEN_ITERS", "DILATE_ITERS", "MIN_SPECKLE_AREA", "MIN_MAIN_AREA", "MERGE_GAP_PX"]
BASE = {k: m1[k] for k in NAMES}
GRID = {
    "MAX_CONTOUR_AREA_PX": [500000, 1000000, 3000000],
    "ADAPTIVE_BLOCKSIZE": [11, 23, 51, 101, 151, 201, 301, 401, 601],
    "ADAPTIVE_C": [-12, -8, -4, -2, 0, 0.4, 2, 4, 8, 12, 20],
    "CONTRAST_2": [0.6, 1.0, 1.4, 2.0, 3.0, 4.5],
    "BRIGHTNESS_2": [0.7, 1.0, 1.5, 2.5, 4.0, 6.0],
    "COLOR": [0, 15, 60, 128, 255],
    "MED_K": [3, 5, 9, 13, 21, 31, 45],
    "OPEN_K": [1, 3, 5, 9, 15, 25],
    "OPEN_ITERS": [0, 1, 2],
    "DILATE_ITERS": [0, 1, 2, 3],
    "MIN_SPECKLE_AREA": [3, 50, 500, 3000],
    "MIN_MAIN_AREA": [25, 500, 5000, 20000],
    "MERGE_GAP_PX": [0, 5, 10, 20, 40],
}

def set_params(p):
    for k, v in p.items(): m1[k] = v
    m1["MASK_BEFORE_TRESHOLD_COLOR"] = (int(p["COLOR"]),) * 3
    for k in ("ADAPTIVE_BLOCKSIZE", "MED_K", "OPEN_K"):            # same normalisation as the script's own CLI overrides
        v = int(round(float(m1[k]))); m1[k] = v if v % 2 == 1 else v + 1
    if m1["OPEN_K"] <= 1: m1["OPEN_ITERS"] = 0

def mic(mask):
    padded = cv2.copyMakeBorder((mask > 0).astype(np.uint8), 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    dist = cv2.distanceTransform(padded, cv2.DIST_L2, 5); _, r, _, c = cv2.minMaxLoc(dist)
    return (c[0] - 1, c[1] - 1), r, dist[1:-1, 1:-1]

# ---------------- tuning photos, parts and labels (computed once) ----------------
coco = COCO(os.path.join(DATA, "train", "_annotations.coco.json"))
gid = [c["id"] for c in coco.loadCats(coco.getCatIds()) if c["name"] == "griparea"][0]
photos = []
for iid in sorted(coco.getImgIds())[::EVERY]:
    name = coco.loadImgs(iid)[0]["file_name"]; img = cv2.imread(os.path.join(DATA, "train", name))
    if img is None: continue
    h, w = img.shape[:2]
    gray = cv2.cvtColor(cv2.convertScaleAbs(img, alpha=1.0, beta=P["brightness_beta"]), cv2.COLOR_BGR2GRAY)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bench = yb["shed_filter"](cnts, w, h)
    gt = np.zeros((h, w), np.uint8)
    for a in coco.loadAnns(coco.getAnnIds(imgIds=iid, catIds=[gid])): gt = np.maximum(gt, coco.annToMask(a))
    parts = []
    for c in bench:
        img_crop, gt_crop = yb["crop_and_rotate_perfect"](img, gt, c)
        _, gt_input = yb["prepare_inputs_perfect"](img_crop, gt_crop)
        _, gr, gd = mic(gt_input)
        if gr > 0: parts.append({"cnt": c, "img_crop": img_crop, "gt": gt_input, "gt_r": gr, "gt_dist": gd})
    step1 = m1["otsu_treshold_filtering"]("x", img, save=False)[0]          # MAT-GOF's own detection (paper step 1)
    photos.append({"name": name, "img": img, "parts": parts, "step1": step1})
N_PARTS = sum(len(p["parts"]) for p in photos)
print("tuning set: %d train photos, %d parts" % (len(photos), N_PARTS), flush=True)

_gray_cache = {"key": None, "data": None}
def masked_grays(p):
    key = (p["MAX_CONTOUR_AREA_PX"], p["COLOR"])
    if _gray_cache["key"] != key:
        data = []
        for ph in photos:
            kept, _, _ = m1["shed_filter"](ph["step1"], ph["img"].shape)
            mg = m1["backgroundremover_1"](ph["img"], kept, "x", None, m1["MASK_BEFORE_TRESHOLD_COLOR"])
            data.append((kept, cv2.cvtColor(mg, cv2.COLOR_BGR2GRAY)))
        _gray_cache.update(key=key, data=data)
    return _gray_cache["data"]

def matgof_mask_roi(ph, kept, gray):
    # steps 4-6 of the method, run on a window around each kept part (same result as on the full frame)
    h, w = gray.shape; pred = np.zeros((h, w), np.uint8)
    margin = m1["ADAPTIVE_BLOCKSIZE"] // 2 + max(m1["MED_K"], m1["MERGE_GAP_PX"], m1["OPEN_K"] * 3) + 24
    for c in kept:
        x, y, bw, bh = cv2.boundingRect(c)
        x0, y0, x1, y1 = max(0, x - margin), max(0, y - margin), min(w, x + bw + margin), min(h, y + bh + margin)
        roi_kept = [k - np.array([x0, y0]) for k in kept if cv2.boundingRect(k)[0] < x1 and cv2.boundingRect(k)[0] + cv2.boundingRect(k)[2] > x0 and cv2.boundingRect(k)[1] < y1 and cv2.boundingRect(k)[1] + cv2.boundingRect(k)[3] > y0]
        thr = m1["adaptivethreshold"](gray[y0:y1, x0:x1], None, "x", None)
        white = m1["backgroundremover_2"](thr, roi_kept, "x", None)
        final, _, _, _ = m1["segment_and_merge_from_gray"](white, "x", None)
        rm = np.zeros((y1 - y0, x1 - x0), np.uint8); cv2.drawContours(rm, final, -1, 1, -1)
        own = np.zeros_like(rm); cv2.drawContours(own, [c - np.array([x0, y0])], -1, 1, -1)
        pred[y0:y1, x0:x1] |= (rm & own)
    return pred

def matgof_mask_full(ph, kept, gray):
    thr = m1["adaptivethreshold"](gray, None, "x", None)
    white = m1["backgroundremover_2"](thr, kept, "x", None)
    final, _, _, _ = m1["segment_and_merge_from_gray"](white, "x", None)
    pred = np.zeros(gray.shape, np.uint8); cv2.drawContours(pred, final, -1, 1, -1)
    return pred

def evaluate(p, mask_fn=matgof_mask_roi):
    set_params(p)
    dists, ious, nores = [], [], 0
    for ph, (kept, gray) in zip(photos, masked_grays(p)):
        pred = mask_fn(ph, kept, gray) if kept else np.zeros(gray.shape, np.uint8)
        for part in ph["parts"]:
            _, pc = yb["crop_and_rotate_perfect"](ph["img"], pred, part["cnt"])
            _, pi = yb["prepare_inputs_perfect"](part["img_crop"], pc)
            inter = np.logical_and(pi, part["gt"]).sum(); union = np.logical_or(pi, part["gt"]).sum()
            ious.append(inter / union if union else 0.0)
            if not pi.any(): nores += 1; dists.append(None); continue
            c, r, _ = mic(pi); d = float(part["gt_dist"][c[1], c[0]])
            if d <= 0:
                od = cv2.distanceTransform((part["gt"] == 0).astype(np.uint8), cv2.DIST_L2, 5); d = -float(od[c[1], c[0]])
            dists.append(d * GRIP_WIDTH_MM / (2.0 * part["gt_r"]))
    v = np.array([d for d in dists if d is not None])
    ok = int((v >= REQUIRED_MM).sum()) if len(v) else 0
    return {"success": ok / N_PARTS, "ok": ok, "miou": float(np.mean(ious)), "no_result": nores, "outside": int((v < 0).sum()) if len(v) else 0,
            "inside_1mm": int((v >= 1.0).sum()) if len(v) else 0, "median_mm": float(np.median(v)) if len(v) else float("nan")}

def score(r): return (r["success"], r["miou"])

# ---------------- check that the fast window path equals the full-frame path ----------------
set_params(BASE); t = time.time(); rb = evaluate(BASE); t_roi = time.time() - t
chk = []
for ph, (kept, gray) in list(zip(photos, masked_grays(BASE)))[:3]:
    a, b = matgof_mask_roi(ph, kept, gray), matgof_mask_full(ph, kept, gray)
    u = np.logical_or(a, b).sum(); chk.append(1.0 if u == 0 else np.logical_and(a, b).sum() / u)
print("window path vs full-frame path, mask IoU on 3 photos:", [round(float(x), 4) for x in chk], "| one evaluation: %.1f s" % t_roi, flush=True)
print("BASELINE (script values) %s" % json.dumps(BASE), flush=True)
print("BASELINE result: success@%.1fmm %.1f %% (%d/%d) | mIoU %.1f %% | no result %d | outside %d | >=1mm inside %d | median %.2f mm" % (REQUIRED_MM, 100 * rb["success"], rb["ok"], N_PARTS, 100 * rb["miou"], rb["no_result"], rb["outside"], rb["inside_1mm"], rb["median_mm"]), flush=True)

# ---------------- coordinate descent ----------------
log = open(os.path.join(OUT, "matgof_parameter_search_log.csv"), "w", newline="", encoding="utf-8"); wr = csv.writer(log)
wr.writerow(["trial", "round", "changed", "value"] + NAMES + ["success", "ok", "miou", "no_result", "outside", "inside_1mm", "median_mm"])
best, best_r, trial, T0 = dict(BASE), rb, 0, time.time()
ORDER = ["MAX_CONTOUR_AREA_PX", "ADAPTIVE_BLOCKSIZE", "ADAPTIVE_C", "CONTRAST_2", "BRIGHTNESS_2", "MED_K", "MERGE_GAP_PX", "MIN_MAIN_AREA",
         "OPEN_K", "OPEN_ITERS", "DILATE_ITERS", "COLOR", "MIN_SPECKLE_AREA"]
seen = {json.dumps(BASE, sort_keys=True): rb}
for rnd in range(1, 6):
    improved = False
    for name in ORDER:
        for val in GRID[name]:
            if time.time() - T0 > TIME_LIMIT_S: break
            cand = dict(best); cand[name] = val
            if name == "OPEN_K" and val > 1 and cand["OPEN_ITERS"] == 0: cand["OPEN_ITERS"] = 1
            key = json.dumps(cand, sort_keys=True)
            if key in seen: continue
            r = evaluate(cand); seen[key] = r; trial += 1
            wr.writerow([trial, rnd, name, val] + [cand[k] for k in NAMES] + [round(r["success"], 4), r["ok"], round(r["miou"], 4), r["no_result"], r["outside"], r["inside_1mm"], round(r["median_mm"], 3)]); log.flush()
            if score(r) > score(best_r):
                best, best_r, improved = cand, r, True
                print("  round %d trial %3d: %s=%s -> success %.1f %% | mIoU %.1f %% | no result %d | outside %d   [%.0f s]" % (rnd, trial, name, val, 100 * r["success"], 100 * r["miou"], r["no_result"], r["outside"], time.time() - T0), flush=True)
        if time.time() - T0 > TIME_LIMIT_S: break
    if not improved or time.time() - T0 > TIME_LIMIT_S: break
log.close()
print("\ntrials: %d | time %.0f s" % (trial, time.time() - T0))
print("BEST parameters:", json.dumps(best))
print("changed vs script:", {k: (BASE[k], best[k]) for k in NAMES if BASE[k] != best[k]})
print("BEST result on the tuning photos: success@%.1fmm %.1f %% (%d/%d) | mIoU %.1f %% | no result %d | outside %d | >=1mm inside %d | median %.2f mm" % (REQUIRED_MM, 100 * best_r["success"], best_r["ok"], N_PARTS, 100 * best_r["miou"], best_r["no_result"], best_r["outside"], best_r["inside_1mm"], best_r["median_mm"]))
json.dump({"base": BASE, "best": best, "base_result": rb, "best_result": best_r}, open(os.path.join(OUT, "matgof_parameter_search_best.json"), "w"))
