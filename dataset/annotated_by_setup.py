# Which raw photo set does every photograph of the annotated set (maskrcnn_600p_700x700, 581 photos) come from?
# Match by original file name and, where several raw files carry the name, by image similarity (same rule as 1_build_test_list.py).
import os, re, csv, collections
import cv2, numpy as np

PROJECT_ROOT = os.environ.get("GRIP_PROJECT_ROOT", os.path.abspath("project_root"))   # see README

ROOT = os.path.join(PROJECT_ROOT, r"datasets")
SMALL = os.path.join(ROOT, "derivative_datasets", "maskrcnn_600p_700x700")
SKIP = ("maskrcnn_", "fastrcnn_", "blackbackground_withlabels", "cropped_datasets", "yolo_dataset", "verified_results")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "annotated_by_setup.csv")

def letterbox(img, size=700):
    h, w = img.shape[:2]; s = size / max(h, w)
    nw, nh = int(round(w * s)), int(round(h * s))
    out = np.zeros((size, size, 3), dtype=np.uint8)
    x0, y0 = (size - nw) // 2, (size - nh) // 2
    out[y0:y0 + nh, x0:x0 + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    return out

def setup_of(path, w, h):
    p = path.lower()
    grey = "grey" in p
    if "controlledlighting_18_11_25" in p: return "D7000 18.11.25 " + ("grey" if grey else "black")
    if "bluelight_zoomed_03_12_25" in p: return "D7000 03.12.25 blue " + ("grey" if grey else "black")
    if "cognex" in p: return "Cognex"
    if "mixed_17_07_25" in p: return "Sony 17.07.25"
    if "mixed_26_07_25" in p: return "Samsung 26.07.25"
    if "blackbackground_1light" in p: return "Xiaomi / 1 light"
    return "other: " + os.path.relpath(os.path.dirname(path), ROOT)[:60]

originals = collections.defaultdict(list)
for folder, _, files in os.walk(ROOT):
    if any(s in folder for s in SKIP): continue
    for f in files:
        if f.lower().endswith((".jpg", ".jpeg", ".bmp", ".png")): originals[f.lower()].append(os.path.join(folder, f))

rows = []; table = collections.defaultdict(collections.Counter)
for split in ("train", "valid", "test"):
    d = os.path.join(SMALL, split)
    for f in sorted(os.listdir(d)):
        if not f.lower().endswith(".jpg"): continue
        base = f.split(".rf.")[0]
        m = re.match(r"(.*)_(JPG|jpg|jpeg|bmp|png)$", base)
        cands = originals.get(("%s.%s" % (m.group(1), m.group(2))).lower(), []) if m else []
        best = None
        if cands:
            small = cv2.imread(os.path.join(d, f)).astype(np.float32)
            for path in cands:
                img = cv2.imread(path)
                if img is None: continue
                diff = float(np.abs(letterbox(img).astype(np.float32) - small).mean())
                better = best is None or diff < best[0] - 0.5 or (abs(diff - best[0]) <= 0.5 and 'main_datasets' in path and 'main_datasets' not in best[1])
                if better: best = (diff, path, img.shape[1], img.shape[0])
        if best is None or best[0] >= 4.0:
            setup = "no original found" + ("" if best is None else " (best diff %.1f)" % best[0])
            setup = "no original found"
            rows.append((split, f, setup, "", ""))
        else:
            setup = setup_of(best[1], best[2], best[3])
            rows.append((split, f, setup, best[1], round(best[0], 2)))
        table[setup][split] += 1
    print(split, "done", flush=True)
with open(OUT, "w", newline="", encoding="utf-8-sig") as fh:
    w = csv.writer(fh); w.writerow(["split", "annotated_file", "setup", "original", "difference"]); w.writerows(rows)
print("%-34s %6s %6s %6s %6s" % ("setup", "train", "valid", "test", "all"))
for k in sorted(table):
    c = table[k]; print("%-34s %6d %6d %6d %6d" % (k, c["train"], c["valid"], c["test"], sum(c.values())))
print("%-34s %6d %6d %6d %6d" % ("TOTAL", *(sum(table[k][s] for k in table) for s in ("train", "valid", "test")), len(rows)))
print("unmatched names:", [r[1].split(".rf.")[0] for r in rows if r[2] == "no original found"][:25])
