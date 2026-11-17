# !pip install opencv-python numpy pycocotools
#
# STEP 1: Builds the list of test photos -> results\test_listesi.csv
#
#   Only LABELED photos that the models did NOT SEE IN TRAINING are used (test + valid splits):
#     a) blackbackground_withlabels  : full resolution, labels at their own size (16 + 34 photos)
#     b) maskrcnn_600p_700x700       : labeled set downscaled to 700x700 (60 + 117 photos). For a photo in this set the
#                                      the full-resolution ORIGINAL is found in the datasets folder by name + image similarity,
#                                      labels are scaled to the original size (700x700 = fitted with aspect ratio kept + black band).
#   Excluded:
#     - 700x700 copies of photos already in (a) (so they are not counted twice)
#     - photos in the 'train' split of blackbackground_withlabels (MAT-GOF was tuned on that split)
#     - photos whose full-resolution original could not be found
#   fastrcnn_600p_700x700 = same photos as maskrcnn_600p_700x700; maskrcnn_320x3200_rotated = part crops of the same
#   photos. Neither is added separately.

import os
import re
import csv
import io
import contextlib
import collections
import cv2
import numpy as np
from pycocotools.coco import COCO

PROJECT_ROOT = os.environ.get("GRIP_PROJECT_ROOT", os.path.abspath("project_root"))   # see README

# ==============================================================================
# 1. PATHS AND SETTINGS
# ==============================================================================
DATASETS_ROOT = os.path.join(PROJECT_ROOT, r"datasets")
DERIVATIVE = os.path.join(DATASETS_ROOT, "derivative_datasets")
FULLRES_SET = os.path.join(DERIVATIVE, "blackbackground_withlabels")
SMALL_SET = os.path.join(DERIVATIVE, "maskrcnn_600p_700x700")
SPLITS = ("test", "valid")
SMALL_SIZE = 700
SAME_PHOTO_LIMIT = 4.0     # same photo if the mean difference (0-255) between the downscaled original and the 700x700 photo is below this

GSR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
OUTPUT_CSV = os.path.join(GSR_DIR, "test_listesi.csv")

# Folders skipped when searching for the original photo (labeled/downscaled/cropped derived sets)
SKIP_FOLDERS = ("maskrcnn_", "fastrcnn_", "blackbackground_withlabels", "cropped_datasets", "yolo_dataset")

# ==============================================================================
# 2. HELPER FUNCTIONS
# ==============================================================================
def load_coco(path):
    with contextlib.redirect_stdout(io.StringIO()):
        return COCO(path)

def letterbox(img, size=SMALL_SIZE):
    # Roboflow "Fit (black edges)" resize: aspect ratio kept, the rest is black
    h, w = img.shape[:2]
    scale = size / max(h, w)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    out = np.zeros((size, size, 3), dtype=np.uint8)
    x0, y0 = (size - new_w) // 2, (size - new_h) // 2
    out[y0:y0 + new_h, x0:x0 + new_w] = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return out, scale, x0, y0

def base_name(file_name):
    return file_name.split(".rf.")[0]            # Roboflow suffix is removed: image_43_aau_JPG

def original_file_name(base):
    m = re.match(r"(.*)_(JPG|jpg|jpeg|bmp|png)$", base)   # image_43_aau_JPG -> image_43_aau.jpg
    return f"{m.group(1)}.{m.group(2)}".lower() if m else None

def source_group(path, width, height):
    p = path.lower()
    if (width, height) == (6000, 4000): return "Sony 6000x4000, black background"
    if (width, height) == (1280, 1024): return "Cognex 1280x1024"
    if "greybackground" in p: return "D7000, gray background"
    return "D7000, black background (other shots)"

# ==============================================================================
# 3. BUILD THE LIST
# ==============================================================================
def run():
    rows = []

    # --- a) full-resolution labeled set ---
    fullres_by_base = collections.defaultdict(list)     # name -> [(split, file)]  (train included: used for exclusion)
    for split in ("train",) + SPLITS:
        coco = load_coco(os.path.join(FULLRES_SET, split, "_annotations.coco.json"))
        for img_id in coco.getImgIds():
            info = coco.loadImgs(img_id)[0]
            fullres_by_base[base_name(info["file_name"])].append((split, info["file_name"]))
            if split in SPLITS:
                rows.append({"kaynak": "D7000, black background (full res., labeled)", "veri_seti": "blackbackground_withlabels", "split": split,
                             "etiket_dosyasi": info["file_name"], "foto_yolu": os.path.join(FULLRES_SET, split, info["file_name"]),
                             "genislik": info["width"], "yukseklik": info["height"], "etiket_olcek": 1.0, "etiket_x0": 0, "etiket_y0": 0,
                             "eslesme_farki": 0.0})
    print(f"a) blackbackground_withlabels: {len(rows)} photos")

    # --- index of the original photos ---
    originals = collections.defaultdict(list)
    for folder, _, files in os.walk(DATASETS_ROOT):
        if any(s in folder for s in SKIP_FOLDERS): continue
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".bmp", ".png")):
                originals[f.lower()].append(os.path.join(folder, f))

    # --- b) 700x700 set: find the original photo ---
    counts = collections.Counter()
    for split in SPLITS:
        coco = load_coco(os.path.join(SMALL_SET, split, "_annotations.coco.json"))
        for img_id in coco.getImgIds():
            info = coco.loadImgs(img_id)[0]
            base = base_name(info["file_name"])
            small = cv2.imread(os.path.join(SMALL_SET, split, info["file_name"])).astype(np.float32)

            # Is this a copy of a photo in (a) (or of the train split MAT-GOF was tuned on)?
            same_as = None
            for fr_split, fr_file in fullres_by_base.get(base, []):
                lb, _, _, _ = letterbox(cv2.imread(os.path.join(FULLRES_SET, fr_split, fr_file)))
                if np.abs(lb.astype(np.float32) - small).mean() < SAME_PHOTO_LIMIT:
                    same_as = fr_split
                    break
            if same_as in SPLITS:
                counts["already in set (a)"] += 1
                continue
            if same_as == "train":
                counts["blackbackground_withlabels train split (MAT-GOF tuning pool)"] += 1
                continue

            # among the candidates with a matching name, the one whose image is most similar
            best = None
            for path in originals.get(original_file_name(base) or "", []):
                img = cv2.imread(path)
                if img is None: continue
                lb, scale, x0, y0 = letterbox(img)
                diff = float(np.abs(lb.astype(np.float32) - small).mean())
                if best is None or diff < best[0] - 1e-6 or (abs(diff - best[0]) <= 1e-6 and "main_datasets" in path and "main_datasets" not in best[1]):
                    best = (diff, path, img.shape[1], img.shape[0], scale, x0, y0)
            if best is None or best[0] >= SAME_PHOTO_LIMIT:
                counts["full-resolution original not found"] += 1
                continue
            diff, path, w, h, scale, x0, y0 = best
            rows.append({"kaynak": source_group(path, w, h), "veri_seti": "maskrcnn_600p_700x700", "split": split,
                         "etiket_dosyasi": info["file_name"], "foto_yolu": path, "genislik": w, "yukseklik": h,
                         "etiket_olcek": scale, "etiket_x0": x0, "etiket_y0": y0, "eslesme_farki": round(diff, 3)})
            counts["eklendi"] += 1
    print("b) maskrcnn_600p_700x700:")
    for k, v in counts.items(): print(f"     {v:>3}  {k}")

    # sort order: source -> split -> file name; foto_no is assigned in this order
    order = ["D7000, black background (full res., labeled)", "D7000, black background (other shots)", "D7000, gray background", "Sony 6000x4000, black background", "Cognex 1280x1024"]
    rows.sort(key=lambda r: (order.index(r["kaynak"]), SPLITS.index(r["split"]), r["etiket_dosyasi"]))
    for i, r in enumerate(rows, start=1): r["foto_no"] = i

    fields = ["foto_no", "kaynak", "veri_seti", "split", "etiket_dosyasi", "foto_yolu", "genislik", "yukseklik", "etiket_olcek", "etiket_x0", "etiket_y0", "eslesme_farki"]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print("\nKaynak                                   | test | valid")
    for k in order:
        print(f"{k:<40} | {sum(1 for r in rows if r['kaynak'] == k and r['split'] == 'test'):>4} | {sum(1 for r in rows if r['kaynak'] == k and r['split'] == 'valid'):>5}")
    print(f"Total photos: {len(rows)}")
    print(f"[OK] {OUTPUT_CSV}")

if __name__ == "__main__":
    run()
