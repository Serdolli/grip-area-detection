# What the raw photo folders contain: per folder the number of images, camera (EXIF), resolution and the share of grey-background photos.
# Also: which raw folders the 581 annotated photographs come from (by the original file name kept in the annotation file names).
import os, collections, json, re
import cv2, numpy as np
from PIL import Image

PROJECT_ROOT = os.environ.get("GRIP_PROJECT_ROOT", os.path.abspath("project_root"))   # see README

MAIN = os.path.join(PROJECT_ROOT, r"datasets\main_datasets")
DER = os.path.join(PROJECT_ROOT, r"datasets\derivative_datasets\maskrcnn_600p_700x700")
EXT = (".jpg", ".jpeg", ".bmp")
raw = collections.defaultdict(list)
for root, dirs, files in os.walk(MAIN):
    if "verified_results" in root: continue
    imgs = [f for f in files if f.lower().endswith(EXT)]
    others = collections.Counter(os.path.splitext(f)[1].lower() for f in files if not f.lower().endswith(EXT + (".ini",)))
    if not imgs and not others: continue
    rel = os.path.relpath(root, MAIN)
    cams, sizes, dates = collections.Counter(), collections.Counter(), []
    for f in imgs[:: max(1, len(imgs) // 6)][:6]:
        try:
            im = Image.open(os.path.join(root, f)); ex = im.getexif()
            cams[(str(ex.get(271, "")).strip() + " " + str(ex.get(272, "")).strip()).strip() or "no EXIF"] += 1
            sizes[im.size] += 1
            if ex.get(306): dates.append(str(ex.get(306))[:10])
        except Exception as e:
            cams["unreadable"] += 1
    print("%-105s %4d img %s | %s | %s | %s" % (rel[-105:], len(imgs), dict(others) if others else "", dict(cams), dict(sizes), sorted(set(dates))[:2]))
    for f in imgs: raw[os.path.splitext(f)[0].lower()].append(rel)

print("\nANNOTATED SET (maskrcnn_600p_700x700): original folder of each photograph, by file name")
for split in ("train", "valid", "test"):
    d = os.path.join(DER, split)
    names = [f for f in os.listdir(d) if f.lower().endswith(".jpg")]
    c = collections.Counter()
    for f in names:
        base = f.split(".rf.")[0]
        stem = re.sub(r"_(jpg|jpeg|bmp|png)$", "", base, flags=re.I).lower()
        hits = raw.get(stem) or raw.get(stem.replace("_", " ")) or raw.get(stem.replace("-", " "))
        top = "?? no raw file with this name" if not hits else " OR ".join(sorted(set(h.split(os.sep)[0] + ("/grey" if "grey" in h.lower() else "") for h in hits)))
        c[top] += 1
    print(" ", split, len(names))
    for k, v in sorted(c.items(), key=lambda kv: -kv[1]): print("      %4d  %s" % (v, k))
