
import os
import glob
import time
import cv2
import numpy as np

PROJECT_ROOT = os.environ.get("GRIP_PROJECT_ROOT", os.path.abspath("project_root"))   # see README

# -------------------- Common paths --------------------
INPUT_FOLDER              = os.path.join(PROJECT_ROOT, "..", r"test\input")
CONTOUR_OUTPUT_FOLDER     = os.path.join(PROJECT_ROOT, "..", r"test\contourdetection_output")
CONTOUR_COORDS_TXT        = os.path.join(os.path.join(PROJECT_ROOT, "..", r"test\contourcoordinates"), "all_coords.txt")
ADAPT_THRESH_OUTPUT       = os.path.join(PROJECT_ROOT, "..", r"test/test_adapt_thresh_output")
MASKED_RESULTS_FOLDER     = os.path.join(PROJECT_ROOT, "..", r"test/masked_results")
OTSU_TREE_VIS_FOLDER      = os.path.join(PROJECT_ROOT, "..", r"test/otsu_tree_vis")

os.makedirs(CONTOUR_OUTPUT_FOLDER, exist_ok=True)
os.makedirs(os.path.dirname(CONTOUR_COORDS_TXT), exist_ok=True)
os.makedirs(ADAPT_THRESH_OUTPUT, exist_ok=True)
os.makedirs(MASKED_RESULTS_FOLDER, exist_ok=True)
os.makedirs(OTSU_TREE_VIS_FOLDER, exist_ok=True)

# -------------------- Step 0: rename images --------------------
def step0_rename_images():
    """Rename all files in INPUT_FOLDER to image_{i}_{i}.ext."""
    files = [f for f in os.listdir(INPUT_FOLDER) if os.path.isfile(os.path.join(INPUT_FOLDER, f))]
    for idx, filename in enumerate(files, start=1):
        ext = os.path.splitext(filename)[1]
        new_name = f"image_{idx}_{idx}{ext}"
        old_path = os.path.join(INPUT_FOLDER, filename)
        new_path = os.path.join(INPUT_FOLDER, new_name)
        if old_path == new_path:
            continue
        try:
            os.rename(old_path, new_path)
        except Exception as e:
            print("rename error:", old_path, "->", new_path, e)
    print("Names are updated.")

# -------------------- Helpers --------------------
def contour_ratio(cnt):
    """Return rect_area/contour_area ratio used for 'rect-like' rejection."""
    B = float(cv2.contourArea(cnt))
    if B <= 0:
        return np.nan
    try:
        (_, _), (rw, rh), _ = cv2.minAreaRect(cnt)
        A = float(rw) * float(rh)
    except Exception:
        A = 0.0
    return (A / B) if B != 0 else np.nan

def draw_and_save_contours_preview(img_bgr, contours, out_path, draw_scale=1, error_threshold=1.4):
    """Draw contours: green = kept, red = rejected (rect-like)."""
    canvas = img_bgr.copy()
    for cnt in contours:
        r = contour_ratio(cnt)
        keep = not (not np.isnan(r) and r > error_threshold)
        color = (0, 255, 0) if keep else (0, 0, 255)
        cv2.drawContours(canvas, [cnt], -1, color, 2 * draw_scale)
    cv2.imwrite(out_path, canvas)

def append_coords_for_image(base_name_no_ext, kept_contours, coords_txt_path):
    """Append only the kept contour coordinates to the TXT file."""
    with open(coords_txt_path, "a", encoding="utf-8") as f:
        f.write(f"=== Image: {base_name_no_ext} ===\n")
        for i, cnt in enumerate(kept_contours, start=1):
            area = float(cv2.contourArea(cnt))
            f.write(f"Contour {i}: area={area:.3f}\n")
            pts = cnt.reshape(-1, 2)
            for (x, y) in pts:
                f.write(f"{int(x)},{int(y)}\n")
            f.write("\n")

def mask_with_contours(img_gray_or_bgr, kept_contours, contour_thickness=2):
    """Mask using only the kept contours."""
    if img_gray_or_bgr.ndim == 2:
        h, w = img_gray_or_bgr.shape
    else:
        h, w = img_gray_or_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    valid_cnts = []
    for cnt in kept_contours:
        if len(cnt) < 3:
            continue
        cnt_np = np.array(cnt, dtype=np.int32).reshape(-1, 1, 2)
        cnt_np[:, 0, 0] = np.clip(cnt_np[:, 0, 0], 0, w - 1)
        cnt_np[:, 0, 1] = np.clip(cnt_np[:, 0, 1], 0, h - 1)
        cv2.fillPoly(mask, [cnt_np], 255)
        valid_cnts.append(cnt_np)

    if img_gray_or_bgr.ndim == 2:
        result = np.full_like(img_gray_or_bgr, 255)
        result[mask == 255] = img_gray_or_bgr[mask == 255]
        for cnt_np in valid_cnts:
            cv2.drawContours(result, [cnt_np], -1, (0,), thickness=contour_thickness)
    else:
        result = np.full_like(img_gray_or_bgr, 255)
        result[mask == 255] = img_gray_or_bgr[mask == 255]
        for cnt_np in valid_cnts:
            cv2.drawContours(result, [cnt_np], -1, (255, 255, 255), thickness=contour_thickness)
    return result

# -------------------- Per-image steps --------------------
def detect_and_shed_contours(src_bgr, min_area=1500, error_threshold=1.4):
    """Detect contours and shed them:
       - keep only area >= min_area
       - reject 'rect-like' contours where rect_area/contour_area > error_threshold
    """
    gray = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours_all, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[:2]
    # area filter
    area_filtered = [c for c in contours_all if cv2.contourArea(c) >= min_area]
    # rect-like shedding
    kept = []
    rejected = []
    for c in area_filtered:
        r = contour_ratio(c)
        if (not np.isnan(r)) and (r > error_threshold):
            rejected.append(c)
        else:
            kept.append(c)
    return kept, rejected

def adapt_thresh_for_image(src_bgr, blocksize=35, C=1, blur_k=(5, 5)):
    """Adaptive mean threshold."""
    gray = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2GRAY)
    if blur_k is not None:
        gray = cv2.blur(gray, blur_k)
    thr = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        blocksize, C
    )
    return thr

def grip_area_for_image(masked_bgr):
    """OTSU + opening + dilate + flood-fill holes + keep big components + fill contours white."""
    MED_K = 7
    OPEN_K = 2
    OPEN_ITERS = 0
    MIN_SPECKLE_AREA = 10
    MIN_MAIN_AREA = 50
    USE_INV = True

    gray = cv2.cvtColor(masked_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, MED_K)

    if USE_INV:
        _, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        _, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (OPEN_K, OPEN_K))
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, k, iterations=OPEN_ITERS)
    thr = cv2.dilate(thr, k, iterations=3)

    ff = thr.copy()
    h, w = ff.shape
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, mask, (0, 0), 255)
    holes = cv2.bitwise_not(ff)
    thr = cv2.bitwise_or(thr, holes)

    nb, labels, stats, _ = cv2.connectedComponentsWithStats(thr, connectivity=8)
    cleaned = np.zeros_like(thr)
    for lab in range(1, nb):
        if stats[lab, cv2.CC_STAT_AREA] >= MIN_SPECKLE_AREA:
            cleaned[labels == lab] = 255

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    vis = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
    for c in contours:
        if cv2.contourArea(c) >= MIN_MAIN_AREA:
            cv2.drawContours(vis, [c], -1, (255, 255, 255), -1)
    return vis

# -------------------- Runner (per image, time in seconds) --------------------
def main():
    total_t0 = time.time()

    # 0) Rename first
    step0_rename_images()

    # Build ordered list after rename (by image index in the name)
    files = [f for f in os.listdir(INPUT_FOLDER) if os.path.isfile(os.path.join(INPUT_FOLDER, f))]
    pairs = []
    for f in files:
        name, _ = os.path.splitext(f)
        if name.startswith("image_"):
            try:
                _, i1, i2 = name.split("_")
                idx = int(i1)
            except Exception:
                idx = 999999
        else:
            idx = 999999
        pairs.append((idx, f))
    pairs.sort(key=lambda x: x[0])

    # Reset coords file
    with open(CONTOUR_COORDS_TXT, "w", encoding="utf-8") as f:
        f.write("")

    for _, fname in pairs:
        per_t0 = time.time()
        base_name = os.path.splitext(fname)[0]
        input_path = os.path.join(INPUT_FOLDER, fname)

        print(f"\n----- Processing: {base_name} -----")
        src_bgr = cv2.imread(input_path, cv2.IMREAD_COLOR)
        if src_bgr is None:
            print("Skip (read error):", input_path)
            continue

        # 1) Detect + shed (area & rect-like) -> kept + rejected
        kept_contours, rejected_contours = detect_and_shed_contours(src_bgr)

        # 2) Adaptive threshold
        thr = adapt_thresh_for_image(src_bgr)

        # 3) Background remove using only the kept contours
        thr_bgr = cv2.cvtColor(thr, cv2.COLOR_GRAY2BGR)
        masked = mask_with_contours(thr_bgr, kept_contours, contour_thickness=2)

        # 4) Grip area on the single masked image
        vis = grip_area_for_image(masked)

        # Floating seconds with two decimals
        per_dt = time.time() - per_t0
        time_tag = f"{per_dt:.2f}seconds"

        # Preview: green (kept) + red (rejected)
        preview_out = os.path.join(CONTOUR_OUTPUT_FOLDER, f"{base_name}_{time_tag}.png")
        draw_and_save_contours_preview(src_bgr, kept_contours + rejected_contours, preview_out)

        # Append only the kept contours to the coords file
        append_coords_for_image(f"{base_name}_{time_tag}", kept_contours, CONTOUR_COORDS_TXT)

        # Save other outputs
        thr_out = os.path.join(ADAPT_THRESH_OUTPUT, f"{base_name}_{time_tag}.png")
        cv2.imwrite(thr_out, thr)

        masked_out = os.path.join(MASKED_RESULTS_FOLDER, f"{base_name}_masked_{time_tag}.png")
        cv2.imwrite(masked_out, masked)

        final_out = os.path.join(OTSU_TREE_VIS_FOLDER, f"{base_name}_otsu_tree_{time_tag}.png")
        cv2.imwrite(final_out, vis)

        print(f"--> {base_name} done in {per_dt:.2f} s (kept: {len(kept_contours)}, rejected: {len(rejected_contours)})")

    total_dt = time.time() - total_t0
    print(f"\n[OK] Per-image pipeline finished. Total time: {total_dt:.2f} s")

if __name__ == "__main__":
    main()
