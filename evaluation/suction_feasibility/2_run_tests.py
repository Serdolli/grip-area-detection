# !pip install ultralytics pycocotools opencv-python numpy torch torchvision
#
# STEP 2: Runs the four methods (MAT-GOF, CIELAB-AMF, HM R-CNN, YOLOv8-seg) on the photos in test_listesi.csv.
#
#   Every part is a "test" and gets a number (test_1, test_2, ...). The same number is the same part for all four methods.
#   What is measured: the vacuum point chosen by the method (MIC center of its own mask) is placed on the LABELED grip area,
#                the distance from the point to the nearest edge of the label is measured in mm.
#                Positive if inside (margin to the edge), negative if outside, empty if the method found nothing.
#   Scale: the width of the labeled grip area of every part is taken as GRIP_WIDTH_MM.
#                mm/pixel = GRIP_WIDTH_MM / (2 x MIC radius of the label)   (separately for each part)
#   Success    : distance >= cup diameter / 2 + tolerance
#
#   Outputs (to the results folder next to this script):
#     tests\matgof\matgof_test_N.png   black-and-white mask + red MIC
#     tests\cielab\cielab_test_N.png   black-and-white mask + red MIC
#     tests\hmrcnn\hmrcnn_test_N.jpg   part photo + mask (blue) + red MIC
#     tests\yolo\yolo_test_N.jpg       part photo + mask (blue) + red MIC
#     grasp_success_rate_parcalar.csv  one row per test and method
#     grasp_success_rate_fotolar.csv   per photo: number of labeled grip areas, number of parts found
#   For the Excel file: 3_make_excel.py
#
#   The methods' code is read from the method scripts in problem1_grip_area/ (and the YOLO baseline);
#   those files are not changed.

import os
import ast
import csv
import io
import contextlib
import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
from ultralytics import YOLO

PROJECT_ROOT = os.environ.get("GRIP_PROJECT_ROOT", os.path.abspath("project_root"))   # see README

# ==============================================================================
# 1. PATHS AND SETTINGS
# ==============================================================================
GRIP_WIDTH_MM = 9.2       # assumed width of the labeled grip area (the widest value in the thesis and the paper)
CUP_DIAMETER_MM = 5.0     # cup diameter       (only for the on-screen summary; change it in the yellow cell in Excel)
TOLERANCE_MM = 1.0        # positioning tolerance (only for the on-screen summary; change it in the yellow cell in Excel)

CODES = os.path.join(PROJECT_ROOT, r"codes_results\conforance_codes")
REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
METHOD_1_SCRIPT = os.path.join(REPO_DIR, "problem1_grip_area", "method1_mat_gof", "method_1_sidechecker_debugged_v3.py")
METHOD_2_SCRIPT = os.path.join(REPO_DIR, "problem1_grip_area", "method2_cielab_amf", "method_2_gripareafinder_v4.py")
METHOD_3_SCRIPT = os.path.join(REPO_DIR, "problem1_grip_area", "method3_hm_rcnn", "method_3_croppertester_for_model_3.py")
YOLO_SCRIPT = os.path.join(REPO_DIR, "problem1_grip_area", "yolov8seg_baseline", "yolov8seg_fullframe_train_test.py")
HM_MODEL_PATH = os.path.join(REPO_DIR, "models", "sametkalip_maskrcnn_160x1600_2_master.pth")
DERIVATIVE = os.path.join(PROJECT_ROOT, r"datasets\derivative_datasets")

GSR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
TEST_LIST_CSV = os.path.join(GSR_DIR, "test_listesi.csv")
TESTS_DIR = os.path.join(GSR_DIR, "tests")
OUTPUT_PARTS_CSV = os.path.join(GSR_DIR, "grasp_success_rate_parcalar.csv")
OUTPUT_PHOTOS_CSV = os.path.join(GSR_DIR, "grasp_success_rate_fotolar.csv")

SAVE_IMAGES = True
IMAGE_WIDTH = 1600                # width of the saved test images (the part lies horizontally)
MASK_TINT_BGR = (255, 170, 0)     # Mask color in the HM R-CNN and YOLO images (blue)
MIC_BGR = (0, 0, 255)             # MIC circle and center (red)

MATGOF_STEP1_OTSU = True          # True: step (1) of the paper (brightness + Otsu). False: the hybrid detector in the script
HM_SCORE_LIMIT = 0.1              # Threshold from the HM R-CNN measurement script (time_and_iou_tester...)

# MAT-GOF parameters: tuned on 24 photos (85 parts) of the 'train' split of blackbackground_withlabels,
# see matgof_parameter_search.py. Values in parentheses are the old values in the method_1 script.
# If an empty dict {} is passed, the script's own values are used.
MATGOF_PARAMS = {
    "MAX_CONTOUR_AREA_PX": 1000000,   # (500000)  so that long parts are not filtered out
    "COLOR": 0,                       # (15)      background color before thresholding
    "CONTRAST_2": 1.0,                # (1.4)
    "BRIGHTNESS_2": 2.5,              # (1.5)
    "ADAPTIVE_BLOCKSIZE": 51,         # (23)
    "ADAPTIVE_C": 0,                  # (0.4)
    "MED_K": 45,                      # (13)
    "OPEN_K": 3,                      # (1)
    "OPEN_ITERS": 2,                  # (0)
    "DILATE_ITERS": 2,                # (2)
    "MIN_SPECKLE_AREA": 50,           # (3)
    "MIN_MAIN_AREA": 20000,           # (25)
    "MERGE_GAP_PX": 5,                # (10)
}

METHODS = ["MAT-GOF", "CIELAB-AMF", "HM R-CNN", "YOLOv8s-seg"]
FOLDERS = {"MAT-GOF": "matgof", "CIELAB-AMF": "cielab", "HM R-CNN": "hmrcnn", "YOLOv8s-seg": "yolo"}

# ==============================================================================
# 2. LOADING FUNCTIONS FROM THE METHOD SCRIPTS
# ==============================================================================
def load_definitions(path):
    # Runs only the imports, UPPER_CASE constants and function definitions of the script.
    # The REAL-LIFE block waiting for calibration (RL_...) and the script's own main flow are not run; the file is not changed.
    with open(path, encoding="utf-8") as f:
        src = f.read()
    ns = {"__name__": "loaded_definitions", "__file__": path}
    for node in ast.parse(src).body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef)):
            keep = True
        elif isinstance(node, ast.Assign):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            text = ast.get_source_segment(src, node)
            keep = len(names) == len(node.targets) and all(n.isupper() and not n.startswith("RL_") for n in names) and "RL_" not in text
        else:
            keep = False
        if not keep: continue
        try:
            exec(compile(ast.Module([node], []), path, "exec"), ns)
        except ImportError:
            pass   # skip if a package not used in this script is missing
    return ns

m1 = load_definitions(METHOD_1_SCRIPT)
m2 = load_definitions(METHOD_2_SCRIPT)
m3 = load_definitions(METHOD_3_SCRIPT)
yb = load_definitions(YOLO_SCRIPT)      # find parts + crop + 160x1600 canvas (same as the HM R-CNN measurement script)
m1["SAVE_INTERMEDIATE"] = False         # do not write intermediate images to disk

# Apply the MAT-GOF parameters (with the rules of the script's own command-line settings: odd kernel size, color triple)
m1.update(MATGOF_PARAMS)
m1["MASK_BEFORE_TRESHOLD_COLOR"] = (int(m1["COLOR"]),) * 3
for _k in ("ADAPTIVE_BLOCKSIZE", "MED_K", "OPEN_K"):
    _v = int(round(float(m1[_k])))
    m1[_k] = _v if _v % 2 == 1 else _v + 1
if m1["OPEN_K"] <= 1:
    m1["OPEN_ITERS"] = 0

P = yb["PARAMS"]
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# ==============================================================================
# 3. MIC AND mm CONVERSION
# ==============================================================================
def mic(mask):
    # Largest inscribed circle of the mask: (center, radius). A 1 px border is added so masks touching the edge are not measured wrong.
    padded = cv2.copyMakeBorder((mask > 0).astype(np.uint8), 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    dist = cv2.distanceTransform(padded, cv2.DIST_L2, 5)
    _, radius, _, center = cv2.minMaxLoc(dist)
    return (center[0] - 1, center[1] - 1), radius, dist[1:-1, 1:-1]

def measure(pred_mask, gt_mask):
    # From the predicted and label masks in the same space (all in pixels):
    #   method's point = MIC center of the predicted mask
    #   distance = distance from this point to the nearest edge of the LABEL (+ inside, - outside)
    _, gt_radius, gt_dist = mic(gt_mask)
    if gt_radius <= 0:
        return None
    if not np.any(pred_mask):
        return {"center": None, "pred_radius": None, "gt_radius": gt_radius, "distance": None}
    center, pred_radius, _ = mic(pred_mask)
    distance = float(gt_dist[center[1], center[0]])
    if distance <= 0:   # point outside the label: distance to the label with a minus sign
        outside_dist = cv2.distanceTransform((gt_mask == 0).astype(np.uint8), cv2.DIST_L2, 5)
        distance = -float(outside_dist[center[1], center[0]])
    return {"center": center, "pred_radius": pred_radius, "gt_radius": gt_radius, "distance": distance}

def to_mm(m):
    mm_per_px = GRIP_WIDTH_MM / (2.0 * m["gt_radius"])
    radius_mm = None if m["pred_radius"] is None else m["pred_radius"] * mm_per_px
    distance_mm = None if m["distance"] is None else m["distance"] * mm_per_px
    return mm_per_px, radius_mm, distance_mm

# ==============================================================================
# 4. METHODS
# ==============================================================================
# --- MAT-GOF: runs on the whole frame, gives a full-resolution mask (steps 1-6 of the method_1 script) ---
def matgof_full_mask(img_orig):
    if MATGOF_STEP1_OTSU:
        contours, _, _, _ = m1["otsu_treshold_filtering"]("x", img_orig, save=False)
    else:
        contours, _, _, _ = m1["step1_alternative_hybrid"]("x", img_orig, save=False)
    kept_contours, _, _ = m1["shed_filter"](contours, img_orig.shape)
    mask = np.zeros(img_orig.shape[:2], dtype=np.uint8)
    if not kept_contours:
        return mask
    masked_gray = m1["backgroundremover_1"](img_orig, kept_contours, "x", None, m1["MASK_BEFORE_TRESHOLD_COLOR"])
    gray_for_adaptive = cv2.cvtColor(masked_gray, cv2.COLOR_BGR2GRAY)
    adaptive_thr = m1["adaptivethreshold"](gray_for_adaptive, kept_contours, "x", None)
    masked_white = m1["backgroundremover_2"](adaptive_thr, kept_contours, "x", None)
    final_contours, _, _, _ = m1["segment_and_merge_from_gray"](masked_white, "x", None)
    cv2.drawContours(mask, final_contours, -1, 1, thickness=-1)
    return mask

# --- CIELAB-AMF: runs per part on its own crop (part loop of the method_2 script) ---
clahe = cv2.createCLAHE(clipLimit=m2["PARAMS"]['clahe_clip'], tileGridSize=m2["PARAMS"]['clahe_grid'])
kernel_morph = cv2.getStructuringElement(cv2.MORPH_RECT, m2["PARAMS"]['morph_kernel'])
kernel_horz = cv2.getStructuringElement(cv2.MORPH_RECT, (m2["PARAMS"]['horz_closing_len'], 1))
kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, m2["PARAMS"]['rect_closing_size'])

def cielab_part_masks(img_orig, full_gt_mask, contour):
    PARAMS, CONFIG = m2["PARAMS"], m2["CONFIG"]
    img_cropped, mask_cropped = m2["crop_and_rotate"](img_orig, contour, eat=CONFIG['eat'])
    l_channel = m2["get_l_channel"](img_cropped)
    l_clahe = clahe.apply(l_channel)
    tophat_t1 = cv2.morphologyEx(l_clahe, cv2.MORPH_TOPHAT, kernel_morph)
    enhanced = cv2.add(l_clahe, tophat_t1)
    blurred = cv2.GaussianBlur(enhanced, PARAMS['gauss_ksize'], PARAMS['gauss_sigma'])
    _, binary_final = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary_final = cv2.bitwise_and(binary_final, binary_final, mask=mask_cropped)
    if CONFIG['horizontal_kernel']:
        binary_final = cv2.morphologyEx(binary_final, cv2.MORPH_CLOSE, kernel_horz)
    binary_final = cv2.bitwise_not(binary_final)
    if CONFIG['rect_closing_after_invert']:
        binary_final = cv2.morphologyEx(binary_final, cv2.MORPH_CLOSE, kernel_rect)
    binary_final = cv2.bitwise_and(binary_final, binary_final, mask=mask_cropped)
    contours_final, _ = cv2.findContours(binary_final, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    pred_mask = np.zeros(binary_final.shape, dtype=np.uint8)
    if contours_final:
        largest_cnt = max(contours_final, key=cv2.contourArea)
        cv2.drawContours(pred_mask, [largest_cnt], -1, 255, -1)
    # The label goes through the same crop + rotation (with the same function)
    gt_cropped, _ = m2["crop_and_rotate"](full_gt_mask * 255, contour, eat=CONFIG['eat'])
    gt_mask = ((gt_cropped > 127) & (mask_cropped > 0)).astype(np.uint8)
    return pred_mask, gt_mask

# --- HM R-CNN: runs on the 160x1600 canvas (as in the measurement script) ---
hm_model = m3["get_custom_maskrcnn"](num_classes=4)
hm_model.load_state_dict(torch.load(HM_MODEL_PATH, map_location=device))
hm_model.to(device).eval()
GRIP_AREA_LABEL_ID = m3["PARAMS"]['grip_area_label_id']
MASK_THRESHOLD = m3["PARAMS"]['mask_threshold']

def hm_rcnn_canvas_mask(img_input):
    img_rgb = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy((img_rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)).unsqueeze(0).to(device)
    with torch.no_grad():
        prediction = hm_model(img_tensor)[0]
    for j, label in enumerate(prediction['labels']):
        if label.item() == GRIP_AREA_LABEL_ID and prediction['scores'][j].item() > HM_SCORE_LIMIT:
            return (prediction['masks'][j, 0].cpu().numpy() > MASK_THRESHOLD).astype(np.uint8)
    return np.zeros(img_input.shape[:2], dtype=np.uint8)

# --- YOLOv8-seg: runs on the whole frame, gives a full-resolution mask ---
yolo_model = YOLO(yb["MODEL_SAVE_PATH"])

def yolo_full_mask(img_orig):
    h_orig, w_orig = img_orig.shape[:2]
    result = yolo_model.predict(img_orig, imgsz=yb["IMGSZ"], conf=yb["CONF"], classes=[yb["GRIP_CLASS"]], retina_masks=True, verbose=False)[0]
    if result.masks is None:
        return np.zeros((h_orig, w_orig), dtype=np.uint8)
    full_pred_mask = (result.masks.data.sum(dim=0) > 0).cpu().numpy().astype(np.uint8)
    if full_pred_mask.shape != (h_orig, w_orig):
        full_pred_mask = cv2.resize(full_pred_mask, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
    return full_pred_mask

# ==============================================================================
# 5. LABELS
# ==============================================================================
_coco_cache = {}

def load_labels(row):
    # Full-resolution label mask of the photo (grip area) and number of labeled grip areas
    key = (row["veri_seti"], row["split"])
    if key not in _coco_cache:
        with contextlib.redirect_stdout(io.StringIO()):
            coco = COCO(os.path.join(DERIVATIVE, row["veri_seti"], row["split"], "_annotations.coco.json"))
        grip_id = [c['id'] for c in coco.loadCats(coco.getCatIds()) if c['name'] == 'griparea'][0]
        ids = {coco.loadImgs(i)[0]['file_name']: i for i in coco.getImgIds()}
        _coco_cache[key] = (coco, grip_id, ids)
    coco, grip_id, ids = _coco_cache[key]
    anns = coco.loadAnns(coco.getAnnIds(imgIds=ids[row["etiket_dosyasi"]], catIds=[grip_id]))
    w, h, scale = int(row["genislik"]), int(row["yukseklik"]), float(row["etiket_olcek"])
    full_gt_mask = np.zeros((h, w), dtype=np.uint8)
    if scale == 1.0:
        for ann in anns: full_gt_mask = np.maximum(full_gt_mask, coco.annToMask(ann))
    else:
        # 700x700 labels to the original size: the black band is removed and divided by the scale
        x0, y0 = float(row["etiket_x0"]), float(row["etiket_y0"])
        for ann in anns:
            for poly in ann['segmentation']:
                pts = (np.array(poly, dtype=np.float64).reshape(-1, 2) - [x0, y0]) / scale
                cv2.fillPoly(full_gt_mask, [np.round(pts).astype(np.int32)], 1)
    return full_gt_mask, len(anns)

# ==============================================================================
# 6. TEST IMAGES (all in the same crop: part horizontal, no text)
# ==============================================================================
def to_canvas(img_orig, full_mask, contour, img_crop):
    # Passes a full-frame mask through the crop + 160x1600 canvas of the HM R-CNN measurement
    _, mask_crop = yb["crop_and_rotate_perfect"](img_orig, full_mask, contour)
    _, mask_input = yb["prepare_inputs_perfect"](img_crop, mask_crop)
    return mask_crop, mask_input

def canvas_geometry(crop_shape):
    h, w = crop_shape[:2]
    scale = min(P['target_w'] / w, P['target_h'] / h)
    new_w, new_h = int(w * scale), int(h * scale)
    return new_w, new_h, (P['target_w'] - new_w) // 2, (P['target_h'] - new_h) // 2

def canvas_point_to_crop(center, radius, crop_shape):
    h, w = crop_shape[:2]
    new_w, new_h, left, top = canvas_geometry(crop_shape)
    return ((center[0] - left + 0.5) * w / new_w - 0.5, (center[1] - top + 0.5) * h / new_h - 0.5), radius * w / new_w

def canvas_mask_to_crop(mask_canvas, crop_shape):
    h, w = crop_shape[:2]
    new_w, new_h, left, top = canvas_geometry(crop_shape)
    inner = mask_canvas[top:top + new_h, left:left + new_w].astype(np.float32)
    return (cv2.resize(inner, (w, h), interpolation=cv2.INTER_LINEAR) > 0.5).astype(np.uint8)

def cielab_crop_to_part_crop(contour, img_shape):
    # CIELAB-AMF works on its own crop; this matrix moves that crop to the crop shared with the other methods.
    # (inverse of method_2 crop_and_rotate -> full frame -> crop_and_rotate_perfect)
    center, size, angle = cv2.minAreaRect(contour)
    if size[0] < size[1]:
        angle += 90
        size = (size[1], size[0])
    diagonal = int(np.sqrt(size[0]**2 + size[1]**2)) + 10
    x1, y1 = max(0, int(center[0] - diagonal // 2)), max(0, int(center[1] - diagonal // 2))
    center_roi = (center[0] - x1, center[1] - y1)
    rot = np.vstack([cv2.getRotationMatrix2D(center_roi, angle, 1.0), [0, 0, 1]])
    crop_to_rot = np.array([[1, 0, center_roi[0] - (int(size[0]) - 1) / 2.0], [0, 1, center_roi[1] - (int(size[1]) - 1) / 2.0], [0, 0, 1]])
    to_full = np.array([[1, 0, x1], [0, 1, y1], [0, 0, 1.0]]) @ np.linalg.inv(rot) @ crop_to_rot

    rect = cv2.minAreaRect(contour)
    center, size, angle = rect
    if size[0] > size[1]:
        angle -= 90
        size = (size[1], size[0])
    w, h = int(size[0]), int(size[1])
    box = cv2.boxPoints(rect)
    x_min, y_min = max(0, int(np.min(box[:, 0]) - 20)), max(0, int(np.min(box[:, 1]) - 20))
    center_roi = (center[0] - x_min, center[1] - y_min)
    M = cv2.getRotationMatrix2D(center_roi, angle, 1.0)
    M[0, 2] += (w / 2) - center_roi[0]
    M[1, 2] += (h / 2) - center_roi[1]
    from_full = np.vstack([M, [0, 0, 1]]) @ np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1.0]])
    return (from_full @ to_full)[:2]

def save_test_image(path, mask_crop, circle, img_crop=None):
    # mask_crop: mask in the vertical crop of the part. circle: ((x, y), radius) in the same crop, or None.
    # If img_crop is given: color (photo + blue mask), otherwise black and white. The part is turned horizontal and scaled to IMAGE_WIDTH.
    h, w = mask_crop.shape[:2]
    f = IMAGE_WIDTH / h
    out_size = (IMAGE_WIDTH, max(1, int(round(w * f))))
    mask = cv2.rotate((mask_crop > 0).astype(np.uint8) * 255, cv2.ROTATE_90_CLOCKWISE)
    mask = cv2.resize(mask, out_size, interpolation=cv2.INTER_LINEAR if f >= 1 else cv2.INTER_AREA) > 127
    if img_crop is None:
        vis = np.zeros((out_size[1], out_size[0], 3), dtype=np.uint8)
        vis[mask] = 255
    else:
        gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
        alpha = float(np.clip(230.0 / max(np.percentile(gray, 99.5), 1.0), 1.0, 4.0))     # make dark photos visible
        vis = cv2.rotate(cv2.convertScaleAbs(img_crop, alpha=alpha, beta=0), cv2.ROTATE_90_CLOCKWISE)
        vis = cv2.resize(vis, out_size, interpolation=cv2.INTER_CUBIC if f >= 1 else cv2.INTER_AREA)
        vis[mask] = (0.6 * vis[mask] + 0.4 * np.array(MASK_TINT_BGR)).astype(np.uint8)
        edges, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(vis, edges, -1, MASK_TINT_BGR, 1, cv2.LINE_AA)
    if circle is not None:
        (x, y), radius = circle
        cx, cy = (h - 1 - y) * f, x * f                      # vertical crop -> horizontal image
        thickness = max(2, int(round(out_size[1] / 70)))
        cv2.circle(vis, (int(round(cx)), int(round(cy))), max(1, int(round(radius * f))), MIC_BGR, thickness, cv2.LINE_AA)
        cv2.circle(vis, (int(round(cx)), int(round(cy))), thickness + 1, MIC_BGR, -1, cv2.LINE_AA)
    if path.lower().endswith(".jpg"):
        cv2.imwrite(path, vis, [cv2.IMWRITE_JPEG_QUALITY, 92])
    else:
        cv2.imwrite(path, vis)

# ==============================================================================
# 7. MAIN LOOP
# ==============================================================================
def run():
    with open(TEST_LIST_CSV, encoding="utf-8-sig") as f:
        photos = list(csv.DictReader(f))
    if SAVE_IMAGES:
        for folder in FOLDERS.values():
            os.makedirs(os.path.join(TESTS_DIR, folder), exist_ok=True)
            for old in os.listdir(os.path.join(TESTS_DIR, folder)):      # images from the previous run are deleted so numbers do not get mixed up
                if "_test_" in old and old.lower().endswith((".png", ".jpg")):
                    os.remove(os.path.join(TESTS_DIR, folder, old))

    rows, photo_rows, test_no = [], [], 0
    for row in photos:
        img_orig = cv2.imread(row["foto_yolu"])
        if img_orig is None:
            print(f"   ! could not read: {row['foto_yolu']}")
            continue
        h_orig, w_orig = img_orig.shape[:2]
        full_gt_mask, label_count = load_labels(row)
        photo_name = row["etiket_dosyasi"].split(".rf.")[0]

        # Parts: as in the HM R-CNN measurement script (brightness + Otsu + shed filter)
        img_bright = cv2.convertScaleAbs(img_orig, alpha=1.0, beta=P['brightness_beta'])
        gray = cv2.cvtColor(img_bright, cv2.COLOR_BGR2GRAY)
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_cnts = yb["shed_filter"](contours, w_orig, h_orig)

        measured, without_label = 0, 0
        if valid_cnts:
            # The two methods that work on the whole frame run once per photo
            matgof_mask = matgof_full_mask(img_orig)
            yolo_mask = yolo_full_mask(img_orig)

            for part_no, cnt in enumerate(valid_cnts, start=1):
                img_crop, gt_crop = yb["crop_and_rotate_perfect"](img_orig, full_gt_mask, cnt)
                img_input, gt_input = yb["prepare_inputs_perfect"](img_crop, gt_crop)

                matgof_crop, matgof_input = to_canvas(img_orig, matgof_mask, cnt, img_crop)
                yolo_crop, yolo_input = to_canvas(img_orig, yolo_mask, cnt, img_crop)
                hm_input = hm_rcnn_canvas_mask(img_input)
                cielab_pred, cielab_gt = cielab_part_masks(img_orig, full_gt_mask, cnt)

                results = {
                    "MAT-GOF":     measure(matgof_input, gt_input),
                    "CIELAB-AMF":  measure(cielab_pred, cielab_gt),
                    "HM R-CNN":    measure(hm_input, gt_input),
                    "YOLOv8s-seg": measure(yolo_input, gt_input),
                }
                if any(m is None for m in results.values()):   # no labeled grip area for this part
                    without_label += 1
                    continue
                test_no += 1
                measured += 1
                for method in METHODS:
                    m = results[method]
                    mm_per_px, radius_mm, distance_mm = to_mm(m)
                    rows.append({"test": test_no, "source": row["kaynak"], "dataset": row["veri_seti"], "split": row["split"],
                                 "photo_no": int(row["foto_no"]), "photo": photo_name, "part": part_no, "method": method,
                                 "point_to_label_edge_mm": distance_mm, "mic_radius_mm": radius_mm,
                                 "point_to_label_edge_px": m["distance"], "mic_radius_px": m["pred_radius"],
                                 "label_mic_radius_px": m["gt_radius"], "mm_per_px": mm_per_px})

                if SAVE_IMAGES:
                    A = cielab_crop_to_part_crop(cnt, img_orig.shape)
                    cielab_crop = cv2.warpAffine(cielab_pred, A, (img_crop.shape[1], img_crop.shape[0]), flags=cv2.INTER_NEAREST)
                    circles = {}
                    for method in ("MAT-GOF", "HM R-CNN", "YOLOv8s-seg"):     # move the points from the canvas to the crop
                        m = results[method]
                        circles[method] = None if m["center"] is None else canvas_point_to_crop(m["center"], m["pred_radius"], img_crop.shape)
                    m = results["CIELAB-AMF"]
                    circles["CIELAB-AMF"] = None if m["center"] is None else (tuple(A @ np.array([m["center"][0], m["center"][1], 1.0])), m["pred_radius"])
                    save_test_image(os.path.join(TESTS_DIR, "matgof", f"matgof_test_{test_no}.png"), matgof_crop, circles["MAT-GOF"])
                    save_test_image(os.path.join(TESTS_DIR, "cielab", f"cielab_test_{test_no}.png"), cielab_crop, circles["CIELAB-AMF"])
                    save_test_image(os.path.join(TESTS_DIR, "hmrcnn", f"hmrcnn_test_{test_no}.jpg"), canvas_mask_to_crop(hm_input, img_crop.shape), circles["HM R-CNN"], img_crop)
                    save_test_image(os.path.join(TESTS_DIR, "yolo", f"yolo_test_{test_no}.jpg"), yolo_crop, circles["YOLOv8s-seg"], img_crop)

        photo_rows.append({"photo_no": int(row["foto_no"]), "source": row["kaynak"], "dataset": row["veri_seti"], "split": row["split"],
                           "photo": photo_name, "width": w_orig, "height": h_orig, "labelled_grip_areas": label_count,
                           "parts_found": len(valid_cnts), "parts_measured": measured, "parts_without_label": without_label,
                           "first_test": (test_no - measured + 1) if measured else "", "last_test": test_no if measured else ""})
        print(f"   photo {int(row['foto_no']):>3}/{len(photos)} | {row['kaynak'][:28]:<28} | {photo_name[:30]:<30} | labels: {label_count} | parts: {measured}", flush=True)

    with open(OUTPUT_PARTS_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow({k: ("" if v is None else (round(v, 4) if isinstance(v, float) else v)) for k, v in r.items()})
    with open(OUTPUT_PHOTOS_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(photo_rows[0].keys()))
        writer.writeheader()
        writer.writerows(photo_rows)

    # --- Summary ---
    need = CUP_DIAMETER_MM / 2.0 + TOLERANCE_MM
    sources = list(dict.fromkeys(r["source"] for r in rows))
    print("\n" + "=" * 118)
    print(f"GRASP SUCCESS RATE | cup {CUP_DIAMETER_MM:g} mm + tolerance {TOLERANCE_MM:g} mm -> the point must be at least {need:g} mm inside the label edge")
    print("=" * 118)
    print(f"{'Source':<40} | {'Parts':<5} | " + " | ".join(f"{m:<14}" for m in METHODS))
    print("-" * 118)
    for name, sel_fn in [(s, (lambda r, s=s: r["source"] == s)) for s in sources] + [("TOTAL", lambda r: True)]:
        cells = []
        for method in METHODS:
            sel = [r for r in rows if r["method"] == method and sel_fn(r)]
            ok = sum(1 for r in sel if r["point_to_label_edge_mm"] is not None and r["point_to_label_edge_mm"] >= need)
            cells.append(f"{100.0 * ok / len(sel):5.1f} % ({ok:>3})")
        print(f"{name:<40} | {len(sel):<5} | " + " | ".join(f"{c:<14}" for c in cells))
    print("-" * 118)
    print(f"Photos: {len(photo_rows)} | labeled grip areas: {sum(p['labelled_grip_areas'] for p in photo_rows)} | measured parts (tests): {test_no}")
    print(f"[OK] {OUTPUT_PARTS_CSV}")
    print(f"[OK] {OUTPUT_PHOTOS_CSV}")
    if SAVE_IMAGES: print(f"[OK] Test images: {TESTS_DIR}")

if __name__ == "__main__":
    run()
