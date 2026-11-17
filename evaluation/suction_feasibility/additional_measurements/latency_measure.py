# Latency of the four pipelines, measured in one session on the same photographs and with the configuration used for the rates.
#   Photographs: the 50 held-out full-resolution Nikon D7000 photographs (4928 x 3264), read into memory beforehand
#                (file reading and JPEG decoding are NOT part of the latency; they are timed separately for information).
#   Each latency runs from the decoded image in memory to the suction point(s), including the suction-point step.
#   Components:  find    = steps (1)-(2): brighten, Otsu, contours, contour filter      (once per frame; CIELAB-AMF and HM R-CNN)
#                cielab  = steps (3)-(6) + suction point                                  (once per part)
#                hm      = crop, 160 x 1600 canvas, Mask R-CNN, suction point             (once per part)
#                matgof  = whole MAT-GOF pipeline on the frame + suction point per region (once per frame)
#                yolo    = YOLOv8s-seg on the frame + suction point per instance          (once per frame)
#   REPEATS passes over all photographs; the first pass is a warm-up and is discarded.
import os, sys, csv, time, subprocess, importlib.util, statistics
import cv2, numpy as np, torch
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # evaluation/suction_feasibility
OUT = os.path.dirname(os.path.abspath(__file__))
REPEATS = int(sys.argv[1]) if len(sys.argv) > 1 else 4
spec = importlib.util.spec_from_file_location("run_tests", os.path.join(HERE, "2_run_tests.py"))
rt = importlib.util.module_from_spec(spec); spec.loader.exec_module(rt)
P, yb, m1, m2 = rt.P, rt.yb, rt.m1, rt.m2
cuda = torch.cuda.is_available()
cv2.setNumThreads(os.cpu_count())      # ultralytics switches OpenCV to one thread on import; restore the default (all cores)

def now():
    if cuda: torch.cuda.synchronize()
    return time.perf_counter()

def gpu_state():
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=clocks.current.graphics,clocks.max.graphics,power.draw,utilization.gpu", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception as e:
        return "n/a (%s)" % e

def point_of(mask):
    # suction point of a region given as a small mask (bounding box of the region)
    return rt.mic(mask)[:2]

# ---- the pipelines, image only (no label handling) ----
def find_parts(img):
    bright = cv2.convertScaleAbs(img, alpha=1.0, beta=P['brightness_beta'])
    gray = cv2.cvtColor(bright, cv2.COLOR_BGR2GRAY)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return yb["shed_filter"](contours, img.shape[1], img.shape[0])

def cielab_part(img, contour):
    PARAMS, CONFIG = m2["PARAMS"], m2["CONFIG"]
    img_cropped, mask_cropped = m2["crop_and_rotate"](img, contour, eat=CONFIG['eat'])
    l_clahe = rt.clahe.apply(m2["get_l_channel"](img_cropped))
    enhanced = cv2.add(l_clahe, cv2.morphologyEx(l_clahe, cv2.MORPH_TOPHAT, rt.kernel_morph))
    blurred = cv2.GaussianBlur(enhanced, PARAMS['gauss_ksize'], PARAMS['gauss_sigma'])
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = cv2.bitwise_and(binary, binary, mask=mask_cropped)
    if CONFIG['horizontal_kernel']: binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, rt.kernel_horz)
    binary = cv2.bitwise_not(binary)
    if CONFIG['rect_closing_after_invert']: binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, rt.kernel_rect)
    binary = cv2.bitwise_and(binary, binary, mask=mask_cropped)
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    pred = np.zeros(binary.shape, dtype=np.uint8)
    if contours:
        cv2.drawContours(pred, [max(contours, key=cv2.contourArea)], -1, 255, -1)
        return point_of(pred)
    return None

_dummy = {}
def hm_part(img, contour):
    key = img.shape[:2]
    if key not in _dummy: _dummy[key] = np.zeros(key, dtype=np.uint8)
    img_crop, gt_crop = yb["crop_and_rotate_perfect"](img, _dummy[key], contour)
    img_input, _ = yb["prepare_inputs_perfect"](img_crop, gt_crop)
    mask = rt.hm_rcnn_canvas_mask(img_input)
    return point_of(mask) if mask.any() else None

def matgof_frame(img):
    if rt.MATGOF_STEP1_OTSU:
        contours, _, _, _ = m1["otsu_treshold_filtering"]("x", img, save=False)
    else:
        contours, _, _, _ = m1["step1_alternative_hybrid"]("x", img, save=False)
    kept, _, _ = m1["shed_filter"](contours, img.shape)
    if not kept: return []
    masked_gray = m1["backgroundremover_1"](img, kept, "x", None, m1["MASK_BEFORE_TRESHOLD_COLOR"])
    adaptive = m1["adaptivethreshold"](cv2.cvtColor(masked_gray, cv2.COLOR_BGR2GRAY), kept, "x", None)
    masked_white = m1["backgroundremover_2"](adaptive, kept, "x", None)
    final, _, _, _ = m1["segment_and_merge_from_gray"](masked_white, "x", None)
    points = []
    for c in final:
        x, y, w, h = cv2.boundingRect(c)
        small = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(small, [c - np.array([x, y])], -1, 255, -1)
        points.append(point_of(small))
    return points

def yolo_frame(img):
    res = rt.yolo_model.predict(img, imgsz=yb["IMGSZ"], conf=yb["CONF"], classes=[yb["GRIP_CLASS"]], retina_masks=True, verbose=False)[0]
    points = []
    if res.masks is not None:
        for k, box in enumerate(res.boxes.xyxy.cpu().numpy().astype(int)):
            x1, y1, x2, y2 = max(0, box[0] - 2), max(0, box[1] - 2), box[2] + 2, box[3] + 2
            points.append(point_of((res.masks.data[k, y1:y2, x1:x2] > 0).to(torch.uint8).cpu().numpy()))      # only the box leaves the GPU
    return points

# ---- photographs ----
photos = [r for r in csv.DictReader(open(os.path.join(HERE, "results", "test_listesi.csv"), encoding="utf-8-sig")) if r["veri_seti"] == "blackbackground_withlabels"]
t0 = time.perf_counter()
images = [cv2.imread(r["foto_yolu"]) for r in photos]
decode_ms = 1000 * (time.perf_counter() - t0) / len(images)
print("photographs:", len(images), images[0].shape, "| reading + decoding: %.0f ms per photograph (not part of the latency)" % decode_ms)
print("GPU before:", gpu_state(), "| torch", torch.__version__, "| cuda", cuda, "| OpenCV threads", cv2.getNumThreads())

rec = {k: [] for k in ("find", "cielab", "hm", "matgof", "yolo")}
per_photo = []
gpu_mid = ""
for rep in range(REPEATS):
    for i, img in enumerate(images):
        a = now(); parts = find_parts(img); t_find = 1000 * (now() - a)
        t_c, t_h = [], []
        for c in parts:
            a = now(); cielab_part(img, c); t_c.append(1000 * (now() - a))
            a = now(); hm_part(img, c); t_h.append(1000 * (now() - a))
        a = now(); matgof_frame(img); t_m = 1000 * (now() - a)
        a = now(); yp = yolo_frame(img); t_y = 1000 * (now() - a)
        if rep == 1 and i == len(images) // 2: gpu_mid = gpu_state()
        if rep == 0: continue                      # warm-up pass
        rec["find"].append(t_find); rec["cielab"] += t_c; rec["hm"] += t_h; rec["matgof"].append(t_m); rec["yolo"].append(t_y)
        per_photo.append({"repeat": rep, "photo": photos[i]["foto_no"], "parts": len(parts), "yolo_instances": len(yp), "find_ms": t_find,
                          "cielab_total_ms": t_find + sum(t_c), "hm_total_ms": t_find + sum(t_h), "matgof_ms": t_m, "yolo_ms": t_y})
    print("pass", rep, "done", flush=True)
print("GPU during:", gpu_mid)
print("GPU after :", gpu_state())

def ms(v): return "%7.1f +/- %5.1f ms (n = %d, median %.1f)" % (statistics.mean(v), statistics.stdev(v), len(v), statistics.median(v))
print("-" * 100)
print("steps (1)-(2), per frame      :", ms(rec["find"]))
print("CIELAB-AMF, per part          :", ms(rec["cielab"]))
print("HM R-CNN, per part            :", ms(rec["hm"]))
print("MAT-GOF, per frame            :", ms(rec["matgof"]))
print("YOLOv8s-seg, per frame        :", ms(rec["yolo"]))
print("-" * 100)
for n in sorted(set(p["parts"] for p in per_photo)):
    sel = [p for p in per_photo if p["parts"] == n]
    if len(sel) < 2: continue
    print("photographs with %d part(s) (n = %3d): CIELAB-AMF %6.1f +/- %5.1f | HM R-CNN %6.1f +/- %5.1f | MAT-GOF %7.1f +/- %6.1f | YOLOv8s-seg %5.1f +/- %4.1f ms" % (
        n, len(sel), statistics.mean(p["cielab_total_ms"] for p in sel), statistics.stdev(p["cielab_total_ms"] for p in sel),
        statistics.mean(p["hm_total_ms"] for p in sel), statistics.stdev(p["hm_total_ms"] for p in sel),
        statistics.mean(p["matgof_ms"] for p in sel), statistics.stdev(p["matgof_ms"] for p in sel),
        statistics.mean(p["yolo_ms"] for p in sel), statistics.stdev(p["yolo_ms"] for p in sel)))
with open(os.path.join(OUT, "latency_measure.csv"), "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(per_photo[0].keys())); w.writeheader()
    for p in per_photo: w.writerow({k: (round(v, 2) if isinstance(v, float) else v) for k, v in p.items()})
print("saved latency_measure.csv")
