# Recomputes Table 2 of the paper and all numbers in the Results section from the CSV files.
#
#   Input   : results\grasp_success_rate_parcalar.csv   point - label edge distances (2_run_tests.py + 2b_fix_cielab_points.py)
#             results\miou_parcalar.csv                 IoU on the same parts (additional_measurements\miou_same_tests.py)
#             results\grasp_success_rate_fotolar.csv    number of labeled grip areas per photo (for coverage)
#   Rule    : successful if the point is at least  cup diameter / 2 + tolerance  inside the label edge.
#             A method that gives no result fails that part.
#   Conditions: D7000 black background, D7000 gray background, Cognex, all (the columns of the paper).
#   Output  : results\tablo2_teorik_oranlar.csv  and a summary on screen (with confidence intervals and paired tests)
#
#   No extra packages needed. If scipy is available, the HM R-CNN - YOLO distance comparison (Wilcoxon) is printed too.

import os
import csv
import math
import statistics

# ==============================================================================
# 1. PATHS AND SETTINGS
# ==============================================================================
HERE = os.path.dirname(os.path.abspath(__file__))
PARTS_CSV = os.path.join(HERE, "results", "grasp_success_rate_parcalar.csv")
IOU_CSV = os.path.join(HERE, "results", "miou_parcalar.csv")
PHOTOS_CSV = os.path.join(HERE, "results", "grasp_success_rate_fotolar.csv")
OUTPUT_CSV = os.path.join(HERE, "results", "tablo2_teorik_oranlar.csv")

CUP_DIAMETER_MM = 5.0
TOLERANCES_MM = [0.5, 1.0]
EXTRA_TOLERANCE_MM = 1.5            # not in the table; for the "stricter tolerance" sentence in the text
GRIP_WIDTH_MM = 9.2
METHODS = ["MAT-GOF", "CIELAB-AMF", "HM R-CNN", "YOLOv8s-seg"]
CONDITIONS = ["D7000 black", "D7000 gray", "Cognex", "All"]

def condition(source):
    s = source.lower()
    return "D7000 gray" if "gri" in s else ("Cognex" if "cognex" in s else ("Sony" if "sony" in s else "D7000 black"))

# ==============================================================================
# 2. STATISTICS
# ==============================================================================
def wilson(k, n, z=1.959964):
    # 95% Wilson confidence interval (percent)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * (c - h), 100 * (c + h)

def mcnemar(a_ok, b_ok):
    # Exact (binomial) McNemar test for paired results, two-sided
    n01 = sum(a and not b for a, b in zip(a_ok, b_ok))
    n10 = sum(b and not a for a, b in zip(a_ok, b_ok))
    n = n01 + n10
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, k) for k in range(0, min(n01, n10) + 1)) / 2 ** n)
    return n01, n10, p

# ==============================================================================
# 3. CALCULATION
# ==============================================================================
def run():
    with open(PARTS_CSV, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    with open(IOU_CSV, encoding="utf-8-sig") as f:
        iou = {(int(r["test"]), r["method"]): float(r["iou_all"]) for r in csv.DictReader(f)}
    d, cond, photo = {}, {}, {}
    for r in rows:
        t = int(r["test"])
        d[(t, r["method"])] = None if r["point_to_label_edge_mm"] == "" else float(r["point_to_label_edge_mm"])
        cond[t] = condition(r["source"]); photo[t] = r["photo_no"]
    tests = sorted(cond)
    ok = lambda t, m, tol: d[(t, m)] is not None and d[(t, m)] >= CUP_DIAMETER_MM / 2.0 + tol
    select = lambda c: [t for t in tests if c == "All" or cond[t] == c]

    print("=" * 112)
    print(f"TABLE 2 | cup {CUP_DIAMETER_MM:g} mm | {len(tests)} parts, {len(set(photo.values()))} photos")
    for tol in TOLERANCES_MM:
        band = GRIP_WIDTH_MM - 2 * (CUP_DIAMETER_MM / 2.0 + tol)
        print(f"   t = {tol:g} mm: the point must lie in the central {band:.1f} mm of the strip width (%{100 * band / GRIP_WIDTH_MM:.0f})")
    print("=" * 112)
    out = []
    for c in CONDITIONS:
        sel = select(c)
        print(f"{c}  (parts {len(sel)}, photos {len(set(photo[t] for t in sel))})")
        for m in METHODS:
            line = {"condition": c, "workpieces": len(sel), "method": m, "miou_percent": round(100 * sum(iou[(t, m)] for t in sel) / len(sel), 1),
                    "no_result": sum(d[(t, m)] is None for t in sel), "point_outside_label": sum(d[(t, m)] is not None and d[(t, m)] < 0 for t in sel)}
            text = []
            for tol in TOLERANCES_MM:
                k = sum(ok(t, m, tol) for t in sel)
                lo, hi = wilson(k, len(sel))
                line[f"successful_t_{tol:g}mm"] = k
                line[f"rate_percent_t_{tol:g}mm"] = round(100.0 * k / len(sel), 1)
                line[f"ci95_low_t_{tol:g}mm"] = round(lo, 1); line[f"ci95_high_t_{tol:g}mm"] = round(hi, 1)
                text.append(f"t={tol:g}: %{100.0 * k / len(sel):5.1f} ({k:>3}) [{lo:4.1f}, {hi:5.1f}]")
            out.append(line)
            print(f"   {m:<12} mIoU %{line['miou_percent']:4.1f} | " + " | ".join(text) + f" | no result {line['no_result']:>2} | outside the label {line['point_outside_label']:>2}")
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        writer.writeheader()
        writer.writerows(out)

    # --- Paired tests and other numbers in the text ---
    print("-" * 112)
    pairs = [("All", "HM R-CNN", "YOLOv8s-seg"), ("All", "HM R-CNN", "MAT-GOF"), ("All", "HM R-CNN", "CIELAB-AMF"), ("All", "YOLOv8s-seg", "MAT-GOF"),
             ("All", "YOLOv8s-seg", "CIELAB-AMF"), ("D7000 black", "MAT-GOF", "HM R-CNN"), ("D7000 black", "MAT-GOF", "YOLOv8s-seg"), ("D7000 black", "HM R-CNN", "YOLOv8s-seg")]
    for c, a, b in pairs:
        sel = select(c)
        for tol in TOLERANCES_MM:
            n01, n10, p = mcnemar([ok(t, a, tol) for t in sel], [ok(t, b, tol) for t in sel])
            print(f"McNemar | {c:<11} | t = {tol:g} mm | {a:<12} - {b:<12} | only the first {n01:>2}, only the second {n10:>2} | p = {p:.4f}")
    print("-" * 112)
    for m in METHODS:
        v = [d[(t, m)] for t in tests if d[(t, m)] is not None]
        between = sum(1 for x in v if CUP_DIAMETER_MM / 2 + TOLERANCES_MM[0] <= x < CUP_DIAMETER_MM / 2 + TOLERANCES_MM[1])
        print(f"{m:<12} | median d = {statistics.median(v):.2f} mm (n = {len(v)}) | passes at t = 0.5 but fails at t = 1: {between}")
    try:
        from scipy.stats import wilcoxon
        res = wilcoxon([d[(t, "HM R-CNN")] for t in tests], [d[(t, "YOLOv8s-seg")] for t in tests])
        print(f"Wilcoxon (paired) HM R-CNN - YOLOv8s-seg distances: p = {res.pvalue:.2e}")
    except ImportError:
        print("scipy is required for the Wilcoxon test (available in base python).")
    # --- Other numbers quoted in the text ---
    print("-" * 112)
    tol = EXTRA_TOLERANCE_MM
    band = GRIP_WIDTH_MM - 2 * (CUP_DIAMETER_MM / 2.0 + tol)
    print(f"Strict tolerance t = {tol:g} mm (central %{100 * band / GRIP_WIDTH_MM:.0f} of the strip; not in Table 2, mentioned in the text):")
    for c in CONDITIONS:
        sel = select(c)
        print(f"   {c:<11} (n = {len(sel):>3}) | " + " | ".join(f"{m} %{100.0 * sum(ok(t, m, tol) for t in sel) / len(sel):.1f}" for m in METHODS))
    for c in ["D7000 black", "D7000 gray", "Cognex", "All"]:
        sel = select(c)
        n01, n10, p = mcnemar([ok(t, "HM R-CNN", tol) for t in sel], [ok(t, "YOLOv8s-seg", tol) for t in sel])
        print(f"McNemar | {c:<11} | t = {tol:g} mm | HM R-CNN - YOLOv8s-seg | only the first {n01:>2}, only the second {n10:>2} | p = {p:.4f}")
    # 95% confidence interval of the HM R-CNN - YOLO difference (paired proportions, Wald), t = 1 mm, all parts
    n01, n10, p = mcnemar([ok(t, "HM R-CNN", 1.0) for t in tests], [ok(t, "YOLOv8s-seg", 1.0) for t in tests])
    n = len(tests)
    diff = (n01 - n10) / n
    se = math.sqrt(n01 + n10 - (n01 - n10) ** 2 / n) / n
    print(f"HM R-CNN - YOLOv8s-seg difference, t = 1 mm, {n} parts: {100 * diff:.1f} points, 95% confidence interval [{100 * (diff - 1.959964 * se):.1f}, {100 * (diff + 1.959964 * se):.1f}]")
    # Test split only (reported separately because the YOLO checkpoint was selected on the valid split)
    split = {int(r["test"]): r["split"] for r in rows}
    sel = [t for t in tests if split[t] == "test"]
    n01, n10, p = mcnemar([ok(t, "HM R-CNN", 1.0) for t in sel], [ok(t, "YOLOv8s-seg", 1.0) for t in sel])
    print(f"Test split only, t = 1 mm (n = {len(sel)}): " + " | ".join(f"{m} %{100.0 * sum(ok(t, m, 1.0) for t in sel) / len(sel):.1f}" for m in METHODS)
          + f" | McNemar HM R-CNN - YOLOv8s-seg {n01}:{n10}, p = {p:.2f}")

    with open(PHOTOS_CSV, encoding="utf-8-sig") as f:
        ph = list(csv.DictReader(f))
    print("-" * 112)
    total_l, total_m = 0, 0
    for c in ["D7000 black", "D7000 gray", "Cognex", "Sony"]:
        sel = [p for p in ph if condition(p["source"]) == c]
        lab, mea = sum(int(p["labelled_grip_areas"]) for p in sel), sum(int(p["parts_measured"]) for p in sel)
        total_l += lab; total_m += mea
        print(f"Coverage | {c:<11} | photos {len(sel):>3} | labeled grip areas {lab:>3} | separated by steps (1)-(2) {mea:>3}")
    print(f"Coverage | total       | photos {len(ph):>3} | labeled grip areas {total_l:>3} | separated {total_m:>3} (%{100 * total_m / total_l:.0f})")
    corrected = sorted(set(int(r["test"]) for r in rows if r.get("corrected") == "1"))
    print(f"Tests re-measured by 2b_fix_cielab_points.py: {corrected}")
    print(f"[OK] {OUTPUT_CSV}")

if __name__ == "__main__":
    run()
