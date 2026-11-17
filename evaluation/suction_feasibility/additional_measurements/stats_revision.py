# Numbers for the revision: rates per condition with 95 % Wilson intervals, paired McNemar tests, mIoU per condition, split check.
# Reads the archive CSV (never writes to it). CIELAB-AMF is given twice: as in the CSV and with the six points that were
# chosen in the repeated part of the method's crop re-measured against the real label (true_d_cielab.py).
import csv, math, itertools, collections
R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
import os
S = os.path.dirname(os.path.abspath(__file__))        # miou_same_tests.csv sits next to this script
rows = list(csv.DictReader(open(R + r"\grasp_success_rate_parcalar.csv", encoding="utf-8-sig")))
iou = {(int(r["test"]), r["method"]): float(r["iou_all"]) for r in csv.DictReader(open(S + r"\miou_same_tests.csv", encoding="utf-8-sig"))}
FIX = {38: 2.49, 85: -10.09, 171: -8.82, 176: -9.50, 237: -9.90, 317: -2.00}
METHODS = ["MAT-GOF", "CIELAB-AMF", "HM R-CNN", "YOLOv8s-seg"]
D = 5.0

def cond(src):
    s = src.lower()
    return "grey" if "gri" in s else ("cognex" if "cognex" in s else "dark")
d = {}; info = {}
for r in rows:
    t = int(r["test"]); v = r["point_to_label_edge_mm"]
    d[(t, r["method"])] = None if v == "" else float(v)
    info[t] = (cond(r["source"]), r["split"], r["source"])
dfix = dict(d)
for t, v in FIX.items(): dfix[(t, "CIELAB-AMF")] = v
tests = sorted(info)

def wilson(k, n, z=1.959964):
    p = k / n; den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * (c - h), 100 * (c + h)
def ok(dd, t, m, tol):
    v = dd[(t, m)]; return v is not None and v >= D / 2 + tol
def mcnemar(dd, sel, a, b, tol):
    n01 = sum(ok(dd, t, a, tol) and not ok(dd, t, b, tol) for t in sel)
    n10 = sum(ok(dd, t, b, tol) and not ok(dd, t, a, tol) for t in sel)
    n = n01 + n10
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, k) for k in range(0, min(n01, n10) + 1)) / 2 ** n)
    return n01, n10, p

GROUPS = [("D7000 dark", lambda t: info[t][0] == "dark"), ("D7000 grey", lambda t: info[t][0] == "grey"),
          ("Cognex", lambda t: info[t][0] == "cognex"), ("dark + Cognex (old Table 2)", lambda t: info[t][0] != "grey"), ("ALL", lambda t: True),
          ("ALL, test split only", lambda t: info[t][1] == "test"), ("ALL, valid split only", lambda t: info[t][1] == "valid")]
for label, dd in (("CSV as it is", d), ("CIELAB-AMF corrected", dfix)):
    print("=" * 110); print(label)
    for name, f in GROUPS:
        sel = [t for t in tests if f(t)]
        print("  %-28s n = %d" % (name, len(sel)))
        for m in METHODS:
            if label != "CSV as it is" and m != "CIELAB-AMF": continue
            out = []
            for tol in (0.5, 1.0):
                k = sum(ok(dd, t, m, tol) for t in sel); lo, hi = wilson(k, len(sel))
                out.append("t=%.1f: %5.1f %% (%3d) [%4.1f, %5.1f]" % (tol, 100 * k / len(sel), k, lo, hi))
            none = sum(dd[(t, m)] is None for t in sel); outside = sum(dd[(t, m)] is not None and dd[(t, m)] < 0 for t in sel)
            mi = 100 * sum(iou[(t, m)] for t in sel) / len(sel)
            print("     %-12s %s | %s | no result %2d | outside %2d | mIoU %.1f" % (m, out[0], out[1], none, outside, mi))
print("=" * 110); print("McNemar (exact, two-sided), CIELAB corrected")
for name, f in (("ALL", lambda t: True), ("dark + Cognex", lambda t: info[t][0] != "grey"), ("D7000 dark", lambda t: info[t][0] == "dark")):
    sel = [t for t in tests if f(t)]
    for tol in (0.5, 1.0):
        for a, b in itertools.combinations(METHODS, 2):
            n01, n10, p = mcnemar(dfix, sel, a, b, tol)
            print("  %-14s t=%.1f  %-12s vs %-12s only-first %2d only-second %2d  p = %.4f" % (name, tol, a, b, n01, n10, p))
# drop of each method from t = 0.5 to t = 1 and the scale-free form of the criterion
print("scale-free band: t=0.5 -> %.1f %% of the strip width, t=1 -> %.1f %%" % (100 * (9.2 - 2 * 3.0) / 9.2, 100 * (9.2 - 2 * 3.5) / 9.2))
print("photos with tests per condition:", collections.Counter(info[t][0] for t in tests))
ph = collections.defaultdict(set)
for r in rows: ph[cond(r["source"])].add(r["photo_no"])
print({k: len(v) for k, v in ph.items()})
