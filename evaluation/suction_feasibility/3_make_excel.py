# !pip install openpyxl
#
# STEP 3: Turns the outputs of 2_run_tests.py into one Excel file -> results\grasp_success_rate.xlsx
#   Summary     : comparison table (grasp success rate), distance statistics, coverage.
#                 Cup diameter and tolerance are in the yellow cells; changing them updates all rates automatically.
#   Tests       : one row per test (part), methods side by side. Test no = image number in the tests folder.
#   Photos      : per photo, number of labeled grip areas, parts found and parts measured.
#   Details     : all measurements (including pixel values and the mm/pixel ratio).
# Note: needs openpyxl.

import os
import csv
from datetime import date
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.formatting.rule import CellIsRule, FormulaRule
from openpyxl.utils import get_column_letter

# ==============================================================================
# 1. PATHS AND SETTINGS
# ==============================================================================
GSR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
PARTS_CSV = os.path.join(GSR_DIR, "grasp_success_rate_parcalar.csv")
PHOTOS_CSV = os.path.join(GSR_DIR, "grasp_success_rate_fotolar.csv")
OUTPUT_XLSX = os.path.join(GSR_DIR, "grasp_success_rate.xlsx")

GRIP_WIDTH_MM = 9.2   # must be the same value as in 2_run_tests.py (used only in the note text)
CUP_DIAMETER_MM = 5   # initial values written into the yellow cells (can be changed in Excel)
TOLERANCE_MM = 1
METHODS = ["MAT-GOF", "CIELAB-AMF", "HM R-CNN", "YOLOv8s-seg"]
SPLIT_ORDER = ["test", "valid"]
SOURCE_NAMES = {      # source name in the csv -> name shown in Excel
    "D7000, black background (full res., labeled)": "D7000, black background (blackbackground_withlabels)",
    "D7000, black background (other shots)": "D7000, black background (other shots)",
    "D7000, gray background": "D7000, gray background",
    "Sony 6000x4000, black background": "Sony 6000x4000, black background",
    "Cognex 1280x1024": "Cognex 1280x1024",
}

FONT = "Arial"
F_NORMAL = Font(name=FONT, size=10)
F_BOLD = Font(name=FONT, size=10, bold=True)
F_TITLE = Font(name=FONT, size=14, bold=True)
F_HEAD = Font(name=FONT, size=10, bold=True, color="FFFFFF")
F_INPUT = Font(name=FONT, size=11, bold=True, color="0000FF")
F_NOTE = Font(name=FONT, size=9, italic=True, color="595959")
FILL_HEAD = PatternFill("solid", fgColor="1F4E79")
FILL_GROUP = PatternFill("solid", fgColor="D9E2F3")
FILL_INPUT = PatternFill("solid", fgColor="FFFF00")
FILL_RED = PatternFill("solid", fgColor="F8CBAD")
FILL_GREEN = PatternFill("solid", fgColor="C6E0B4")
FILL_GREY = PatternFill("solid", fgColor="E7E6E6")
THIN = Side(style="thin", color="BFBFBF")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
CENTER = Alignment(horizontal="center", vertical="center", wrap_text=True)
LEFT = Alignment(horizontal="left", vertical="center", wrap_text=True)

def num(v):
    return None if v == "" else float(v)

def head(ws, row, col, text):
    cell = ws.cell(row=row, column=col, value=text)
    cell.font, cell.fill, cell.alignment, cell.border = F_HEAD, FILL_HEAD, CENTER, BORDER

# ==============================================================================
# 2. READ THE CSVs
# ==============================================================================
with open(PARTS_CSV, encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))
with open(PHOTOS_CSV, encoding="utf-8-sig") as f:
    photos = list(csv.DictReader(f))
for r in rows + photos:
    r["source"] = SOURCE_NAMES.get(r["source"], r["source"])

tests = {}   # test no -> {method: row}
for r in rows:
    tests.setdefault(int(r["test"]), {})[r["method"]] = r
test_nos = sorted(tests)
sources = list(dict.fromkeys(SOURCE_NAMES.values()))
sources += [s for s in dict.fromkeys(p["source"] for p in photos) if s not in sources]

wb = Workbook()

# ==============================================================================
# 3. "Tests" SHEET (one row per test)
# ==============================================================================
ws = wb.active
ws.title = "Tests"
FIRST = 3                       # first data row
LAST = FIRST + len(test_nos) - 1
COL_DIST = 7                    # G..J : distance (mm)
COL_OK = 11                     # K..N : successful?
COL_RAD = 15                    # O..R : MIC radius of the method's own mask (mm)

groups = [(1, 6, "Test  (test no = image number in the tests folder: matgof_test_N, cielab_test_N, hmrcnn_test_N, yolo_test_N)"),
          (COL_DIST, COL_DIST + 3, "Distance from the point to the label edge (mm)  |  minus = point outside the label, empty = no result"),
          (COL_OK, COL_OK + 3, "Successful?  (against the required distance on the Summary sheet)"),
          (COL_RAD, COL_RAD + 3, "MIC radius of the method's own mask (mm)")]
for c0, c1, text in groups:
    ws.merge_cells(start_row=1, start_column=c0, end_row=1, end_column=c1)
    cell = ws.cell(row=1, column=c0, value=text)
    cell.font, cell.fill, cell.alignment = F_BOLD, FILL_GROUP, CENTER
for c, h in enumerate(["Test no", "Source", "Split", "Photo no", "Photo", "Part no"] + METHODS * 3, start=1):
    head(ws, 2, c, h)

for i, t in enumerate(test_nos):
    row = FIRST + i
    first = tests[t][METHODS[0]]
    for c, v in enumerate([t, first["source"], first["split"], int(first["photo_no"]), first["photo"], int(first["part"])], start=1):
        ws.cell(row=row, column=c, value=v)
    for j, method in enumerate(METHODS):
        r = tests[t].get(method)
        ws.cell(row=row, column=COL_DIST + j, value=num(r["point_to_label_edge_mm"]) if r else None)
        ws.cell(row=row, column=COL_RAD + j, value=num(r["mic_radius_mm"]) if r else None)
        d = f"{get_column_letter(COL_DIST + j)}{row}"
        ws.cell(row=row, column=COL_OK + j, value=f'=IF({d}="","no result",IF(AND({d}>0,{d}>=\'Summary\'!$B$7),"yes","no"))')
    for c in range(1, COL_RAD + 4):
        cell = ws.cell(row=row, column=c)
        cell.font, cell.border = F_NORMAL, BORDER
        cell.alignment = LEFT if c in (2, 5) else CENTER
        if c >= COL_DIST and not (COL_OK <= c < COL_RAD):
            cell.number_format = "0.00"

dist_range = f"{get_column_letter(COL_DIST)}{FIRST}:{get_column_letter(COL_DIST + 3)}{LAST}"
ok_range = f"{get_column_letter(COL_OK)}{FIRST}:{get_column_letter(COL_OK + 3)}{LAST}"
first_d = f"{get_column_letter(COL_DIST)}{FIRST}"
first_ok = f"{get_column_letter(COL_OK)}{FIRST}"
ws.conditional_formatting.add(dist_range, FormulaRule(formula=[f'ISBLANK({first_d})'], fill=FILL_GREY, stopIfTrue=True))
ws.conditional_formatting.add(dist_range, CellIsRule(operator="lessThan", formula=["0"], fill=FILL_RED))
ws.conditional_formatting.add(ok_range, FormulaRule(formula=[f'{first_ok}="yes"'], fill=FILL_GREEN))
ws.conditional_formatting.add(ok_range, FormulaRule(formula=[f'{first_ok}="no"'], fill=FILL_RED))
ws.conditional_formatting.add(ok_range, FormulaRule(formula=[f'{first_ok}="no result"'], fill=FILL_GREY))
ws.freeze_panes = ws.cell(row=FIRST, column=COL_DIST)
ws.auto_filter.ref = f"A2:{get_column_letter(COL_RAD + 3)}{LAST}"
ws.row_dimensions[1].height = 32
ws.row_dimensions[2].height = 20
for c, w in zip(range(1, COL_RAD + 4), [8, 40, 8, 8, 24, 9] + [13] * 12):
    ws.column_dimensions[get_column_letter(c)].width = w

T_SOURCE = f"'Tests'!$B${FIRST}:$B${LAST}"
T_SPLIT = f"'Tests'!$C${FIRST}:$C${LAST}"
def t_dist(j): return f"'Tests'!${get_column_letter(COL_DIST + j)}${FIRST}:${get_column_letter(COL_DIST + j)}${LAST}"
def t_ok(j): return f"'Tests'!${get_column_letter(COL_OK + j)}${FIRST}:${get_column_letter(COL_OK + j)}${LAST}"

# ==============================================================================
# 4. "Photos" SHEET
# ==============================================================================
sp = wb.create_sheet("Photos")
photo_cols = [("photo_no", "Photo no", 8), ("source", "Source", 40), ("split", "Split", 8), ("photo", "Photo", 34), ("width", "Width (px)", 10),
              ("height", "Height (px)", 10), ("labelled_grip_areas", "Labeled grip area", 11), ("parts_found", "Part found by the part finder", 14),
              ("parts_measured", "Measured parts (tests)", 11), ("parts_without_label", "Part without a label", 11), ("first_test", "First test no", 9), ("last_test", "Last test no", 9)]
for c, (_, title, width) in enumerate(photo_cols, start=1):
    head(sp, 1, c, title)
    sp.column_dimensions[get_column_letter(c)].width = width
sp.row_dimensions[1].height = 44
for i, p in enumerate(photos, start=2):
    for c, (key, _, _) in enumerate(photo_cols, start=1):
        v = p[key]
        cell = sp.cell(row=i, column=c, value=(v if key in ("source", "split", "photo") else (None if v == "" else int(v))))
        cell.font, cell.border = F_NORMAL, BORDER
        cell.alignment = LEFT if key in ("source", "photo") else CENTER
P_LAST = len(photos) + 1
sp.freeze_panes = "A2"
sp.auto_filter.ref = f"A1:{get_column_letter(len(photo_cols))}{P_LAST}"
def p_col(letter): return f"'Photos'!${letter}$2:${letter}${P_LAST}"
P_SOURCE, P_SPLIT, P_LABELS, P_FOUND, P_MEASURED = p_col("B"), p_col("C"), p_col("G"), p_col("H"), p_col("I")

# ==============================================================================
# 5. "Summary" SHEET
# ==============================================================================
so = wb.create_sheet("Summary", 0)
so["A1"] = "Grasp success rate: does the vacuum point chosen by the method fit on the labeled grip area?"
so["A1"].font = F_TITLE
so["A2"] = "The point chosen by each method is placed on the hand-labeled grip area; the test is successful if its distance to the nearest edge is larger than the required distance."
so["A2"].font = F_NORMAL

so["A4"] = "Cells you can change (yellow)"
so["A4"].font = F_BOLD
inputs = [(5, "Cup diameter (mm)", CUP_DIAMETER_MM, "Author's choice (05.10.2026). Changing it updates the rates below automatically."),
          (6, "Tolerans (mm)", TOLERANCE_MM, "Author's choice (05.10.2026)."),
          (7, "Required distance (mm)", "=B5/2+B6", "Calculation: cup diameter / 2 + tolerance. If the yellow cells are empty it is 0: a point inside the label counts as successful.")]
for row, label, value, note in inputs:
    so.cell(row=row, column=1, value=label).font = F_BOLD
    cell = so.cell(row=row, column=2, value=value)
    cell.border, cell.alignment, cell.number_format = BORDER, CENTER, "0.00"
    if row in (5, 6):
        cell.font, cell.fill = F_INPUT, FILL_INPUT
    else:
        cell.font = F_BOLD
    so.cell(row=row, column=3, value=note).font = F_NOTE

def style_row(row, n_cols, bold_first=True):
    for c in range(1, n_cols + 1):
        cell = so.cell(row=row, column=c)
        cell.font, cell.border = (F_BOLD if (c == 1 and bold_first) else F_NORMAL), BORDER
        cell.alignment = LEFT if c == 1 else CENTER

# --- 5a) comparison table ---
top = 9
so.cell(row=top, column=1, value="1) Grasp success rate (successful tests / all tests)").font = F_BOLD
for c, h in enumerate(["Grup", "Photo", "Test (part)"] + METHODS + [f"{m} successful" for m in METHODS], start=1):
    head(so, top + 1, c, h)
so.row_dimensions[top + 1].height = 30
group_rows = [("All tests", None, None)] + [(f"{s} split only", "split", s) for s in SPLIT_ORDER] + [(s, "source", s) for s in sources]
row = top + 2
RATE_ROWS = {}
for label, kind, value in group_rows:
    so.cell(row=row, column=1, value=label)
    if kind is None:
        photos_f = f'=COUNTIF({P_MEASURED},">0")'
        tests_f = f"=ROWS({T_SOURCE})"
        ok_f = [f'=COUNTIF({t_ok(j)},"yes")' for j in range(4)]
    else:
        crit_t = f"{T_SPLIT if kind == 'split' else T_SOURCE}"
        crit_p = f"{P_SPLIT if kind == 'split' else P_SOURCE}"
        key = f'"{value}"' if kind == "split" else f"$A{row}"
        photos_f = f'=COUNTIFS({crit_p},{key},{P_MEASURED},">0")'
        tests_f = f"=COUNTIF({crit_t},{key})"
        ok_f = [f'=COUNTIFS({crit_t},{key},{t_ok(j)},"yes")' for j in range(4)]
    so.cell(row=row, column=2, value=photos_f)
    so.cell(row=row, column=3, value=tests_f)
    for j in range(4):
        so.cell(row=row, column=8 + j, value=ok_f[j])
        cell = so.cell(row=row, column=4 + j, value=f'=IF($C{row}=0,"-",{get_column_letter(8 + j)}{row}/$C{row})')
        cell.number_format = "0.0%"
    style_row(row, 11)
    for j in range(4): so.cell(row=row, column=4 + j).number_format = "0.0%"
    if kind is None:
        for c in range(1, 12): so.cell(row=row, column=c).font = F_BOLD
    RATE_ROWS[label] = row
    row += 1

# --- 5b) distance statistics ---
top = row + 1
so.cell(row=top, column=1, value="2) Distance from the point to the label edge (all tests)").font = F_BOLD
for c, h in enumerate(["Method", "Test (part)", "No result", "Point outside the label", "Median distance (mm)", "Mean distance (mm)",
                       "Smallest distance (mm)", "Largest distance (mm)", "Successful tests", "Success rate"], start=1):
    head(so, top + 1, c, h)
so.row_dimensions[top + 1].height = 30
for j, method in enumerate(METHODS):
    row = top + 2 + j
    d, ok = t_dist(j), t_ok(j)
    values = [method, f"=ROWS({d})", f"=COUNTBLANK({d})", f'=COUNTIF({d},"<0")',
              f'=IF(COUNT({d})=0,"",MEDIAN({d}))', f'=IF(COUNT({d})=0,"",AVERAGE({d}))',
              f'=IF(COUNT({d})=0,"",MIN({d}))', f'=IF(COUNT({d})=0,"",MAX({d}))',
              f'=COUNTIF({ok},"yes")', f"=IF(B{row}=0,0,I{row}/B{row})"]
    for c, v in enumerate(values, start=1):
        so.cell(row=row, column=c, value=v)
    style_row(row, 10)
    for c in range(5, 9): so.cell(row=row, column=c).number_format = "0.00"
    so.cell(row=row, column=10).number_format = "0.0%"
row = top + 2 + len(METHODS)

# --- 5c) coverage ---
top = row + 1
so.cell(row=top, column=1, value="3) Coverage: how many labeled grip areas in the test photos could be measured?").font = F_BOLD
for c, h in enumerate(["Source", "Photo", "Labeled grip area", "Part found by the part finder", "Measured parts (tests)", "Measured / labeled"], start=1):
    head(so, top + 1, c, h)
so.row_dimensions[top + 1].height = 44
row = top + 2
for label in sources + ["Total"]:
    so.cell(row=row, column=1, value=label)
    if label == "Total":
        fs = [f"=ROWS({P_SOURCE})", f"=SUM({P_LABELS})", f"=SUM({P_FOUND})", f"=SUM({P_MEASURED})"]
    else:
        fs = [f"=COUNTIF({P_SOURCE},$A{row})", f"=SUMIF({P_SOURCE},$A{row},{P_LABELS})", f"=SUMIF({P_SOURCE},$A{row},{P_FOUND})", f"=SUMIF({P_SOURCE},$A{row},{P_MEASURED})"]
    for c, v in enumerate(fs, start=2):
        so.cell(row=row, column=c, value=v)
    so.cell(row=row, column=6, value=f'=IF(C{row}=0,"-",E{row}/C{row})')
    style_row(row, 6)
    so.cell(row=row, column=6).number_format = "0.0%"
    if label == "Total":
        for c in range(1, 7): so.cell(row=row, column=c).font = F_BOLD
    row += 1

# --- 5d) notes ---
notes = [
    "How to read",
    "Test: one part in a photo. The test no is the number of the images in the tests folder (matgof_test_N, cielab_test_N, hmrcnn_test_N, yolo_test_N are the same part).",
    "Point: center of the largest inscribed circle (MIC) of the method's own mask.",
    "Distance: distance from this point to the nearest edge of the labeled grip area. Plus = point inside the label (margin to the edge), minus = point outside the label (distance to the label), empty = the method gave no result for that part.",
    f"Scale (assumption): the width of the labeled grip area of every part was taken as {GRIP_WIDTH_MM:g} mm (the widest value in the thesis and the paper; author's choice, 05.10.2026). So the largest possible distance is {GRIP_WIDTH_MM/2:.2f} mm. For variants that are actually narrower, values come out up to 23% larger.",
    "Successful: distance greater than zero AND greater than or equal to the required distance (B7). The denominator of the success rate is all tests; a method that gives no result fails that test.",
    "Photos: only labeled photos the models did not see in training (test and valid splits). blackbackground_withlabels with its own full-resolution labels; the other sources with the full-resolution originals of the test/valid photos of the maskrcnn_600p_700x700 set (labels scaled up to the original size). List: test_listesi.csv.",
    "Parts: found with the part finder of the HM R-CNN measurement script (brightness +220, Otsu, shed filter), and all four methods are measured on the same parts. The part finder works only on a very dark background; on gray backgrounds and in Sony photos the background also saturates to white, so parts cannot be separated. That is why those sources give no or very few tests (see table 3).",
    "Parts that touch, overlap or extend past the frame are deliberately removed by the shed filter; they are not part of the tests either.",
    "MAT-GOF: run with the brightness + Otsu step as in the paper and with the parameters tuned on 24 photos of the 'train' split of blackbackground_withlabels (MATGOF_PARAMS in 2_run_tests.py; search: matgof_parameter_search.py). Not tuned separately for other cameras or shots.",
    "These results are not physical trials but a geometric calculation on the labels.",
    f"Source: 2_run_tests.py -> grasp_success_rate_parcalar.csv, grasp_success_rate_fotolar.csv ({date.today().strftime('%d.%m.%Y')}).",
]
for i, text in enumerate(notes):
    cell = so.cell(row=row + 1 + i, column=1, value=text)
    cell.font = F_BOLD if i == 0 else F_NORMAL
    if i:
        so.merge_cells(start_row=row + 1 + i, start_column=1, end_row=row + 1 + i, end_column=11)
        cell.alignment = LEFT
        so.row_dimensions[row + 1 + i].height = 30
for c, w in zip(range(1, 12), [46, 11, 12, 13, 13, 13, 13, 13, 13, 13, 13]):
    so.column_dimensions[get_column_letter(c)].width = w

# ==============================================================================
# 6. "Details" SHEET (all measurements)
# ==============================================================================
sd = wb.create_sheet("Details")
detail_cols = [("test", "Test no", 8, "0"), ("source", "Source", 40, None), ("dataset", "Dataset the label was taken from", 26, None), ("split", "Split", 8, None),
               ("photo_no", "Photo no", 8, "0"), ("photo", "Photo", 24, None), ("part", "Part no", 9, "0"), ("method", "Method", 14, None),
               ("point_to_label_edge_mm", "Distance from the point to the label edge (mm)", 18, "0.00"), ("mic_radius_mm", "MIC radius of the method (mm)", 16, "0.00"),
               ("point_to_label_edge_px", "Distance (px)", 12, "0.0"), ("mic_radius_px", "MIC radius of the method (px)", 16, "0.0"),
               ("label_mic_radius_px", "MIC radius of the label (px)", 16, "0.0"), ("mm_per_px", "mm / pixel", 11, "0.0000")]
for c, (_, title, width, _) in enumerate(detail_cols, start=1):
    head(sd, 1, c, title)
    sd.column_dimensions[get_column_letter(c)].width = width
sd.row_dimensions[1].height = 42
rows_sorted = sorted(rows, key=lambda r: (int(r["test"]), METHODS.index(r["method"])))
for i, r in enumerate(rows_sorted, start=2):
    for c, (key, _, _, fmt) in enumerate(detail_cols, start=1):
        v = r[key]
        cell = sd.cell(row=i, column=c, value=(v if fmt is None else (None if v == "" else float(v))))
        cell.font, cell.border = F_NORMAL, BORDER
        cell.alignment = LEFT if key in ("source", "dataset", "photo") else CENTER
        if fmt: cell.number_format = fmt
sd.freeze_panes = "A2"
sd.auto_filter.ref = f"A1:{get_column_letter(len(detail_cols))}{len(rows_sorted) + 1}"

wb.save(OUTPUT_XLSX)
print(f"[OK] Excel file written: {OUTPUT_XLSX}  ({len(test_nos)} tests, {len(rows)} measurements, {len(photos)} photos)")
