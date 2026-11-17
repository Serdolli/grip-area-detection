# !pip install pywin32 opencv-python numpy      (PowerPoint must be installed)
#
# Produces Figure 2 of the paper (results\Fig2_workpiece_variants.png, 2592 x 1200 px).
#
#   Source: slide 1 of the thesis presentation (three part photos + three caption lines + two lines)
#           notes_meetings\thesis\images_tables_of_method.pptx
#   The presentation file is NOT modified: a temporary copy is opened read-only, changes are made in memory and never saved.
#   Steps:
#     - caption lines are shortened and set to 36 pt  -> 6.1 pt at the 58.1 mm width used in the paper
#       (Springer rule: text inside a figure must be at least 6 pt)
#     - since the lines get larger, shapes and lines are shifted slightly up/down
#     - the slide is exported 2560 px wide and the same region as the paper figure is cropped (ratio 2.16)

import os
import shutil
import tempfile
import cv2
import numpy as np
import win32com.client

PROJECT_ROOT = os.environ.get("GRIP_PROJECT_ROOT", os.path.abspath("project_root"))   # see README

# ==============================================================================
# 1. PATHS AND SETTINGS
# ==============================================================================
SOURCE_PPTX = os.path.join(PROJECT_ROOT, r"notes_meetings\thesis\images_tables_of_method.pptx")
HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PNG = os.path.join(HERE, "results", "Fig2_workpiece_variants.png")

EXPORT_W, EXPORT_H = 2560, 1440        # slide 960 x 540 pt
FONT_SIZE = 36
TEXTS = {"Subtitle 2": "(a) Longest Variant: maximum size in the product family",
         "TextBox 7": "(b) Shortest Variant: minimum size in the product family",
         "TextBox 10": "(c) Wheeled Variant: average size, with plastic wheels"}
TOPS = {"Picture 4": 41.0, "Subtitle 2": 124.0, "Straight Connector 12": 178.0, "Picture 6": 190.0, "TextBox 7": 271.0,
        "Straight Connector 13": 325.0, "Picture 9": 337.0, "TextBox 10": 430.0}
CROP_LEFT_PT, CROP_TOP_PT = -13.5, 29.3   # top-left corner of the paper figure on the slide (pt)
CROP_W, CROP_H = 2592, 1200               # cropped region (px)

# ==============================================================================
# 2. GENERATION
# ==============================================================================
def run():
    tmp_dir = tempfile.mkdtemp()
    tmp_pptx = os.path.join(tmp_dir, "kopya.pptx")
    tmp_png = os.path.join(tmp_dir, "slayt1.png")
    shutil.copy2(SOURCE_PPTX, tmp_pptx)

    app = win32com.client.DispatchEx("PowerPoint.Application")
    try:
        pres = app.Presentations.Open(tmp_pptx, ReadOnly=True, Untitled=False, WithWindow=False)
        slide = pres.Slides(1)
        box_w = slide.Shapes("TextBox 7").Width
        for name, text in TEXTS.items():
            shape = slide.Shapes(name)
            shape.TextFrame.AutoSize = 0
            shape.TextFrame.WordWrap = False
            tr = shape.TextFrame.TextRange
            tr.Text = text
            tr.Font.Size = FONT_SIZE
            tr.ParagraphFormat.Alignment = 1          # left aligned
            shape.Left = slide.Shapes("TextBox 7").Left
            shape.Width = box_w
            assert tr.BoundWidth <= box_w - 14, "line does not fit in the box: " + text
        for name, top in TOPS.items():
            slide.Shapes(name).Top = top
        slide.Export(tmp_png, "PNG", EXPORT_W, EXPORT_H)
        pres.Saved = True
        pres.Close()
    finally:
        app.Quit()

    img = cv2.imread(tmp_png)
    k = EXPORT_W / 960.0                              # pixels per pt
    pad = int(round(-CROP_LEFT_PT * k))               # the figure starts 13.5 pt from the left edge of the slide
    img = cv2.copyMakeBorder(img, 0, 0, pad, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    y_line = int(round(TOPS["Straight Connector 12"] * k))
    x0 = pad - int(round(5.4 * k))                    # the first line starts 5.4 pt from the left edge of the slide
    img[y_line - 12:y_line + 12, x0:pad] = img[y_line - 12:y_line + 12, pad + 2:pad + 3]
    y0 = int(round(CROP_TOP_PT * k))
    out = img[y0:y0 + CROP_H, 0:CROP_W]
    assert out.shape[:2] == (CROP_H, CROP_W)
    os.makedirs(os.path.dirname(OUTPUT_PNG), exist_ok=True)
    cv2.imwrite(OUTPUT_PNG, out, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"[OK] {OUTPUT_PNG}")
    print(f"   Text size inside the figure in the paper: {FONT_SIZE * (58.1 / 25.4 * 72) / (CROP_W / k):.2f} pt (figure is 58.1 mm wide)")

if __name__ == "__main__":
    run()
