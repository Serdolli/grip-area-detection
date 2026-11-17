import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

PROJECT_ROOT = os.environ.get("GRIP_PROJECT_ROOT", os.path.abspath("project_root"))   # see README

# =====================================================================
# SETTINGS
# =====================================================================
# Folder where the output is saved
KAYIT_KLASORU = os.path.join(PROJECT_ROOT, "..")
DOSYA_ADI = "kalibrasyon_kagidi.pdf"  # PDF is the safest for millimeter-accurate printing

# Grid dimensions
SATIR_SAYISI = 10
SUTUN_SAYISI = 14

# Physical dimensions (millimeters)
DAIRE_CAPI_MM = 5.0      # Full diameter of the filled circle
X_ARALIGI_MM  = 20.0     # Center-to-center distance along the X axis
Y_ARALIGI_MM  = 20.0     # Center-to-center distance along the Y axis

KENAR_BOSLUGU_MM = 20.0  # White margin at the edges of the sheet

# =====================================================================
# GENERATION ENGINE
# =====================================================================
def create_calibration_board():
    # Convert the total width and height to inches (1 inch = 25.4 mm)
    # Matplotlib takes the figure size in inches.
    total_width_mm = (SUTUN_SAYISI - 1) * X_ARALIGI_MM + (2 * KENAR_BOSLUGU_MM)
    total_height_mm = (SATIR_SAYISI - 1) * Y_ARALIGI_MM + (2 * KENAR_BOSLUGU_MM)
    
    fig_width_in = total_width_mm / 25.4
    fig_height_in = total_height_mm / 25.4
    
    # Create the figure at its full physical size
    fig, ax = plt.subplots(figsize=(fig_width_in, fig_height_in), dpi=300)
    ax.set_aspect('equal')
    ax.axis('off')  # Hide the axis lines
    fig.patch.set_facecolor('white')
    
    # Radius
    r = DAIRE_CAPI_MM / 2.0
    
    # Loop that draws the circles (filled solid disks)
    for row in range(SATIR_SAYISI):
        for col in range(SUTUN_SAYISI):
            # Compute the center coordinates
            cx = KENAR_BOSLUGU_MM + (col * X_ARALIGI_MM)
            # The matplotlib y axis goes from bottom to top, so it is flipped to lay out from top to bottom
            cy = total_height_mm - (KENAR_BOSLUGU_MM + (row * Y_ARALIGI_MM))
            
            # Create a completely filled black circle
            circle = patches.Circle((cx, cy), radius=r, facecolor='black', edgecolor='none', fill=True)
            ax.add_patch(circle)
            
    # Set the limits
    ax.set_xlim(0, total_width_mm)
    ax.set_ylim(0, total_height_mm)
    
    # Folder check and saving
    os.makedirs(KAYIT_KLASORU, exist_ok=True)
    tam_yol = os.path.join(KAYIT_KLASORU, DOSYA_ADI)
    
    plt.tight_layout(pad=0)
    plt.savefig(tam_yol, format='pdf', bbox_inches='tight', pad_inches=0)
    plt.close()
    
    print(f"\n[OK] Calibration sheet created!")
    print(f"File: {tam_yol}")
    print(f"Physical size: {total_width_mm:.1f} mm x {total_height_mm:.1f} mm")
    print(f"[!] PRINTER SETTING: always select 'Actual Size' when printing. Do NOT use 'Fit'!")

if __name__ == "__main__":
    create_calibration_board()