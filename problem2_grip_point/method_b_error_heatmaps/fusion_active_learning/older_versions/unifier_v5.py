"""
UNiFiER - active-learning robot coordinate selector
Multi-model committee + human in the loop

  * Support for 8 heatmaps (A to H)
  * Pixels that are ZERO or NaN in any heatmap -> ignored completely
  * The success criterion is always 0 or 1 (binary) - the scale question was removed
  * The suction cup width (mm) is asked once at the start of the session and saved to session_config.pkl
  * The suggested pixel is marked on all maps and saved at every step
  * The robot mm coordinate is given from the pixel coordinate + H matrix
  * 21 different models, weighted committee vote
  * Detailed summary at every step: pixel, robot mm, heatmap values, model predictions
"""

import os
import pickle
import numpy as np
import pandas as pd
import warnings
import cv2
import lightgbm as lgb
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb

warnings.filterwarnings("ignore")

# The success criterion is fixed - do not change it
MAX_SCORE = 1.0


# ==============================================================
# 1. FILE PATHS
# ==============================================================
PATHS = {
    # -- Heatmap PKL inputs (each an ndarray) --
    'heatmap_A': r"path\to\heatmap_A.pkl",
    'heatmap_B': r"path\to\heatmap_B.pkl",
    'heatmap_C': r"path\to\heatmap_C.pkl",
    'heatmap_D': r"path\to\heatmap_D.pkl",
    'heatmap_E': r"path\to\heatmap_E.pkl",
    'heatmap_F': r"path\to\heatmap_F.pkl",
    'heatmap_G': r"path\to\heatmap_G.pkl",
    'heatmap_H': r"path\to\heatmap_H.pkl",

    # -- Homography matrix --
    'H_npy':     r"path\to\H.npy",

    # -- Reference image (the mark is drawn on it) --
    'master_img': r"path\to\MASTER.bmp",

    # -- Output folder --
    'output_dir': r"path\to\test",
}

# Heatmap name labels (for readable logs)
HEATMAP_LABELS = {
    'heatmap_A': 'A (mask/distance)',
    'heatmap_B': 'B (camera error)',
    'heatmap_C': 'C (transform error)',
    'heatmap_D': 'D (robot error)',
    'heatmap_E': 'E (map 5)',
    'heatmap_F': 'F (map 6)',
    'heatmap_G': 'G (map 7)',
    'heatmap_H': 'H (map 8)',
}

# Active heatmap keys (easy to switch on and off)
ACTIVE_HEATMAPS = [
    'heatmap_A',
    'heatmap_B',
    'heatmap_C',
    'heatmap_D',
    'heatmap_E',
    'heatmap_F',
    'heatmap_G',
    'heatmap_H',
]


# ==============================================================
# 2. HELPER FUNCTIONS
# ==============================================================

def load_heatmap(path: str, label: str) -> np.ndarray:
    """
    Loads an ndarray from a PKL file.
    If the data is a dictionary (coordinate-based sparse matrix), converts it to a dense matrix.
    """
    with open(path, 'rb') as f:
        data = pickle.load(f)

    if isinstance(data, dict):
        keys = list(data.keys())

        # sparse coordinate dictionary with (x, y) tuple keys
        if len(keys) > 0 and isinstance(keys[0], tuple):
            print(f"  [!] {label} detected as a coordinate dictionary. Converting to a dense matrix...")
            arr = np.zeros((1080, 1280), dtype=np.float32)
            for coord, val in data.items():
                if len(coord) == 2:
                    c1, c2 = int(coord[0]), int(coord[1])
                    try:
                        arr[c2, c1] = float(val)
                    except IndexError:
                        arr[c1, c2] = float(val)
            print(f"  [OK] {label:30s} loaded. Shape: {arr.shape}  min={arr.min():.3f}  max={arr.max():.3f}")
            return arr

        # Single-key dictionary - extract the array inside
        elif len(keys) == 1:
            data = data[keys[0]]
        else:
            raise ValueError(f"{label} is in an unrecognized dictionary format. Keys: {keys}")

    arr = np.array(data, dtype=np.float32)
    print(f"  [OK] {label:30s} loaded. Shape: {arr.shape}  min={arr.min():.3f}  max={arr.max():.3f}")
    return arr


def pixel_to_world(x_px: int, y_px: int, H: np.ndarray):
    """Pixel -> robot mm conversion with the homography."""
    pt = H @ np.array([x_px, y_px, 1.0], dtype=np.float64)
    return float(pt[0] / pt[2]), float(pt[1] / pt[2])


def build_valid_space(maps: dict) -> pd.DataFrame:
    """
    Finds the intersection of the pixels that are >0 and not NaN in all active maps.
    All heatmap values of every valid pixel are written to a DataFrame.
    """
    print("\nBuilding the valid pixel space...")
    keys = list(maps.keys())
    first = maps[keys[0]]
    H_img, W_img = first.shape[:2]

    rows = []
    for y in range(H_img):
        for x in range(W_img):
            vals = {}
            valid = True
            for k, arr in maps.items():
                v = float(arr[y, x])
                if v == 0.0 or np.isnan(v):
                    valid = False
                    break
                vals[k] = v
            if valid:
                rows.append({'x_px': x, 'y_px': y, **vals})

    df = pd.DataFrame(rows)
    print(f"  -> Total valid pixels: {len(df):,} / {H_img * W_img:,}")
    if len(df) == 0:
        raise RuntimeError("No valid pixel found! There is no overlapping non-zero region in all maps.")
    return df


def get_fresh_models() -> dict:
    """Extended model set (LightGBM + others)."""
    return {
        # -- LIGHTGBM --
        'LGBM_Hizli':       lgb.LGBMRegressor(n_estimators=30,  learning_rate=0.10, num_leaves=7,  verbose=-1),
        'LGBM_Derin':       lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, max_depth=5,   verbose=-1),
        'LGBM_Agresif':     lgb.LGBMRegressor(n_estimators=50,  learning_rate=0.20, num_leaves=15, verbose=-1),
        'LGBM_Dengeli':     lgb.LGBMRegressor(n_estimators=75,  learning_rate=0.08, feature_fraction=0.8, verbose=-1),
        'LGBM_Muhafazakar': lgb.LGBMRegressor(n_estimators=40,  learning_rate=0.01, min_child_samples=5,  verbose=-1),
        'LightX_DART':      lgb.LGBMRegressor(boosting_type='dart', n_estimators=60,  learning_rate=0.10, verbose=-1),
        'LightX_GOSS':      lgb.LGBMRegressor(boosting_type='goss', n_estimators=50,  learning_rate=0.05, verbose=-1),
        'LightX_GBDT':      lgb.LGBMRegressor(boosting_type='gbdt', n_estimators=120, max_depth=7,        verbose=-1),

        # -- XGBOOST --
        'XGB_Hizli':   xgb.XGBRegressor(n_estimators=50,  max_depth=4, learning_rate=0.10, verbosity=0),
        'XGB_Derin':   xgb.XGBRegressor(n_estimators=150, max_depth=6, learning_rate=0.05, verbosity=0),
        'XGB_Agresif': xgb.XGBRegressor(n_estimators=100, max_depth=8, learning_rate=0.15, verbosity=0),

        # -- RANDOM FOREST --
        'RF_Klasik': RandomForestRegressor(n_estimators=100, max_depth=8),
        'RF_Derin':  RandomForestRegressor(n_estimators=200, max_depth=None),

        # -- EXTRA TREES --
        'ET_Hizli': ExtraTreesRegressor(n_estimators=100),
        'ET_Derin': ExtraTreesRegressor(n_estimators=200),

        # -- KNN --
        'KNN_5':  KNeighborsRegressor(n_neighbors=5),
        'KNN_10': KNeighborsRegressor(n_neighbors=10),

        # -- LINEAR --
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.01),

        # -- NN --
        'NN_Sinir': MLPRegressor(hidden_layer_sizes=(32, 16), activation='relu', max_iter=1000, random_state=42),
        'NN_Buyuk': MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', max_iter=1000, random_state=42),
    }


def mark_on_image(img_path: str, out_path: str, x_px: int, y_px: int, label: str):
    """Draws a crosshair + circle + text on the given image and saves it."""
    if not os.path.exists(img_path):
        print(f"  [!] Image not found: {img_path}")
        return
    img = cv2.imread(img_path)
    if img is None:
        print(f"  [!] Image could not be read: {img_path}")
        return
    cv2.drawMarker(img, (x_px, y_px), (0, 255, 255),
                   markerType=cv2.MARKER_CROSS, markerSize=50, thickness=2)
    cv2.circle(img, (x_px, y_px), 20, (0, 0, 255), 2)
    cv2.putText(img, label, (x_px + 12, y_px - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, img)
    print(f"  Marked -> {out_path}")


def mark_on_heatmap_array(arr: np.ndarray, out_path: str, x_px: int, y_px: int, label: str):
    """Colors the raw heatmap ndarray and marks the target point on it."""
    clean = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    vmax = clean.max()
    if vmax == 0:
        vmax = 1.0
    norm    = np.clip((clean / vmax) * 255, 0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)
    cv2.drawMarker(colored, (x_px, y_px), (0, 255, 255),
                   markerType=cv2.MARKER_CROSS, markerSize=50, thickness=2)
    cv2.circle(colored, (x_px, y_px), 20, (0, 0, 255), 2)
    cv2.putText(colored, label, (x_px + 12, y_px - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, colored)
    print(f"  Heatmap marked -> {out_path}")


def print_step_summary(tur: int, x_px: int, y_px: int,
                        x_mm: float, y_mm: float,
                        vantuz_mm: float,
                        hmap_vals: dict, model_preds: dict,
                        komite_pred: float, n_samples: int):
    """Prints a detailed summary to the console at every step."""
    sep = "=" * 62
    print(f"\n{sep}")
    print(f"  ROUND {tur:03d}  |  Memory: {n_samples} experiences")
    print(sep)
    print(f"  Pixel coordinate : X={x_px},  Y={y_px}")
    print(f"  Robot coordinate : X={x_mm:.2f} mm,  Y={y_mm:.2f} mm")
    print(f"  Suction cup width : {vantuz_mm:.1f} mm")
    print()
    print("  -- Heatmap values ------------------------------")
    for k, v in hmap_vals.items():
        lbl = HEATMAP_LABELS.get(k, k)
        print(f"    {lbl:30s}: {v:.4f}")
    print()
    if n_samples >= 3:
        print("  -- Model predictions (0.0 - 1.0) ------------------")
        for name, pred in model_preds.items():
            bar_len = int(pred * 20)
            bar = "#" * bar_len + "." * (20 - bar_len)
            print(f"    {name:22s}: [{bar}] {pred:.3f}")
        print(f"\n  COMMITTEE PREDICTION : {komite_pred:.3f} / 1.0")
    else:
        print("  [INFO] Not enough experience yet (<3) -> exploration mode active")
    print(sep)


# ==============================================================
# 3. MAIN PROGRAM
# ==============================================================

def main():
    print("\n" + "+" + "="*60 + "+")
    print("|" + "  UNiFiER v3 - active-learning robot coordinate selector  ".center(60) + "|")
    print("+" + "="*60 + "+\n")

    out_dir = PATHS['output_dir']
    os.makedirs(out_dir, exist_ok=True)

    memory_csv        = os.path.join(out_dir, "hafiza.csv")
    report_xlsx       = os.path.join(out_dir, "rapor.xlsx")
    models_dir        = os.path.join(out_dir, "modeller")
    session_cfg_path  = os.path.join(out_dir, "session_config.pkl")
    os.makedirs(models_dir, exist_ok=True)

    # -- Load the heatmaps --------------------------------------
    print("Loading the heatmaps...")
    maps = {}
    for key in ACTIVE_HEATMAPS:
        maps[key] = load_heatmap(PATHS[key], HEATMAP_LABELS[key])

    feature_keys = list(maps.keys())

    # -- H matrix -----------------------------------------------
    H = np.load(PATHS['H_npy'])
    print(f"\n  [OK] H matrix loaded.\n{H}\n")

    # -- Valid pixel space --------------------------------------
    df_space = build_valid_space(maps)

    # -- Session configuration: suction cup width ----------------
    # Asked again only when the program is restarted; not asked again in the same session.
    if os.path.exists(session_cfg_path):
        with open(session_cfg_path, 'rb') as f:
            cfg = pickle.load(f)
        vantuz_mm = cfg['vantuz_mm']
        print(f"  Existing session configuration loaded:")
        print(f"     Suction cup width: {vantuz_mm:.1f} mm")
        print(f"     Success scale    : 0 or 1 (fixed)")
    else:
        print("\n" + "-"*62)
        print("  SESSION CONFIGURATION (asked once every time the program starts)")
        print("-"*62)
        while True:
            try:
                vantuz_mm = float(input("  Enter the suction cup width (mm) (e.g. 30.0): ").strip())
                if vantuz_mm > 0:
                    break
            except ValueError:
                pass
            print("  [!] Enter a valid positive number.")
        with open(session_cfg_path, 'wb') as f:
            pickle.dump({'vantuz_mm': vantuz_mm}, f)
        print(f"  [OK] Suction cup width locked: {vantuz_mm:.1f} mm")
        print(f"  [OK] Success scale locked: 0 or 1 (fixed)")

    # -- Models and confidence scores ----------------------------
    models       = get_fresh_models()
    trust_scores = {name: 1.0 for name in models}

    # -- Load or create the report ------------------------------
    if os.path.exists(report_xlsx):
        df_report = pd.read_excel(report_xlsx)
    else:
        cols = (
            ['Round', 'Tarih', 'X_Piksel', 'Y_Piksel', 'X_mm', 'Y_mm', 'Vantuz_mm']
            + [HEATMAP_LABELS[k] for k in feature_keys]
            + list(models.keys())
            + ['Komite_Tahmini', 'Gercek_Puan']
        )
        df_report = pd.DataFrame(columns=cols)

    # ========================================================
    # ACTIVE LEARNING LOOP
    # ========================================================
    while True:

        # Load the memory
        if os.path.exists(memory_csv):
            df_memory = pd.read_csv(memory_csv)
        else:
            df_memory = pd.DataFrame(columns=['x_px', 'y_px'] + feature_keys + ['score'])

        n_samples = len(df_memory)
        tur_no    = n_samples + 1

        komite_pred = 0.0
        model_preds = {name: 0.0 for name in models}

        # -- DECISION: ENOUGH DATA? -------------
        if n_samples < 3:
            # Exploration: a random safe pixel with the highest heatmap_A value
            best_row = (df_space
                        .sort_values(by='heatmap_A', ascending=False)
                        .head(100)
                        .sample(1)
                        .iloc[0])
        else:
            X_train = df_memory[feature_keys].values
            y_train = df_memory['score'].values

            scaler      = MinMaxScaler()
            X_train_sc  = scaler.fit_transform(X_train)
            X_space_raw = df_space[feature_keys].values
            X_space_sc  = scaler.transform(X_space_raw)

            preds_space = {}
            for name, model in models.items():
                if 'NN' in name:
                    model.fit(X_train_sc, y_train)
                    preds_space[name] = np.clip(model.predict(X_space_sc), 0.0, 1.0)
                else:
                    model.fit(X_train, y_train)
                    preds_space[name] = np.clip(model.predict(X_space_raw), 0.0, 1.0)
                with open(os.path.join(models_dir, f"{name}.pkl"), 'wb') as f:
                    pickle.dump(model, f)

            # Update the confidence: from the last prediction error
            if n_samples > 3:
                last       = df_memory.iloc[-1]
                last_X_raw = last[feature_keys].values.reshape(1, -1)
                last_X_sc  = scaler.transform(last_X_raw)
                last_real  = last['score']
                for name, model in models.items():
                    p = model.predict(last_X_sc if 'NN' in name else last_X_raw)[0]
                    trust_scores[name] = 1.0 / (1.0 + abs(p - last_real))

            # Weighted committee score
            total_trust   = sum(trust_scores.values())
            committee_arr = np.zeros(len(df_space))
            for name in models:
                w = trust_scores[name] / total_trust
                committee_arr += preds_space[name] * w

            df_space = df_space.copy()
            df_space['_committee'] = committee_arr

            best_idx    = df_space['_committee'].idxmax()
            best_row    = df_space.loc[best_idx]
            komite_pred = float(best_row['_committee'])
            model_preds = {
                name: float(preds_space[name][df_space.index.get_loc(best_idx)])
                for name in models
            }

        # -- SELECTED PIXEL INFO -----------------------------------
        x_px = int(best_row['x_px'])
        y_px = int(best_row['y_px'])
        x_mm, y_mm = pixel_to_world(x_px, y_px, H)
        hmap_vals = {k: float(best_row[k]) for k in feature_keys}

        # -- SUMMARY OUTPUT ----------------------------------------
        print_step_summary(tur_no, x_px, y_px, x_mm, y_mm,
                           vantuz_mm, hmap_vals, model_preds,
                           komite_pred, n_samples)

        # -- MARKING ON THE IMAGES -------------------------------
        ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
        lbl = f"T{tur_no:03d}"

        mark_on_image(
            PATHS['master_img'],
            os.path.join(out_dir, "isaretli", f"master_T{tur_no:03d}.png"),
            x_px, y_px, lbl
        )
        for k, arr in maps.items():
            mark_on_heatmap_array(
                arr,
                os.path.join(out_dir, "isaretli", f"{k}_T{tur_no:03d}.png"),
                x_px, y_px, lbl
            )

        # -- USER SCORE (only 0 or 1) -------------------------------
        print(f"\n  Move the robot to X={x_mm:.2f} mm, Y={y_mm:.2f} mm and try it.")
        raw = input("  > Enter a score (0 = failure, 1 = success) | Quit: 'q' | Skip: 's': ").strip().lower()

        if raw == 'q':
            print("\n  Exit. Report saved.")
            break

        if raw == 's':
            print("  Skipped.")
            continue

        if raw not in ('0', '1'):
            print("  [!] Invalid input! Enter only 0 or 1.")
            continue

        gercek_puan = float(raw)

        # -- SAVE TO MEMORY ----------------------------------------
        new_exp = {
            'x_px': x_px, 'y_px': y_px,
            **{k: hmap_vals[k] for k in feature_keys},
            'score': gercek_puan,
        }
        df_memory = pd.concat([df_memory, pd.DataFrame([new_exp])], ignore_index=True)
        df_memory.to_csv(memory_csv, index=False)

        # -- SAVE TO THE REPORT -------------------------------------
        rep_row = {
            'Round'           : tur_no,
            'Tarih'         : ts,
            'X_Piksel'      : x_px,
            'Y_Piksel'      : y_px,
            'X_mm'          : round(x_mm, 2),
            'Y_mm'          : round(y_mm, 2),
            'Vantuz_mm'     : vantuz_mm,
            'Komite_Tahmini': round(komite_pred, 3),
            'Gercek_Puan'   : gercek_puan,
        }
        for k in feature_keys:
            rep_row[HEATMAP_LABELS[k]] = round(hmap_vals[k], 4)
        for name, pred in model_preds.items():
            rep_row[name] = round(pred, 3)

        df_report = pd.concat([df_report, pd.DataFrame([rep_row])], ignore_index=True)
        df_report.to_excel(report_xlsx, index=False)

        print(f"  [OK] Round {tur_no} saved. Memory: {n_samples + 1} experiences.")


if __name__ == "__main__":
    main()