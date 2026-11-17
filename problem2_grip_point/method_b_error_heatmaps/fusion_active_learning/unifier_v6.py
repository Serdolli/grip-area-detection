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
  * build_valid_space is vectorized with numpy
  * Gaussian process (Bayesian) exploration with UCB in the first 40 rounds, then the committee
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
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb

PROJECT_ROOT = os.environ.get("GRIP_PROJECT_ROOT", os.path.abspath("project_root"))   # see README

warnings.filterwarnings("ignore")

# The success criterion is fixed - do not change it
MAX_SCORE = 1.0


# ==============================================================
# 1. FILE PATHS
# ==============================================================
PATHS = {
    # -- Heatmap PKL inputs (each an ndarray) --
    'heatmap_A': os.path.join(PROJECT_ROOT, r"codes_results\grippointfinder\unified_point_finder algorithm\unifier_2\test_2\heatmaps_pkl\13_part1_distance_transform.pkl"),
    'heatmap_B': os.path.join(PROJECT_ROOT, r"codes_results\grippointfinder\unified_point_finder algorithm\unifier_2\test_2\heatmaps_pkl\camera_error_heatmap.pkl"),
    'heatmap_C': os.path.join(PROJECT_ROOT, r"codes_results\grippointfinder\unified_point_finder algorithm\unifier_2\test_2\heatmaps_pkl\robot_error_pixel_space.pkl"),
    'heatmap_D': os.path.join(PROJECT_ROOT, r"codes_results\grippointfinder\unified_point_finder algorithm\unifier_2\test_2\transformation\transform_error_1280x1024.pkl"),
    'heatmap_E': os.path.join(PROJECT_ROOT, r"codes_results\grippointfinder\unified_point_finder algorithm\unifier_2\test_2\heatmaps_pkl\condition_number_pixel_space.pkl"),
    'heatmap_F': os.path.join(PROJECT_ROOT, r"codes_results\grippointfinder\unified_point_finder algorithm\unifier_2\test_2\heatmaps_pkl\manipulability_pixel_space.pkl"),
    'heatmap_G': os.path.join(PROJECT_ROOT, r"codes_results\grippointfinder\unified_point_finder algorithm\unifier_2\test_2\heatmaps_pkl\radial_bias_heatmap.pkl"),
    'heatmap_H': os.path.join(PROJECT_ROOT, r"codes_results\grippointfinder\unified_point_finder algorithm\unifier_2\test_2\heatmaps_pkl\randomheatmap.pkl"),

    # -- Homography matrix --
    'H_npy':     os.path.join(PROJECT_ROOT, r"codes_results\grippointfinder\unified_point_finder algorithm\unifier_2\test_2\other_inputs\H.npy"),

    # -- Reference image (the mark is drawn on it) --
    'master_img': os.path.join(PROJECT_ROOT, r"codes_results\grippointfinder\unified_point_finder algorithm\unifier_2\test_2\other_inputs\13_MASTER_TOPOGRAPHY.bmp"),

    # -- Output folder --
    'output_dir': os.path.join(PROJECT_ROOT, r"codes_results\grippointfinder\unified_point_finder algorithm\unifier_3\heatmaps\outputs"),
}

# Heatmap name labels (for readable logs)
HEATMAP_LABELS = {
    'heatmap_A': 'A (mask/distance)',
    'heatmap_B': 'B (camera error)',
    'heatmap_C': 'C (robot error)',
    'heatmap_D': 'D (transform error)',
    'heatmap_E': 'E (condition score)',
    'heatmap_F': 'F (manipulability score)',
    'heatmap_G': 'G (center-focused score)',
    'heatmap_H': 'H (Random Heatmap)',
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
            arr = np.zeros((1024, 1280), dtype=np.float32)
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
    keys  = list(maps.keys())
    first = maps[keys[0]]
    H_img, W_img = first.shape[:2]

    # Stack all maps into a single (8, H, W) array
    stack = np.stack([maps[k] for k in keys], axis=0).astype(np.float32)

    # Valid mask: > 0 and finite (not NaN/inf) in every map at the same time
    valid_mask = np.all((stack > 0) & np.isfinite(stack), axis=0)  # (H, W)

    # Extract the coordinates of the valid pixels
    ys, xs = np.where(valid_mask)

    # Build the DataFrame in one go
    data = {'x_px': xs, 'y_px': ys}
    for k, arr in maps.items():
        data[k] = arr[ys, xs]

    df = pd.DataFrame(data).reset_index(drop=True)
    print(f"  -> Total valid pixels: {len(df):,} / {H_img * W_img:,}")
    if len(df) == 0:
        raise RuntimeError("No valid pixel found! There is no overlapping non-zero region in all maps.")
    return df


def get_fresh_models() -> dict:
    """Committee of tree-based models and an NN."""
    return {
        # -- LIGHTGBM --
        'LGBM_Hizli':       lgb.LGBMRegressor(n_estimators=30,  learning_rate=0.10, num_leaves=7,  verbose=-1),
        'LGBM_Dengeli':     lgb.LGBMRegressor(n_estimators=75,  learning_rate=0.08, feature_fraction=0.8, verbose=-1),
        'LightX_GOSS':      lgb.LGBMRegressor(boosting_type='goss', n_estimators=50,  learning_rate=0.05, verbose=-1),

        # -- XGBOOST --
        'XGB_Derin':        xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.05, verbosity=0),
        'XGB_Hafif':        xgb.XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, verbosity=0),

        # -- ENSEMBLE / TREES --
        'RF_Klasik':        RandomForestRegressor(n_estimators=100, max_depth=8),
        'ExtraTrees':       ExtraTreesRegressor(n_estimators=100, max_depth=8),

        # -- NN (already gets 0-1 scaled data through the if 'NN' in name block) --
        'NN_Sinir':         MLPRegressor(hidden_layer_sizes=(32, 16), activation='relu', max_iter=1000, random_state=42),
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


def bayesian_explore(df_space: pd.DataFrame,
                     df_memory: pd.DataFrame,
                     feature_keys: list,
                     scaler: MinMaxScaler) -> pd.Series:
    """
    Bayesian exploration with a Gaussian Process (for the first 40 rounds).

    Strategy: UCB (Upper Confidence Bound)
      UCB = mu + kappa * sigma
      mu    = mean success probability predicted by the GP
      sigma = uncertainty predicted by the GP (standard deviation)
      kappa = exploration/exploitation balance (2.0 = balanced)

    Because of the computational cost, the GP works on a subset of 2000 candidates
    chosen by the best heatmap_A values instead of the whole df_space.
    The pixel with the highest UCB score in this subset is chosen.
    """
    KAPPA        = 2.0   # exploration/exploitation balance: larger means more exploration
    CANDIDATE_N  = 2000  # Number of candidate pixels evaluated by the GP

    X_train = df_memory[feature_keys].values
    y_train = df_memory['score'].values.astype(int)

    # The GP does not work if all classes are the same -> fall back to simple exploration with heatmap_A
    if len(np.unique(y_train)) < 2:
        return (df_space
                .sort_values(by='heatmap_A', ascending=False)
                .head(100)
                .sample(1)
                .iloc[0])

    # GP model: RBF kernel, normalized features
    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
    gp     = GaussianProcessClassifier(kernel=kernel, max_iter_predict=100,
                                       warm_start=False, random_state=42)

    X_train_sc = scaler.transform(X_train)
    gp.fit(X_train_sc, y_train)

    # Candidate subset - the best 2000 pixels by heatmap_A
    candidates = (df_space
                  .sort_values(by='heatmap_A', ascending=False)
                  .head(CANDIDATE_N)
                  .copy())

    X_cand = scaler.transform(candidates[feature_keys].values)

    # GP probability prediction (for class=1) and UCB computation
    proba      = gp.predict_proba(X_cand)[:, 1]          # mu-like
    # Uncertainty proxy of the GP classifier: the most undecided point is near p=0.5
    # sigma proxy = 1 - |2p - 1|  (close to 0.5 -> high uncertainty)
    uncertainty = 1.0 - np.abs(2.0 * proba - 1.0)
    ucb_scores  = proba + KAPPA * uncertainty

    best_local_idx          = np.argmax(ucb_scores)
    candidates['_ucb']      = ucb_scores
    best_row                = candidates.iloc[best_local_idx]

    print(f"  Bayesian exploration (GP-UCB) | "
          f"mu={proba[best_local_idx]:.3f}  "
          f"sigma={uncertainty[best_local_idx]:.3f}  "
          f"UCB={ucb_scores[best_local_idx]:.3f}")

    return best_row


def print_step_summary(tur: int, x_px: int, y_px: int,
                        x_mm: float, y_mm: float,
                        vantuz_mm: float,
                        hmap_vals: dict, model_preds: dict,
                        komite_pred: float, n_samples: int,
                        exploration_turns: int = 40):
    """Prints a detailed summary to the console at every step."""
    sep = "=" * 62

    if n_samples < 3:
        faz = "Phase 1 - initial exploration (heatmap_A)"
    elif n_samples < exploration_turns:
        faz = f"Phase 2 - Bayesian exploration (GP-UCB)  [{n_samples}/{exploration_turns}]"
    else:
        faz = "Phase 3 - committee mode (21 models)"

    print(f"\n{sep}")
    print(f"  ROUND {tur:03d}  |  {faz}")
    print(f"  Memory: {n_samples} experiences")
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
    if n_samples >= exploration_turns:
        print("  -- Model predictions (0.0 - 1.0) ------------------")
        for name, pred in model_preds.items():
            bar_len = int(pred * 20)
            bar = "#" * bar_len + "." * (20 - bar_len)
            print(f"    {name:22s}: [{bar}] {pred:.3f}")
        print(f"\n  COMMITTEE PREDICTION : {komite_pred:.3f} / 1.0")
    elif n_samples >= 3:
        print("  [INFO] The committee model does not run during the Bayesian exploration phase.")
        print(f"     Committee mode starts at round {exploration_turns}.")
    else:
        print("  [INFO] Not enough experience yet (<3) -> initial exploration active")
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
        print(f"  [INFO] If the suction cup changes, delete this file: {session_cfg_path}")

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

        # -- DECISION: WHICH MODE? -----------------------------------
        #
        #  Round  1 -  2 : not enough data for the GP yet (at least 2 classes needed)
        #                -> simple random exploration by heatmap_A
        #  Round  3 - 40 : Bayesian exploration (GP-UCB)
        #  Round 41+     : full committee mode (21 models, weighted vote)
        #
        EXPLORATION_TURNS = 40  # Change this value to extend the phase

        if n_samples < 3:
            # Not enough data for the GP yet -> simple heatmap_A exploration
            # (heatmap_A is a distance transform, so it shows the safest regions)
            best_row = (df_space
                        .sort_values(by='heatmap_A', ascending=False)
                        .head(100)
                        .sample(1)
                        .iloc[0])

        elif n_samples < EXPLORATION_TURNS:
            # -- BAYESIAN EXPLORATION PHASE (rounds 3-40) ----------------
            # The scaler has to be fitted here too because the GP expects normalized input
            scaler_exp = MinMaxScaler()
            scaler_exp.fit(df_memory[feature_keys].values)
            best_row = bayesian_explore(df_space, df_memory, feature_keys, scaler_exp)

        else:
            # -- FULL COMMITTEE MODE (round 41+) -------------------------
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
            if n_samples > EXPLORATION_TURNS:
                last       = df_memory.iloc[-1]
                last_X_raw = last[feature_keys].values.reshape(1, -1)
                last_X_sc  = scaler.transform(last_X_raw)
                last_real  = last['score']
                for name, model in models.items():
                    X_input = last_X_sc if 'NN' in name else last_X_raw
                    p = model.predict(X_input)[0]
                    trust_scores[name] = 1.0 / (1.0 + abs(p - last_real))

            # Weighted committee score
            total_trust   = sum(trust_scores.values())
            committee_arr = np.zeros(len(df_space))
            for name in models:
                w = trust_scores[name] / total_trust
                committee_arr += preds_space[name] * w

            # df_space is not modified - committee stays a local Series
            committee_series = pd.Series(committee_arr, index=df_space.index)
            best_idx         = committee_series.idxmax()
            best_row         = df_space.loc[best_idx]
            komite_pred      = float(committee_series[best_idx])
            loc              = df_space.index.get_loc(best_idx)
            model_preds      = {name: float(preds_space[name][loc]) for name in models}

        # -- SELECTED PIXEL INFO -----------------------------------
        x_px = int(best_row['x_px'])
        y_px = int(best_row['y_px'])
        x_mm, y_mm = pixel_to_world(x_px, y_px, H)
        hmap_vals = {k: float(best_row[k]) for k in feature_keys}

        # -- SUMMARY OUTPUT ----------------------------------------
        print_step_summary(tur_no, x_px, y_px, x_mm, y_mm,
                           vantuz_mm, hmap_vals, model_preds,
                           komite_pred, n_samples,
                           exploration_turns=40)

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
