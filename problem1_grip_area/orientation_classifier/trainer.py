import os
import glob
import numpy as np
import joblib
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tqdm import tqdm

PROJECT_ROOT = os.environ.get("GRIP_PROJECT_ROOT", os.path.abspath("project_root"))   # see README

# =============================================================================
# --- 1. SETTINGS (CONFIG) ---
# =============================================================================

# Main folder with the TXT files
DATASET_PATH = os.path.join(PROJECT_ROOT, r"datasets\derivative_datasets\cropped_datasets\cropped_for_rf\optimizing_datasets\trial_clahe+tophat_2")

# Where the model and the scaler are saved
OUTPUT_MODEL_DIR = os.path.join(PROJECT_ROOT, r"codes_results\sidecheckermethodes\1d_adaptive_lightx\final_model")

LABEL_MAP = {"alt": 0, "ust": 1}

# =============================================================================
# --- 2. LOAD THE DATA (WITH FILE NAMES) ---
# =============================================================================

def load_data_from_txt(dataset_path):
    X = []          # signals
    y = []          # labels
    filenames = []  # file names

    print(f"Loading the dataset: {dataset_path}")
    
    for label_name, label_id in LABEL_MAP.items():
        folder_path = os.path.join(dataset_path, label_name)
        txt_files = glob.glob(os.path.join(folder_path, "*.txt"))
        
        print(f"  -> reading class '{label_name}' ({len(txt_files)} files)...")
        
        for txt_file in tqdm(txt_files, desc=f"Loading {label_name}"):
            try:
                with open(txt_file, 'r') as f:
                    content = f.read().strip()
                    signal = np.fromstring(content, sep=',')
                
                if signal.shape[0] > 0:
                    X.append(signal)
                    y.append(label_id)
                    # only the file name (e.g. "5_flip.txt")
                    filenames.append(os.path.basename(txt_file))
            except Exception as e:
                print(f"Error: {txt_file} -> {e}")

    return np.array(X), np.array(y), np.array(filenames)

# =============================================================================
# --- 3. TRAINING ---
# =============================================================================

def train_lightx():
    if not os.path.exists(OUTPUT_MODEL_DIR):
        os.makedirs(OUTPUT_MODEL_DIR)

    # 1. Load the data (with the file names)
    X, y, filenames = load_data_from_txt(DATASET_PATH)
    
    print("-" * 50)
    print(f"TOTAL SAMPLES: {len(X)}")
    print(f"Signal length (input nodes): {X.shape[1] if len(X) > 0 else 0}")
    print("-" * 50)

    if len(X) == 0:
        print("ERROR: no data found!")
        return

    # 2. Train / test split
    # the file names are split in the same order
    X_train, X_test, y_train, y_test, files_train, files_test = train_test_split(
        X, y, filenames, test_size=0.01, random_state=42, stratify=y
    )

    print(f"Training set: {len(X_train)}")
    print(f"Test Seti:   {len(X_test)}")

    # 3. Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Model
    print(">>> Training the model (LightGBM)...")
    model = lgb.LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.05,
        num_leaves=93,
        objective='binary',
        random_state=42,
        verbose=-1
    )

    # 5. Train
    model.fit(X_train_scaled, y_train)
    print(">>> Training finished.")

    # 6. Test
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    
    print("\n" + "="*30)
    print(f"TEST ACCURACY: %{acc * 100:.2f}")
    print("="*30)
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred)) 
    
    print("\nDetailed report:")
    print(classification_report(y_test, y_pred, target_names=["ALT (0)", "UST (1)"]))

    # --- 7. LIST THE MISCLASSIFIED FILES ---
    print("\n" + "!"*40)
    print("!!! MISCLASSIFIED FILES !!!")
    print("!"*40)
    
    # indices of the wrong predictions
    wrong_indices = np.where(y_test != y_pred)[0]
    
    if len(wrong_indices) == 0:
        print("No errors on the test set.")
    else:
        for idx in wrong_indices:
            fname = files_test[idx]
            
            # readable labels
            real_label = "UST" if y_test[idx] == 1 else "ALT"
            pred_label = "UST" if y_pred[idx] == 1 else "ALT"
            
            print(f"File: {fname:<15} | True: {real_label} -> Predicted: {pred_label}")

    print("!"*40 + "\n")

    # 8. Save
    model_save_path = os.path.join(OUTPUT_MODEL_DIR, "lgbm_model.pkl")
    scaler_save_path = os.path.join(OUTPUT_MODEL_DIR, "scaler.pkl")

    joblib.dump(model, model_save_path)
    joblib.dump(scaler, scaler_save_path)

    print(f"[OK] MODEL SAVED: {model_save_path}")

if __name__ == "__main__":
    train_lightx()