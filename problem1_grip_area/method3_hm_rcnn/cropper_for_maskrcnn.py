import os
import json
import cv2
import numpy as np

PROJECT_ROOT = os.environ.get("GRIP_PROJECT_ROOT", os.path.abspath("project_root"))   # see README

# ==============================================================================
# SETTINGS
# ==============================================================================

# Root folder of the original data (contains train, valid, test)
INPUT_BASE_DIR = os.path.join(PROJECT_ROOT, r"datasets\derivative_datasets\maskrcnn_600p_700x700")

# Root folder where the new, clean images will be saved
OUTPUT_BASE_DIR = os.path.join(PROJECT_ROOT, r"datasets\derivative_datasets\maskrcnn_320x3200_rotated")

TARGET_W = 320
TARGET_H = 3200

# Category names exactly as written in the COCO JSON
MAIN_PARTS = ["alt", "ust"] 
GRIP_AREA = ["griparea"]

# ==============================================================================
# GEOMETRY AND MATH FUNCTIONS
# ==============================================================================

def transform_polygon(polygon, M):
    pts = np.array(polygon).reshape(-1, 2)
    pts_expanded = np.expand_dims(pts, axis=1)
    transformed_pts = cv2.transform(pts_expanded, M)
    return transformed_pts.flatten().tolist()

def match_parts_and_grips(annotations, categories):
    main_anns = []
    grip_anns = []
    
    main_cat_ids = [c['id'] for c in categories if c['name'].lower() in MAIN_PARTS]
    grip_cat_ids = [c['id'] for c in categories if c['name'].lower() in GRIP_AREA]
    
    for ann in annotations:
        if ann['category_id'] in main_cat_ids:
            main_anns.append(ann)
        elif ann['category_id'] in grip_cat_ids:
            grip_anns.append(ann)
            
    matched_pairs = []
    
    for m_ann in main_anns:
        m_pts = np.array(m_ann['segmentation'][0]).reshape(-1, 2).astype(np.float32)
        my_grips = []
        for g_ann in grip_anns:
            g_pts = np.array(g_ann['segmentation'][0]).reshape(-1, 2)
            g_center = g_pts.mean(axis=0)
            if cv2.pointPolygonTest(m_pts, (g_center[0], g_center[1]), False) >= 0:
                my_grips.append(g_ann)
        matched_pairs.append((m_ann, my_grips))
        
    return matched_pairs

def process_single_part(image, main_ann, grip_anns):
    h_orig, w_orig = image.shape[:2]
    pts = np.array(main_ann['segmentation'][0]).reshape(-1, 2).astype(np.float32)
    
    rect = cv2.minAreaRect(pts)
    center, size, angle = rect
    rect_w, rect_h = size
    
    if rect_w > rect_h:
        angle -= 90
        rect_w, rect_h = rect_h, rect_w
        
    M_rot = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    cos = np.abs(M_rot[0, 0])
    sin = np.abs(M_rot[0, 1])
    nW = int((h_orig * sin) + (w_orig * cos))
    nH = int((h_orig * cos) + (w_orig * sin))
    M_rot[0, 2] += (nW / 2) - center[0]
    M_rot[1, 2] += (nH / 2) - center[1]
    
    rotated_img = cv2.warpAffine(image, M_rot, (nW, nH), flags=cv2.INTER_LINEAR, borderValue=(0,0,0))
    
    new_center = np.dot(M_rot, np.array([center[0], center[1], 1.0]))
    x_min = max(0, int(new_center[0] - rect_w / 2))
    x_max = min(nW, int(new_center[0] + rect_w / 2))
    y_min = max(0, int(new_center[1] - rect_h / 2))
    y_max = min(nH, int(new_center[1] + rect_h / 2))
    
    cropped_img = rotated_img[y_min:y_max, x_min:x_max]
    
    if cropped_img.shape[0] == 0 or cropped_img.shape[1] == 0:
        return None, None
        
    crop_h, crop_w = cropped_img.shape[:2]
    scale = min(TARGET_W / crop_w, TARGET_H / crop_h)
    
    new_w = int(crop_w * scale)
    new_h = int(crop_h * scale)
    resized_img = cv2.resize(cropped_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    pad_w = TARGET_W - new_w
    pad_h = TARGET_H - new_h
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    
    final_img = cv2.copyMakeBorder(resized_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    
    updated_anns = []
    for ann in [main_ann] + grip_anns:
        poly = ann['segmentation'][0]
        poly_rot = transform_polygon(poly, M_rot)
        poly_crop = [p - x_min if i % 2 == 0 else p - y_min for i, p in enumerate(poly_rot)]
        poly_scale = [p * scale for p in poly_crop]
        poly_final = [p + left if i % 2 == 0 else p + top for i, p in enumerate(poly_scale)]
        
        xs = poly_final[0::2]
        ys = poly_final[1::2]
        new_bbox = [min(xs), min(ys), max(xs)-min(xs), max(ys)-min(ys)]
        
        new_ann = ann.copy()
        new_ann['segmentation'] = [poly_final]
        new_ann['bbox'] = new_bbox
        new_ann['area'] = new_bbox[2] * new_bbox[3]
        updated_anns.append(new_ann)

    return final_img, updated_anns

# ==============================================================================
# MAIN ENGINE (folder by folder)
# ==============================================================================

def process_folder(split_name):
    """Processes one split folder (train, valid, test)."""
    input_dir = os.path.join(INPUT_BASE_DIR, split_name)
    json_path = os.path.join(input_dir, "_annotations.coco.json")
    
    if not os.path.exists(json_path):
        print(f"[WARNING] Folder or JSON for {split_name} not found, skipping.")
        return

    output_dir = os.path.join(OUTPUT_BASE_DIR, split_name)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nProcessing folder [{split_name.upper()}]...")

    with open(json_path, 'r') as f:
        coco = json.load(f)
        
    categories = coco['categories']
    
    img_dict = {img['id']: img for img in coco['images']}
    ann_dict = {}
    for ann in coco['annotations']:
        ann_dict.setdefault(ann['image_id'], []).append(ann)

    split_coco = {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "categories": categories,
        "images": [],
        "annotations": []
    }

    # IDs restart for every folder (required by the COCO standard)
    global_img_id = 1
    global_ann_id = 1

    for img_id, img_info in img_dict.items():
        if img_id not in ann_dict: continue
        
        image_path = os.path.join(input_dir, img_info['file_name'])
        image = cv2.imread(image_path)
        if image is None: 
            print(f"  [ERROR] Could not read {img_info['file_name']}.")
            continue
            
        matched_pairs = match_parts_and_grips(ann_dict[img_id], categories)
        
        part_counter = 1
        for main_ann, grip_anns in matched_pairs:
            final_img, updated_anns = process_single_part(image, main_ann, grip_anns)
            
            if final_img is not None:
                base_name = os.path.splitext(img_info['file_name'])[0]
                new_filename = f"{base_name}_part{part_counter}.jpg"
                
                # Save the image
                cv2.imwrite(os.path.join(output_dir, new_filename), final_img)
                
                # Update the JSON
                new_img_info = {
                    "id": global_img_id,
                    "file_name": new_filename,
                    "width": TARGET_W,
                    "height": TARGET_H
                }
                split_coco["images"].append(new_img_info)
                
                for u_ann in updated_anns:
                    u_ann['id'] = global_ann_id
                    u_ann['image_id'] = global_img_id
                    split_coco["annotations"].append(u_ann)
                    global_ann_id += 1
                    
                global_img_id += 1
                part_counter += 1

    # Save the new JSON
    out_json_path = os.path.join(output_dir, "_annotations.coco.json")
    with open(out_json_path, 'w') as f:
        json.dump(split_coco, f, indent=2)
        
    print(f"[OK] [{split_name.upper()}] done! {len(split_coco['images'])} images created.")

def main():
    print("Geometric dataset engine started (existing folders are kept)...")
    
    # Process the train, valid and test folders in turn
    for split in ["train", "valid", "test"]:
        process_folder(split)
        
    print("\nALL DONE! The new dataset is ready.")

if __name__ == "__main__":
    main()