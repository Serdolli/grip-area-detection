import os
import torch
import torchvision
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torch.utils.data import DataLoader, Dataset
from pycocotools.coco import COCO
import numpy as np
from PIL import Image

PROJECT_ROOT = os.environ.get("GRIP_PROJECT_ROOT", os.path.abspath("project_root"))   # see README

# ==============================================================================
# 1. FILE PATHS AND HYPERPARAMETERS
# ==============================================================================

BASE_DIR = os.path.join(PROJECT_ROOT, r"datasets\derivative_datasets\maskrcnn_320x3200_rotated")

# Train paths
TRAIN_IMG_DIR = os.path.join(BASE_DIR, "train")
TRAIN_ANN_FILE = os.path.join(BASE_DIR, "train", "_annotations.coco.json")

# Validation paths (used to check whether the model memorizes)
VAL_IMG_DIR = os.path.join(BASE_DIR, "valid")
VAL_ANN_FILE = os.path.join(BASE_DIR, "valid", "_annotations.coco.json")

MODEL_SAVE_PATH = "sametkalip_maskrcnn_160x1600_2_master.pth"

# --- Architecture settings ---
NUM_CLASSES = 4             # 0: background, 1: alt, 2: griparea, 3: ust
IMG_MIN_SIZE = 160          # Prevent torchvision from breaking our padding
IMG_MAX_SIZE = 1600

# --- Training settings ---
BATCH_SIZE = 1              # 320x3200 strains the VRAM, 2 is ideal (use 1 if you get OOM)
NUM_EPOCHS = 30             
LEARNING_RATE = 0.002       # Learning in small steps
MOMENTUM = 0.9              
WEIGHT_DECAY = 0.0005       

# --- Hardware determinism ---
torch.backends.cudnn.benchmark = True # Input size is fixed, so enable the cuDNN fast path

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# ==============================================================================
# 2. DATASET READER (COCO FORMAT)
# ==============================================================================

class SametKalipDataset(Dataset):
    def __init__(self, root, annotation):
        self.root = root
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        coco_annotation = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert("RGB")
        img = torchvision.transforms.functional.to_tensor(img)

        num_objs = len(coco_annotation)
        boxes, labels, masks = [], [], []

        for i in range(num_objs):
            xmin = coco_annotation[i]['bbox'][0]
            ymin = coco_annotation[i]['bbox'][1]
            xmax = xmin + coco_annotation[i]['bbox'][2]
            ymax = ymin + coco_annotation[i]['bbox'][3]
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(coco_annotation[i]['category_id'])
            masks.append(coco.annToMask(coco_annotation[i]))

        if len(boxes) > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            masks = torch.as_tensor(np.array(masks), dtype=torch.uint8)
        else:
            boxes = torch.empty((0, 4), dtype=torch.float32)
            labels = torch.empty((0,), dtype=torch.int64)
            masks = torch.empty((0, img.shape[1], img.shape[2]), dtype=torch.uint8)
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = torch.tensor([img_id])

        return img, target

    def __len__(self):
        return len(self.ids)

def collate_fn(batch):
    return tuple(zip(*batch))

# ==============================================================================
# 3. MODEL BUILDER
# ==============================================================================

def get_custom_maskrcnn(num_classes):
    print("Loading thin, elongated anchors and the 28x28 mask head...")
    
    # HYPERPARAMETER 1: Custom anchors for thin, elongated parts
    anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
    aspect_ratios = ((1.0, 2.0, 5.0, 10.0),) * len(anchor_sizes) # Vertical and square only
    rpn_anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)

    # HYPERPARAMETER 2: High-resolution mask pooler for sharp edges
    mask_roi_pool = MultiScaleRoIAlign(
        featmap_names=['0', '1', '2', '3'], 
        output_size=28, # Standard 14 changed to 28: 4x finer mask
        sampling_ratio=2
    )

    # Build the model with weights=None first (avoids the mismatched-anchor error)
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(
        weights=None,
        min_size=IMG_MIN_SIZE,
        max_size=IMG_MAX_SIZE,
        rpn_anchor_generator=rpn_anchor_generator,
        mask_roi_pool=mask_roi_pool,
        box_detections_per_img=25, 
        rpn_pre_nms_top_n_train=1000,
        rpn_post_nms_top_n_train=500
    )

    # Load the backbone weights by hand
    print("Loading backbone weights...")
    state_dict = torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.COCO_V1.get_state_dict(progress=True)
    for key in list(state_dict.keys()):
        if 'rpn' in key or 'roi_heads' in key:
            del state_dict[key]
    model.load_state_dict(state_dict, strict=False)

    # Set up the classification and box predictor
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Set up the mask predictor
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    return model

# ==============================================================================
# 4. TRAINING AND VALIDATION LOOP
# ==============================================================================

def main():
    print(f"Samet Kalip master plan training starts! Device: {device}")
    
    # Prepare the DataLoaders
    print("Loading datasets...")
    train_dataset = SametKalipDataset(TRAIN_IMG_DIR, TRAIN_ANN_FILE)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    val_dataset = SametKalipDataset(VAL_IMG_DIR, VAL_ANN_FILE)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # Prepare the model and the optimizer
    model = get_custom_maskrcnn(NUM_CLASSES)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    
    # Scheduler that divides the learning rate by 10 at epoch 15 (fine-tuning)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

    print("Training started!\n" + "="*60)

    for epoch in range(NUM_EPOCHS):
        
        # --- TRAIN STAGE ---
        model.train() 
        train_epoch_loss = 0
        train_mask_loss = 0

        for images, targets in train_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            train_epoch_loss += losses.item()
            train_mask_loss += loss_dict["loss_mask"].item()

        lr_scheduler.step()
        avg_train_loss = train_epoch_loss / len(train_loader)
        avg_train_mask_loss = train_mask_loss / len(train_loader)

        # --- VALIDATION STAGE ---
        val_epoch_loss = 0
        val_mask_loss = 0
        
        with torch.no_grad(): 
            for val_images, val_targets in val_loader:
                val_images = list(image.to(device) for image in val_images)
                val_targets = [{k: v.to(device) for k, v in t.items()} for t in val_targets]

                val_loss_dict = model(val_images, val_targets)
                val_losses = sum(loss for loss in val_loss_dict.values())

                val_epoch_loss += val_losses.item()
                val_mask_loss += val_loss_dict["loss_mask"].item()

        avg_val_loss = val_epoch_loss / len(val_loader)
        avg_val_mask_loss = val_mask_loss / len(val_loader)

        # TERMINAL REPORT
        print(f"Epoch [{epoch+1:02d}/{NUM_EPOCHS}]")
        print(f"  TRAIN -> Total: {avg_train_loss:.4f} | Mask: {avg_train_mask_loss:.4f}")
        print(f"  VALID -> Total: {avg_val_loss:.4f} | Mask: {avg_val_mask_loss:.4f}")
        print("-" * 60)

    # Save the model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\n[OK] Training finished! Model saved: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()