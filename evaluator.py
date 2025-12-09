import os
import cv2
import numpy as np
import torch
import torchvision
import segmentation_models_pytorch as smp
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

from coco_mask_extractor import get_coco_masks

COCO_IMAGE_DIR = "coco"
COCO_IMAGE_FILENAME = "000000049810.jpg"
IMAGE_PATH = os.path.join(COCO_IMAGE_DIR, COCO_IMAGE_FILENAME)

SAM_CHECKPOINT = "sam_vit_h_4b8939.pth"

# Output masks
OUT_SAM = "sam_predicted_mask.png"
OUT_UNET = "unet_predicted_mask.png"
OUT_RCNN = "maskrcnn_predicted_mask.png"

def compute_iou(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return 1.0 if union == 0 else inter / union


def load_mask(path):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path)
    return (mask > 0).astype(np.uint8)

# SAM Segmentation
def run_sam():
    print("\n---- Running SAM without prompts ---- ")

    img_bgr = cv2.imread(IMAGE_PATH)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not load {IMAGE_PATH}")
    h, w = img_bgr.shape[:2]

    sam = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT)
    mask_gen = SamAutomaticMaskGenerator(sam)

    masks = mask_gen.generate(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    print(f"SAM found {len(masks)} masks.")

    combined = np.zeros((h, w), dtype=np.uint8)
    for m in masks:
        combined = np.maximum(combined, m["segmentation"].astype(np.uint8))

    cv2.imwrite(OUT_SAM, combined * 255)
    print(f"SAM mask saved to {OUT_SAM}")

    return combined

# U-Net 
def run_unet():
    print("\n---- Running U-Net ---- ")

    img = cv2.imread(IMAGE_PATH)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]

    # Load model
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
    )
    model.eval()

    # Preprocess
    resized = cv2.resize(img_rgb, (256, 256))
    x = resized / 255.0
    x = np.transpose(x, (2, 0, 1))
    x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        pred = model(x).squeeze().numpy()

    mask = (pred > 0).astype(np.uint8)
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    cv2.imwrite(OUT_UNET, mask * 255)
    print(f"U-Net mask saved to {OUT_UNET}")

    return mask

# Mask R-CNN
def run_maskrcnn():
    print("\n---- Running Mask R-CNN ---- ")

    img = cv2.imread(IMAGE_PATH)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    x = torchvision.transforms.functional.to_tensor(img_rgb)
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
    model.eval()

    with torch.no_grad():
        pred = model([x])[0]

    scores = pred["scores"].cpu().numpy()
    if len(scores) == 0:
        print("Mask R-CNN found no objects.")
        mask = np.zeros(img_rgb.shape[:2], dtype=np.uint8)
    else:
        best = scores.argmax()
        mask = (pred["masks"][best, 0].cpu().numpy() > 0.5).astype(np.uint8)

    cv2.imwrite(OUT_RCNN, mask * 255)
    print(f"Mask R-CNN mask saved to {OUT_RCNN}")

    return mask

# Evaluation start
if __name__ == "__main__":
    print("---- Running Evaluator ---- ")

    # Run all models
    mask_sam = run_sam()
    mask_unet = run_unet()
    mask_rcnn = run_maskrcnn()

    # Load COCO ground truth masks
    gt_masks = get_coco_masks(COCO_IMAGE_FILENAME)

    print(f"\nLoaded {len(gt_masks)} COCO Ground Truth masks for {COCO_IMAGE_FILENAME}")

    # IoU table
    results = {
        "SAM": [],
        "U-Net": [],
        "Mask R-CNN": []
    }

    for i, gt in enumerate(gt_masks):
        sam_iou = compute_iou(mask_sam, gt)
        unet_iou = compute_iou(mask_unet, gt)
        rcnn_iou = compute_iou(mask_rcnn, gt)

        results["SAM"].append(sam_iou)
        results["U-Net"].append(unet_iou)
        results["Mask R-CNN"].append(rcnn_iou)

        print(f"\nGround Truth Mask {i}:")
        print(f"  SAM        IoU = {sam_iou:.4f}")
        print(f"  U-Net      IoU = {unet_iou:.4f}")
        print(f"  Mask R-CNN IoU = {rcnn_iou:.4f}")

    # Print summary table
    print(" ---- Final IoU Summary ---- ")
    for model in results:
        best = max(results[model]) if results[model] else 0
        print(f"{model:12s}  Best IoU = {best:.4f}")
