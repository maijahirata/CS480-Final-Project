# coco_mask_extractor.py
import os
import cv2
from pycocotools.coco import COCO

COCO_VAL_IMAGES = "val2017/"
COCO_VAL_ANN = "annotations/instances_val2017.json"
OUTPUT_MASK_DIR = "coco_gt_masks/"

os.makedirs(OUTPUT_MASK_DIR, exist_ok=True)
coco = COCO(COCO_VAL_ANN)

def get_coco_masks(filename):
    img_id = None
    for img in coco.dataset["images"]:
        if img["file_name"] == filename:
            img_id = img["id"]
            break

    if img_id is None:
        raise FileNotFoundError(f"{filename} not found in COCO annotation JSON.")

    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)

    if len(anns) == 0:
        print(f"No annotations found for {filename}.")
        return []

    masks = []
    for ann in anns:
        mask = coco.annToMask(ann)
        masks.append(mask)

        out_path = os.path.join(
            OUTPUT_MASK_DIR,
            f"{filename.replace('.jpg','')}_ann{ann['id']}.png"
        )
        cv2.imwrite(out_path, mask * 255)

    return masks
