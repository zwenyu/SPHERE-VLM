import json, os
import torch
from torch.utils.data import Dataset
from PIL import Image

class COCODataset(Dataset):
    def __init__(self, annotations_dir, annotations_json, save_suffix, img_dir,
        transform=None, raise_error=True):
        self.annotations_dir = annotations_dir
        self.annotations_json = annotations_json
        self.img_dir = img_dir
        self.transform = transform
        self.raise_error = raise_error

        if os.path.exists(os.path.join(annotations_dir, f"{annotations_json}_{save_suffix}.json")):
            with open(os.path.join(annotations_dir, f"{annotations_json}_{save_suffix}.json")) as f:
                self.coco_annotations = json.load(f)
        else:
            with open(os.path.join(annotations_dir, annotations_json + ".json"), "r") as f:
                self.coco_annotations = json.load(f)

    def __len__(self):
        return len(self.coco_annotations)

    def __getitem__(self, idx):
        ann = self.coco_annotations[idx]
        source_img_id = ann["metadata"]["source_img_id"]
        full_source_img_id = (12 - len(source_img_id)) * "0" + source_img_id
        img_path = f"{self.img_dir}/{full_source_img_id}.jpg"
        try:
            image = Image.open(img_path).convert("RGB")
        except:
            image = None
            if self.raise_error:
                raise ValueError(f"For Qn {ann['question_id']}: Image {img_path} not found.")

        if self.transform:
            image = self.transform(image)

        return idx, image, img_path, ann

