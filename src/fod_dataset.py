from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageEnhance, ImageFilter


class FODDetectionDataset(Dataset):
    """
    PyTorch Dataset for FOD-A object detection.

    Expected YOLO dataset structure:

    yolo/
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    │
    └── labels/
        ├── train/
        ├── val/
        └── test/

    YOLO label format:

    class_id x_center y_center width height

    Coordinates are normalized to [0, 1].

    Augmentation
    ------------
    augment=False
        No augmentation.

    augment=True
        Apply random augmentation.

    augmentation_strength:
        1 -> Mild
        2 -> Medium
        3 -> Strong
    """

    def __init__(
        self,
        root_dir,
        split="train",
        img_size=640,
        transform=None,
        augment=False,
        augmentation_strength=1
    ):

        self.root_dir = Path(root_dir)
        self.split = split
        self.img_size = img_size

        self.transform = transform

        # ----------------------------------------------------
        # Augmentation configuration
        # ----------------------------------------------------

        self.augment = augment

        if augmentation_strength not in [1, 2, 3]:
            raise ValueError(
                "augmentation_strength must be 1, 2, or 3"
            )

        self.augmentation_strength = (
            augmentation_strength
        )

        # ----------------------------------------------------
        # Directories
        # ----------------------------------------------------

        self.image_dir = (
            self.root_dir /
            "images" /
            split
        )

        self.label_dir = (
            self.root_dir /
            "labels" /
            split
        )

        # ----------------------------------------------------
        # Image files
        # ----------------------------------------------------

        self.image_files = sorted(
            self.image_dir.glob("*.jpg")
        )

        if len(self.image_files) == 0:

            raise RuntimeError(
                f"No images found in {self.image_dir}"
            )

    # ========================================================
    # LENGTH
    # ========================================================

    def __len__(self):

        return len(self.image_files)

    # ========================================================
    # READ YOLO LABEL
    # ========================================================

    def read_labels(self, label_path):

        boxes = []
        labels = []

        if not label_path.exists():

            return (
                torch.zeros(
                    (0, 4),
                    dtype=torch.float32
                ),
                torch.zeros(
                    (0,),
                    dtype=torch.long
                )
            )

        with open(
            label_path,
            "r",
            encoding="utf-8"
        ) as f:

            lines = [
                line.strip()
                for line in f
                if line.strip()
            ]

        for line in lines:

            values = line.split()

            if len(values) != 5:
                continue

            try:

                class_id = int(values[0])

                x_center = float(values[1])
                y_center = float(values[2])
                width = float(values[3])
                height = float(values[4])

            except ValueError:

                continue

            labels.append(class_id)

            # ------------------------------------------------
            # YOLO xywh -> normalized xyxy
            # ------------------------------------------------

            xmin = x_center - width / 2
            ymin = y_center - height / 2

            xmax = x_center + width / 2
            ymax = y_center + height / 2

            boxes.append([
                xmin,
                ymin,
                xmax,
                ymax
            ])

        if len(boxes) == 0:

            return (
                torch.zeros(
                    (0, 4),
                    dtype=torch.float32
                ),
                torch.zeros(
                    (0,),
                    dtype=torch.long
                )
            )

        boxes = torch.tensor(
            boxes,
            dtype=torch.float32
        )

        labels = torch.tensor(
            labels,
            dtype=torch.long
        )

        return boxes, labels

    # ========================================================
    # AUGMENTATION
    # ========================================================

    def apply_augmentation(
        self,
        image,
        boxes,
        labels
    ):
        """
        Apply detection-safe augmentations.

        Strength 1:
            Mild

        Strength 2:
            Medium

        Strength 3:
            Strong
        """

        strength = self.augmentation_strength

        # ----------------------------------------------------
        # 1. Horizontal Flip
        # ----------------------------------------------------

        flip_probability = {
            1: 0.30,
            2: 0.50,
            3: 0.60
        }[strength]

        if random.random() < flip_probability:

            image = image.transpose(
                Image.Transpose.FLIP_LEFT_RIGHT
            )

            if len(boxes) > 0:

                boxes = boxes.clone()

                old_xmin = boxes[:, 0].clone()

                old_xmax = boxes[:, 2].clone()

                boxes[:, 0] = 1.0 - old_xmax
                boxes[:, 2] = 1.0 - old_xmin

        # ----------------------------------------------------
        # 2. Brightness
        # ----------------------------------------------------

        brightness_factor = {
            1: (0.90, 1.10),
            2: (0.80, 1.20),
            3: (0.70, 1.30)
        }[strength]

        if random.random() < 0.50:

            factor = random.uniform(
                *brightness_factor
            )

            image = ImageEnhance.Brightness(
                image
            ).enhance(factor)

        # ----------------------------------------------------
        # 3. Contrast
        # ----------------------------------------------------

        contrast_factor = {
            1: (0.90, 1.10),
            2: (0.80, 1.20),
            3: (0.70, 1.30)
        }[strength]

        if random.random() < 0.50:

            factor = random.uniform(
                *contrast_factor
            )

            image = ImageEnhance.Contrast(
                image
            ).enhance(factor)

        # ----------------------------------------------------
        # 4. Color saturation
        # ----------------------------------------------------

        saturation_factor = {
            1: (0.90, 1.10),
            2: (0.80, 1.20),
            3: (0.70, 1.30)
        }[strength]

        if random.random() < 0.30:

            factor = random.uniform(
                *saturation_factor
            )

            image = ImageEnhance.Color(
                image
            ).enhance(factor)

        # ----------------------------------------------------
        # 5. Gaussian Blur
        # ----------------------------------------------------

        blur_probability = {
            1: 0.10,
            2: 0.20,
            3: 0.30
        }[strength]

        if random.random() < blur_probability:

            radius = {
                1: 0.5,
                2: 0.8,
                3: 1.2
            }[strength]

            image = image.filter(
                ImageFilter.GaussianBlur(
                    radius=radius
                )
            )

        return image, boxes, labels

    # ========================================================
    # RESIZE IMAGE + BOXES
    # ========================================================

    def resize_image_and_boxes(
        self,
        image,
        boxes
    ):

        original_width, original_height = (
            image.size
        )

        image = image.resize(
            (
                self.img_size,
                self.img_size
            ),
            Image.Resampling.BILINEAR
        )

        if len(boxes) == 0:

            return image, boxes

        # ----------------------------------------------------
        # Normalized xyxy -> pixel xyxy
        # ----------------------------------------------------

        boxes = boxes.clone()

        boxes[:, 0] *= original_width
        boxes[:, 1] *= original_height
        boxes[:, 2] *= original_width
        boxes[:, 3] *= original_height

        # ----------------------------------------------------
        # Resize scaling
        # ----------------------------------------------------

        scale_x = (
            self.img_size /
            original_width
        )

        scale_y = (
            self.img_size /
            original_height
        )

        boxes[:, 0] *= scale_x
        boxes[:, 1] *= scale_y
        boxes[:, 2] *= scale_x
        boxes[:, 3] *= scale_y

        return image, boxes

    # ========================================================
    # CLIP BOXES TO IMAGE
    # ========================================================

    def clip_boxes(
        self,
        boxes
    ):

        if len(boxes) == 0:
            return boxes

        boxes = boxes.clone()

        boxes[:, 0] = boxes[:, 0].clamp(
            0,
            self.img_size
        )

        boxes[:, 1] = boxes[:, 1].clamp(
            0,
            self.img_size
        )

        boxes[:, 2] = boxes[:, 2].clamp(
            0,
            self.img_size
        )

        boxes[:, 3] = boxes[:, 3].clamp(
            0,
            self.img_size
        )

        return boxes

    # ========================================================
    # REMOVE INVALID BOXES
    # ========================================================

    def remove_invalid_boxes(
        self,
        boxes,
        labels
    ):

        if len(boxes) == 0:

            return boxes, labels

        widths = (
            boxes[:, 2] -
            boxes[:, 0]
        )

        heights = (
            boxes[:, 3] -
            boxes[:, 1]
        )

        valid = (
            (widths > 1.0) &
            (heights > 1.0)
        )

        boxes = boxes[valid]
        labels = labels[valid]

        return boxes, labels

    # ========================================================
    # GET ITEM
    # ========================================================

    def __getitem__(self, index):

        image_path = self.image_files[index]

        # ----------------------------------------------------
        # Load image
        # ----------------------------------------------------

        image = Image.open(
            image_path
        ).convert("RGB")

        # ----------------------------------------------------
        # Label path
        # ----------------------------------------------------

        label_path = (
            self.label_dir /
            f"{image_path.stem}.txt"
        )

        # ----------------------------------------------------
        # Read YOLO labels
        # ----------------------------------------------------

        boxes, labels = self.read_labels(
            label_path
        )

        # ----------------------------------------------------
        # Augmentation
        # ----------------------------------------------------

        if self.augment:

            image, boxes, labels = (
                self.apply_augmentation(
                    image,
                    boxes,
                    labels
                )
            )

        # ----------------------------------------------------
        # Resize
        # ----------------------------------------------------

        image, boxes = (
            self.resize_image_and_boxes(
                image,
                boxes
            )
        )

        # ----------------------------------------------------
        # Clip boxes
        # ----------------------------------------------------

        boxes = self.clip_boxes(
            boxes
        )

        # ----------------------------------------------------
        # Remove invalid boxes
        # ----------------------------------------------------

        boxes, labels = (
            self.remove_invalid_boxes(
                boxes,
                labels
            )
        )

        # ----------------------------------------------------
        # Optional external transform
        # ----------------------------------------------------

        if self.transform is not None:

            image, boxes, labels = (
                self.transform(
                    image,
                    boxes,
                    labels
                )
            )

        # ----------------------------------------------------
        # PIL -> NumPy -> Tensor
        # ----------------------------------------------------

        image = torch.from_numpy(
            np.array(image)
        )

        # HWC -> CHW
        image = image.permute(
            2,
            0,
            1
        )

        # uint8 -> float32
        image = image.float() / 255.0

        # ----------------------------------------------------
        # Target
        # ----------------------------------------------------

        target = {

            "boxes": boxes,

            "labels": labels,

            "image_id": image_path.stem,

            "image_path": str(
                image_path
            )
        }

        return image, target

    # ========================================================
    # ENABLE / DISABLE AUGMENTATION
    # ========================================================

    def set_augmentation(
        self,
        enabled=True,
        strength=None
    ):
        """
        Change augmentation after Dataset creation.

        Example:

            dataset.set_augmentation(True, 2)

        No Dataset recreation is required.
        """

        self.augment = enabled

        if strength is not None:

            if strength not in [1, 2, 3]:

                raise ValueError(
                    "strength must be 1, 2, or 3"
                )

            self.augmentation_strength = (
                strength
            )


# ============================================================
# COLLATE FUNCTION
# ============================================================

def detection_collate_fn(batch):

    """
    Custom collate function for object detection.

    Images are stacked because all images have the same size.

    Targets remain a list because every image can contain
    a different number of objects.
    """

    images = []
    targets = []

    for image, target in batch:

        images.append(image)
        targets.append(target)

    images = torch.stack(
        images,
        dim=0
    )

    return images, targets


# ============================================================
# DATALOADER BUILDER
# ============================================================

def create_dataloader(
    root_dir,
    split="train",
    img_size=640,
    batch_size=16,
    shuffle=True,
    num_workers=0,
    transform=None,
    augment=False,
    augmentation_strength=1
):

    dataset = FODDetectionDataset(
        root_dir=root_dir,
        split=split,
        img_size=img_size,
        transform=transform,
        augment=augment,
        augmentation_strength=augmentation_strength
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=detection_collate_fn
    )

    return dataset, dataloader