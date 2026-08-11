from pathlib import Path
import xml.etree.ElementTree as ET
from collections import Counter

import pandas as pd
import numpy as np
import random

import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.patches as patches

class FODEDA:

    def __init__(self, dataset_root, built_root):

        self.dataset_root = Path(dataset_root)
        self.built_root = Path(built_root)

        self.image_dir = (
            self.dataset_root / "JPEGImages"
        )

        self.annotation_dir = (
            self.dataset_root / "Annotations"
        )

        self.main_dir = (
            self.dataset_root /
            "ImageSets" /
            "Main"
        )

        self.metadata_file = (
            self.main_dir /
            "CategorizationData" /
            "FOD_categorization_annotations.csv"
        )

    # ========================================================
    # READ XML ANNOTATIONS
    # ========================================================

    def load_annotations(self):

        records = []

        xml_files = sorted(
            self.annotation_dir.glob("*.xml")
        )

        for xml_path in xml_files:

            try:

                tree = ET.parse(xml_path)
                root = tree.getroot()

                size = root.find("size")

                if size is None:
                    continue

                image_width = float(
                    size.findtext("width")
                )

                image_height = float(
                    size.findtext("height")
                )

                objects = root.findall("object")

                for obj in objects:

                    class_name = obj.findtext(
                        "name"
                    )

                    bbox = obj.find("bndbox")

                    if bbox is None:
                        continue

                    xmin = float(
                        bbox.findtext("xmin")
                    )

                    ymin = float(
                        bbox.findtext("ymin")
                    )

                    xmax = float(
                        bbox.findtext("xmax")
                    )

                    ymax = float(
                        bbox.findtext("ymax")
                    )

                    width = xmax - xmin
                    height = ymax - ymin

                    area = width * height

                    image_area = (
                        image_width *
                        image_height
                    )

                    relative_area = (
                        area / image_area
                    )

                    aspect_ratio = (
                        width / height
                        if height > 0
                        else np.nan
                    )

                    records.append({

                        "image_id":
                            xml_path.stem,

                        "class":
                            class_name,

                        "image_width":
                            image_width,

                        "image_height":
                            image_height,

                        "xmin":
                            xmin,

                        "ymin":
                            ymin,

                        "xmax":
                            xmax,

                        "ymax":
                            ymax,

                        "bbox_width":
                            width,

                        "bbox_height":
                            height,

                        "bbox_area":
                            area,

                        "relative_area":
                            relative_area,

                        "aspect_ratio":
                            aspect_ratio
                    })

            except Exception as e:

                print(
                    f"Error reading {xml_path.name}: {e}"
                )

        return pd.DataFrame(records)

    # ========================================================
    # OBJECTS PER IMAGE
    # ========================================================

    def objects_per_image(self, annotations_df):

        result = (
            annotations_df
            .groupby("image_id")
            .size()
            .reset_index(
                name="object_count"
            )
        )

        return result

    # ========================================================
    # CLASS DISTRIBUTION
    # ========================================================

    def class_distribution(
        self,
        annotations_df
    ):

        result = (
            annotations_df["class"]
            .value_counts()
            .reset_index()
        )

        result.columns = [
            "class",
            "object_count"
        ]

        return result

    # ========================================================
    # IMAGE RESOLUTION
    # ========================================================

    def resolution_distribution(
        self,
        annotations_df
    ):

        result = (
            annotations_df[
                [
                    "image_width",
                    "image_height"
                ]
            ]
            .drop_duplicates()
            .value_counts()
            .reset_index(
                name="image_count"
            )
        )

        return result

    # ========================================================
    # WEATHER
    # ========================================================

    def weather_distribution(self):

        if not self.metadata_file.exists():

            print(
                "Metadata file not found."
            )

            return None

        df = pd.read_csv(
            self.metadata_file
        )

        return (
            df["Weather"]
            .value_counts()
            .sort_index()
            .reset_index()
        )

    # ========================================================
    # LIGHTING
    # ========================================================

    def lighting_distribution(self):

        if not self.metadata_file.exists():

            print(
                "Metadata file not found."
            )

            return None

        df = pd.read_csv(
            self.metadata_file
        )

        return (
            df["Light"]
            .value_counts()
            .sort_index()
            .reset_index()
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    def summary(self, annotations_df):

        summary = {

            "total_objects":
                len(annotations_df),

            "unique_images":
                annotations_df[
                    "image_id"
                ].nunique(),

            "unique_classes":
                annotations_df[
                    "class"
                ].nunique(),

            "mean_objects_per_image":
                annotations_df
                .groupby("image_id")
                .size()
                .mean(),

            "median_bbox_area":
                annotations_df[
                    "bbox_area"
                ].median(),

            "median_relative_bbox_area":
                annotations_df[
                    "relative_area"
                ].median(),

            "median_aspect_ratio":
                annotations_df[
                    "aspect_ratio"
                ].median()
        }

        return summary
    
    # ========================================================
    # Randomly selecting 10 images and drawing the ground-truth bounding boxes
    # ========================================================
    def visualize_random_samples(
        self,
        num_samples=10,
        seed=42,
        figsize=(12, 25)
    ):
        """
        Randomly select images and display them in subplots
        with their ground-truth bounding boxes.
        """

        image_files = list(
            self.image_dir.glob("*.jpg")
        )

        if len(image_files) == 0:
            print("No images found.")
            return

        # Reproducible random selection
        random.seed(seed)

        num_samples = min(
            num_samples,
            len(image_files)
        )

        selected_images = random.sample(
            image_files,
            num_samples
        )

        # --------------------------------------------------------
        # Create subplot grid
        # --------------------------------------------------------

        cols = 2
        rows = (num_samples + cols - 1) // cols

        fig, axes = plt.subplots(
            rows,
            cols,
            figsize=figsize
        )

        # Convert axes to 1D array
        axes = np.asarray(axes).reshape(-1)

        # --------------------------------------------------------
        # Draw each image
        # --------------------------------------------------------

        for ax, image_path in zip(
            axes,
            selected_images
        ):

            # Load image
            image = Image.open(
                image_path
            ).convert("RGB")

            ax.imshow(image)

            # Corresponding annotation
            xml_path = (
                self.annotation_dir /
                f"{image_path.stem}.xml"
            )

            object_count = 0

            if xml_path.exists():

                try:

                    tree = ET.parse(xml_path)
                    root = tree.getroot()

                    for obj in root.findall("object"):

                        class_name = obj.findtext(
                            "name",
                            default="Unknown"
                        )

                        bbox = obj.find("bndbox")

                        if bbox is None:
                            continue

                        xmin = float(
                            bbox.findtext("xmin")
                        )

                        ymin = float(
                            bbox.findtext("ymin")
                        )

                        xmax = float(
                            bbox.findtext("xmax")
                        )

                        ymax = float(
                            bbox.findtext("ymax")
                        )

                        # ------------------------------------------------
                        # Draw rectangle
                        # ------------------------------------------------

                        rect = patches.Rectangle(
                            (xmin, ymin),
                            xmax - xmin,
                            ymax - ymin,
                            linewidth=2,
                            edgecolor="red",
                            facecolor="none"
                        )

                        ax.add_patch(rect)

                        # ------------------------------------------------
                        # Class label
                        # ------------------------------------------------

                        ax.text(
                            xmin,
                            max(ymin - 4, 5),
                            class_name,
                            fontsize=8,
                            color="white",
                            bbox=dict(
                                facecolor="red",
                                alpha=0.7,
                                pad=2
                            )
                        )

                        object_count += 1

                except Exception as e:

                    ax.set_title(
                        f"{image_path.stem}\nXML Error"
                    )

                    print(
                        f"Error reading {xml_path.name}: {e}"
                    )

            # --------------------------------------------------------
            # Image title
            # --------------------------------------------------------

            ax.set_title(
                f"{image_path.stem} | "
                f"{object_count} object(s)",
                fontsize=10
            )

            ax.axis("off")

        # --------------------------------------------------------
        # Hide unused subplot
        # --------------------------------------------------------

        for ax in axes[num_samples:]:
            ax.axis("off")

        # --------------------------------------------------------
        # Overall title
        # --------------------------------------------------------

        fig.suptitle(
            "Random FOD-A Samples with Ground-Truth Bounding Boxes",
            fontsize=16,
            y=0.995
        )

        plt.tight_layout(
            rect=[0, 0, 1, 0.98]
        )

        plt.show()