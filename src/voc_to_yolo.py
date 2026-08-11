from pathlib import Path
import shutil
import xml.etree.ElementTree as ET


class VOCToYOLO:

    def __init__(
        self,
        dataset_root,
        built_root,
        output_root,
        class_names
    ):

        self.dataset_root = Path(dataset_root)
        self.built_root = Path(built_root)
        self.output_root = Path(output_root)

        self.class_names = class_names

        self.class_to_id = {
            name: idx
            for idx, name in enumerate(class_names)
        }

        # ----------------------------------------------------
        # VOC paths
        # ----------------------------------------------------

        self.image_dir = (
            self.dataset_root / "JPEGImages"
        )

        self.annotation_dir = (
            self.dataset_root / "Annotations"
        )

        # ----------------------------------------------------
        # Split paths
        # ----------------------------------------------------

        self.split_dir = (
            self.built_root / "splits"
        )

        # ----------------------------------------------------
        # YOLO paths
        # ----------------------------------------------------

        self.image_output_dirs = {}

        self.label_output_dirs = {}

        for split in ["train", "val", "test"]:

            self.image_output_dirs[split] = (
                self.output_root /
                "images" /
                split
            )

            self.label_output_dirs[split] = (
                self.output_root /
                "labels" /
                split
            )

            self.image_output_dirs[split].mkdir(
                parents=True,
                exist_ok=True
            )

            self.label_output_dirs[split].mkdir(
                parents=True,
                exist_ok=True
            )

    # ========================================================
    # READ SPLIT
    # ========================================================

    def read_split(self, split):

        split_file = (
            self.split_dir /
            f"{split}.txt"
        )

        if not split_file.exists():

            raise FileNotFoundError(
                f"Split file not found: {split_file}"
            )

        return [
            line.strip()
            for line in split_file.read_text().splitlines()
            if line.strip()
        ]

    # ========================================================
    # CONVERT VOC BOX TO YOLO BOX
    # ========================================================

    @staticmethod
    def voc_to_yolo(
        xmin,
        ymin,
        xmax,
        ymax,
        image_width,
        image_height
    ):

        box_width = xmax - xmin
        box_height = ymax - ymin

        x_center = (
            xmin + xmax
        ) / 2.0

        y_center = (
            ymin + ymax
        ) / 2.0

        # Normalize
        x_center /= image_width
        y_center /= image_height

        box_width /= image_width
        box_height /= image_height

        return (
            x_center,
            y_center,
            box_width,
            box_height
        )

    # ========================================================
    # CONVERT ONE XML
    # ========================================================

    def convert_annotation(
        self,
        image_id,
        output_label_path
    ):

        xml_path = (
            self.annotation_dir /
            f"{image_id}.xml"
        )

        tree = ET.parse(xml_path)
        root = tree.getroot()

        size = root.find("size")

        image_width = float(
            size.findtext("width")
        )

        image_height = float(
            size.findtext("height")
        )

        yolo_lines = []

        for obj in root.findall("object"):

            class_name = obj.findtext("name")

            if class_name not in self.class_to_id:

                raise ValueError(
                    f"Unknown class '{class_name}' "
                    f"in {xml_path.name}"
                )

            class_id = self.class_to_id[
                class_name
            ]

            bbox = obj.find("bndbox")

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

            (
                x_center,
                y_center,
                width,
                height
            ) = self.voc_to_yolo(
                xmin,
                ymin,
                xmax,
                ymax,
                image_width,
                image_height
            )

            yolo_lines.append(
                f"{class_id} "
                f"{x_center:.6f} "
                f"{y_center:.6f} "
                f"{width:.6f} "
                f"{height:.6f}"
            )

        output_label_path.write_text(
            "\n".join(yolo_lines),
            encoding="utf-8"
        )

    # ========================================================
    # CONVERT SPLIT
    # ========================================================

    def convert_split(self, split):

        ids = self.read_split(split)

        print(
            f"\nConverting {split}: "
            f"{len(ids)} images"
        )

        converted = 0

        for image_id in ids:

            # ------------------------------------------------
            # Source image
            # ------------------------------------------------

            source_image = (
                self.image_dir /
                f"{image_id}.jpg"
            )

            # ------------------------------------------------
            # Destination image
            # ------------------------------------------------

            destination_image = (
                self.image_output_dirs[split] /
                f"{image_id}.jpg"
            )

            # ------------------------------------------------
            # Copy image
            # ------------------------------------------------

            shutil.copy2(
                source_image,
                destination_image
            )

            # ------------------------------------------------
            # Create YOLO label
            # ------------------------------------------------

            label_path = (
                self.label_output_dirs[split] /
                f"{image_id}.txt"
            )

            self.convert_annotation(
                image_id,
                label_path
            )

            converted += 1

        print(
            f"{split}: {converted} converted"
        )

    # ========================================================
    # CREATE DATA.YAML
    # ========================================================

    def create_data_yaml(self):

        yaml_path = (
            self.output_root /
            "data.yaml"
        )

        names_string = "\n".join(
            [
                f"  {idx}: {name}"
                for idx, name
                in enumerate(self.class_names)
            ]
        )

        content = f"""path: {self.output_root.resolve().as_posix()}

train: images/train
val: images/val
test: images/test

nc: {len(self.class_names)}

names:
{names_string}
"""

        yaml_path.write_text(
            content,
            encoding="utf-8"
        )

        print(
            f"\nCreated: {yaml_path}"
        )

    # ========================================================
    # BUILD YOLO DATASET
    # ========================================================

    def build(self):

        print("=" * 70)
        print("VOC → YOLO DATASET CONVERSION")
        print("=" * 70)

        print("\nClasses:")

        for idx, name in enumerate(
            self.class_names
        ):
            print(
                f"  {idx}: {name}"
            )

        # ----------------------------------------------------
        # Convert all splits
        # ----------------------------------------------------

        for split in [
            "train",
            "val",
            "test"
        ]:

            self.convert_split(split)

        # ----------------------------------------------------
        # Create YAML
        # ----------------------------------------------------

        self.create_data_yaml()

        print("\n" + "=" * 70)
        print("YOLO DATASET BUILD COMPLETE")
        print("=" * 70)