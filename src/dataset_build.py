from pathlib import Path
import xml.etree.ElementTree as ET
import json

from sklearn.model_selection import train_test_split


# ============================================================
# DATASET BUILDER
# ============================================================

class FODDatasetBuilder:

    def __init__(
        self,
        dataset_root,
        output_root,
        val_size=0.20,
        random_seed=42
    ):
        """
        Parameters
        ----------
        dataset_root : str or Path
            Path to VOC2007 folder.

        output_root : str or Path
            Where cleaned dataset information will be saved.

        val_size : float
            Fraction of cleaned trainval data used for validation.

        random_seed : int
            Seed for reproducible train/validation split.
        """

        self.dataset_root = Path(dataset_root)
        self.output_root = Path(output_root)

        self.val_size = val_size
        self.random_seed = random_seed

        # ----------------------------------------------------
        # Dataset directories
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Split files
        # ----------------------------------------------------

        self.trainval_file = (
            self.main_dir / "trainval.txt"
        )

        self.test_file = (
            self.main_dir / "test.txt"
        )

        # ----------------------------------------------------
        # Output directories
        # ----------------------------------------------------

        self.output_root.mkdir(
            parents=True,
            exist_ok=True
        )

        self.split_dir = (
            self.output_root / "splits"
        )

        self.report_dir = (
            self.output_root / "reports"
        )

        self.split_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.report_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    # ========================================================
    # VALIDATE DATASET STRUCTURE
    # ========================================================

    def validate_dataset_structure(self):

        required_paths = [
            self.dataset_root,
            self.image_dir,
            self.annotation_dir,
            self.main_dir,
            self.trainval_file,
            self.test_file
        ]

        missing = [
            str(path)
            for path in required_paths
            if not path.exists()
        ]

        if missing:

            raise FileNotFoundError(
                "Missing dataset components:\n"
                + "\n".join(missing)
            )

        print("Dataset structure: OK")

    # ========================================================
    # READ SPLIT FILE
    # ========================================================

    @staticmethod
    def read_split_file(file_path):

        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as f:

            ids = [
                line.strip()
                for line in f
                if line.strip()
            ]

        return ids

    # ========================================================
    # REMOVE DUPLICATES
    # ========================================================

    @staticmethod
    def remove_duplicates(ids):

        return list(dict.fromkeys(ids))

    # ========================================================
    # CHECK IMAGE EXISTS
    # ========================================================

    def image_exists(self, image_id):

        return (
            self.image_dir /
            f"{image_id}.jpg"
        ).exists()

    # ========================================================
    # CHECK XML EXISTS
    # ========================================================

    def annotation_exists(self, image_id):

        return (
            self.annotation_dir /
            f"{image_id}.xml"
        ).exists()

    # ========================================================
    # VALIDATE XML
    # ========================================================

    def validate_annotation(self, image_id):

        """
        Validate one Pascal VOC annotation.

        Checks:
        - XML can be parsed
        - image size exists
        - object class exists
        - bounding box exists
        - coordinates are valid
        - coordinates are inside image boundaries
        """

        xml_path = (
            self.annotation_dir /
            f"{image_id}.xml"
        )

        result = {
            "valid": True,
            "reason": None,
            "num_objects": 0
        }

        try:

            tree = ET.parse(xml_path)

            root = tree.getroot()

        except Exception as e:

            result["valid"] = False
            result["reason"] = (
                f"xml_parse_error: {str(e)}"
            )

            return result

        # ----------------------------------------------------
        # Image size
        # ----------------------------------------------------

        size = root.find("size")

        if size is None:

            result["valid"] = False
            result["reason"] = "missing_size"

            return result

        try:

            width = float(
                size.findtext("width")
            )

            height = float(
                size.findtext("height")
            )

        except Exception:

            result["valid"] = False
            result["reason"] = "invalid_image_size"

            return result

        if width <= 0 or height <= 0:

            result["valid"] = False
            result["reason"] = "non_positive_image_size"

            return result

        # ----------------------------------------------------
        # Objects
        # ----------------------------------------------------

        objects = root.findall("object")

        result["num_objects"] = len(objects)

        for object_index, obj in enumerate(objects):

            # ------------------------------------------------
            # Class
            # ------------------------------------------------

            class_name = obj.findtext("name")

            if not class_name or not class_name.strip():

                result["valid"] = False
                result["reason"] = (
                    f"missing_class_object_{object_index}"
                )

                return result

            # ------------------------------------------------
            # Bounding box
            # ------------------------------------------------

            bbox = obj.find("bndbox")

            if bbox is None:

                result["valid"] = False
                result["reason"] = (
                    f"missing_bbox_object_{object_index}"
                )

                return result

            try:

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

            except Exception:

                result["valid"] = False
                result["reason"] = (
                    f"invalid_bbox_values_object_{object_index}"
                )

                return result

            # ------------------------------------------------
            # Coordinate ordering
            # ------------------------------------------------

            if xmax <= xmin:

                result["valid"] = False
                result["reason"] = (
                    f"invalid_x_coordinates_object_{object_index}"
                )

                return result

            if ymax <= ymin:

                result["valid"] = False
                result["reason"] = (
                    f"invalid_y_coordinates_object_{object_index}"
                )

                return result

            # ------------------------------------------------
            # Boundary check
            # ------------------------------------------------

            if xmin < 0 or ymin < 0:

                result["valid"] = False
                result["reason"] = (
                    f"negative_bbox_coordinate_object_{object_index}"
                )

                return result

            if xmax > width or ymax > height:

                result["valid"] = False
                result["reason"] = (
                    f"bbox_outside_image_object_{object_index}"
                )

                return result

        return result

    # ========================================================
    # VALIDATE IMAGE ID
    # ========================================================

    def validate_image_id(self, image_id):

        """
        Check whether an image ID has:
        - image
        - XML
        - valid XML annotations
        """

        result = {
            "image_id": image_id,
            "valid": True,
            "reason": None
        }

        # ----------------------------------------------------
        # Image
        # ----------------------------------------------------

        if not self.image_exists(image_id):

            result["valid"] = False
            result["reason"] = "missing_image"

            return result

        # ----------------------------------------------------
        # Annotation
        # ----------------------------------------------------

        if not self.annotation_exists(image_id):

            result["valid"] = False
            result["reason"] = "missing_annotation"

            return result

        # ----------------------------------------------------
        # XML validation
        # ----------------------------------------------------

        annotation_result = (
            self.validate_annotation(image_id)
        )

        if not annotation_result["valid"]:

            result["valid"] = False
            result["reason"] = (
                annotation_result["reason"]
            )

            return result

        return result

    # ========================================================
    # VALIDATE SPLIT
    # ========================================================

    def validate_split(self, ids, split_name):

        valid_ids = []
        invalid_records = []

        print(
            f"\nValidating {split_name} split..."
        )

        for image_id in ids:

            result = self.validate_image_id(
                image_id
            )

            if result["valid"]:

                valid_ids.append(image_id)

            else:

                invalid_records.append(
                    result
                )

        print(
            f"{split_name}: "
            f"{len(valid_ids)} valid, "
            f"{len(invalid_records)} invalid"
        )

        return valid_ids, invalid_records

    # ========================================================
    # SAVE IDS
    # ========================================================

    def save_ids(self, ids, filename):

        output_file = (
            self.split_dir / filename
        )

        with open(
            output_file,
            "w",
            encoding="utf-8"
        ) as f:

            for image_id in ids:

                f.write(
                    f"{image_id}\n"
                )

        return output_file

    # ========================================================
    # SAVE JSON
    # ========================================================

    def save_json(self, data, filename):

        output_file = (
            self.report_dir / filename
        )

        with open(
            output_file,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                data,
                f,
                indent=4
            )

        return output_file

    # ========================================================
    # BUILD DATASET
    # ========================================================

    def build(self):

        print("=" * 70)
        print("FOD-A DATASET BUILD")
        print("=" * 70)

        # ----------------------------------------------------
        # 1. Validate structure
        # ----------------------------------------------------

        self.validate_dataset_structure()

        # ----------------------------------------------------
        # 2. Read official split files
        # ----------------------------------------------------

        original_trainval = (
            self.read_split_file(
                self.trainval_file
            )
        )

        original_test = (
            self.read_split_file(
                self.test_file
            )
        )

        # ----------------------------------------------------
        # 3. Remove duplicates
        # ----------------------------------------------------

        trainval = self.remove_duplicates(
            original_trainval
        )

        test = self.remove_duplicates(
            original_test
        )

        print("\nOriginal split information:")
        print(
            "Trainval entries:",
            len(original_trainval)
        )
        print(
            "Trainval unique:",
            len(trainval)
        )
        print(
            "Test entries:",
            len(original_test)
        )
        print(
            "Test unique:",
            len(test)
        )

        # ----------------------------------------------------
        # 4. Detect train/test overlap
        # ----------------------------------------------------

        overlap = sorted(
            set(trainval).intersection(
                set(test)
            )
        )

        print(
            "\nTrain/Test overlap:",
            len(overlap)
        )

        if overlap:

            print(
                "Overlapping IDs:",
                overlap
            )

        # ----------------------------------------------------
        # 5. Remove test IDs from trainval
        # ----------------------------------------------------

        clean_trainval = [
            image_id
            for image_id in trainval
            if image_id not in set(test)
        ]

        print(
            "Trainval after leakage removal:",
            len(clean_trainval)
        )

        # ----------------------------------------------------
        # 6. Validate TEST
        # ----------------------------------------------------

        valid_test, invalid_test = (
            self.validate_split(
                test,
                "TEST"
            )
        )

        # ----------------------------------------------------
        # 7. Validate TRAIN/VAL pool
        # ----------------------------------------------------

        valid_trainval, invalid_trainval = (
            self.validate_split(
                clean_trainval,
                "TRAINVAL"
            )
        )

        # ----------------------------------------------------
        # 8. Create train / validation split
        # ----------------------------------------------------

        train_ids, val_ids = train_test_split(
            valid_trainval,
            test_size=self.val_size,
            random_state=self.random_seed,
            shuffle=True
        )

        train_ids = sorted(train_ids)
        val_ids = sorted(val_ids)
        valid_test = sorted(valid_test)

        print("\nFINAL SPLIT")
        print("-" * 70)

        print(
            f"Train      : {len(train_ids)}"
        )

        print(
            f"Validation : {len(val_ids)}"
        )

        print(
            f"Test       : {len(valid_test)}"
        )

        # ----------------------------------------------------
        # 9. Verify no overlap
        # ----------------------------------------------------

        train_set = set(train_ids)
        val_set = set(val_ids)
        test_set = set(valid_test)

        train_val_overlap = (
            train_set.intersection(val_set)
        )

        train_test_overlap = (
            train_set.intersection(test_set)
        )

        val_test_overlap = (
            val_set.intersection(test_set)
        )

        print("\nFINAL OVERLAP CHECK")
        print("-" * 70)

        print(
            "Train / Val:",
            len(train_val_overlap)
        )

        print(
            "Train / Test:",
            len(train_test_overlap)
        )

        print(
            "Val / Test:",
            len(val_test_overlap)
        )

        # ----------------------------------------------------
        # 10. Save split files
        # ----------------------------------------------------

        self.save_ids(
            train_ids,
            "train.txt"
        )

        self.save_ids(
            val_ids,
            "val.txt"
        )

        self.save_ids(
            valid_test,
            "test.txt"
        )

        # ----------------------------------------------------
        # 11. Create report
        # ----------------------------------------------------

        report = {

            "dataset_root": str(
                self.dataset_root
            ),

            "random_seed": self.random_seed,

            "validation_size": self.val_size,

            "original": {

                "trainval_entries":
                    len(original_trainval),

                "trainval_unique":
                    len(trainval),

                "test_entries":
                    len(original_test),

                "test_unique":
                    len(test)
            },

            "duplicates": {

                "trainval_removed":
                    len(original_trainval)
                    - len(trainval),

                "test_removed":
                    len(original_test)
                    - len(test)
            },

            "train_test_overlap": {

                "count":
                    len(overlap),

                "ids":
                    overlap
            },

            "validation": {

                "invalid_trainval":
                    len(invalid_trainval),

                "invalid_test":
                    len(invalid_test)
            },

            "final_split": {

                "train":
                    len(train_ids),

                "validation":
                    len(val_ids),

                "test":
                    len(valid_test)
            },

            "final_overlap": {

                "train_val":
                    len(train_val_overlap),

                "train_test":
                    len(train_test_overlap),

                "val_test":
                    len(val_test_overlap)
            }
        }

        self.save_json(
            report,
            "dataset_build_report.json"
        )

        # ----------------------------------------------------
        # 12. Save invalid records
        # ----------------------------------------------------

        invalid_records = (
            invalid_trainval +
            invalid_test
        )

        self.save_json(
            invalid_records,
            "invalid_samples.json"
        )

        print("\nFiles created:")
        print(
            self.split_dir / "train.txt"
        )
        print(
            self.split_dir / "val.txt"
        )
        print(
            self.split_dir / "test.txt"
        )
        print(
            self.report_dir /
            "dataset_build_report.json"
        )
        print(
            self.report_dir /
            "invalid_samples.json"
        )

        print("\nDataset build completed.")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    PROJECT_ROOT = (
        Path(__file__).resolve().parent.parent
    )

    DATASET_ROOT = (
        PROJECT_ROOT /
        "dataset" /
        "VOC2007"
    )

    OUTPUT_ROOT = (
        PROJECT_ROOT /
        "dataset" /
        "built"
    )

    builder = FODDatasetBuilder(
        dataset_root=DATASET_ROOT,
        output_root=OUTPUT_ROOT,
        val_size=0.20,
        random_seed=42
    )

    builder.build()