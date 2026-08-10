from pathlib import Path
import xml.etree.ElementTree as ET
from collections import Counter


def get_dataset_paths(dataset_root):
    """
    Return important FOD-A dataset paths.
    """

    dataset_root = Path(dataset_root)

    image_dir = dataset_root / "JPEGImages"
    annotation_dir = dataset_root / "Annotations"
    main_dir = dataset_root / "ImageSets" / "Main"

    return image_dir, annotation_dir, main_dir


def get_image_ids(image_dir):
    """
    Get image IDs from JPEGImages.
    """

    return {
        image_path.stem
        for image_path in Path(image_dir).glob("*.jpg")
    }


def get_annotation_ids(annotation_dir):
    """
    Get annotation IDs from Annotations.
    """

    return {
        xml_path.stem
        for xml_path in Path(annotation_dir).glob("*.xml")
    }


def get_basic_statistics(image_dir, annotation_dir):
    """
    Calculate basic dataset statistics.
    """

    image_ids = get_image_ids(image_dir)
    annotation_ids = get_annotation_ids(annotation_dir)

    annotated_ids = image_ids.intersection(annotation_ids)

    images_without_xml = image_ids - annotation_ids
    xml_without_image = annotation_ids - image_ids

    return {
        "total_images": len(image_ids),
        "total_xml": len(annotation_ids),
        "annotated_images": len(annotated_ids),
        "images_without_xml": len(images_without_xml),
        "xml_without_image": len(xml_without_image),
        "image_ids": image_ids,
        "annotation_ids": annotation_ids,
    }


def get_resolution_statistics(annotation_dir):
    """
    Calculate image resolution distribution from XML files.
    """

    resolution_counter = Counter()
    invalid_xml = []

    annotation_dir = Path(annotation_dir)

    for xml_path in annotation_dir.glob("*.xml"):

        try:

            tree = ET.parse(xml_path)
            root = tree.getroot()

            size = root.find("size")

            if size is None:
                invalid_xml.append(xml_path.name)
                continue

            width = int(float(size.findtext("width")))
            height = int(float(size.findtext("height")))

            resolution_counter[(width, height)] += 1

        except Exception:

            invalid_xml.append(xml_path.name)

    return resolution_counter, invalid_xml


def count_total_objects(annotation_dir):
    """
    Count total annotated objects.
    """

    total_objects = 0

    for xml_path in Path(annotation_dir).glob("*.xml"):

        try:

            tree = ET.parse(xml_path)
            root = tree.getroot()

            total_objects += len(
                root.findall("object")
            )

        except Exception:
            continue

    return total_objects


def get_class_distribution(annotation_dir):
    """
    Count number of objects for each class.
    """

    class_counter = Counter()

    for xml_path in Path(annotation_dir).glob("*.xml"):

        try:

            tree = ET.parse(xml_path)
            root = tree.getroot()

            for obj in root.findall("object"):

                class_name = obj.findtext("name")

                if class_name:
                    class_counter[class_name] += 1

        except Exception:
            continue

    return class_counter


def read_split_file(file_path):
    """
    Read trainval.txt or test.txt.
    """

    return [
        line.strip()
        for line in Path(file_path).read_text().splitlines()
        if line.strip()
    ]


def analyze_splits(main_dir):
    """
    Analyze trainval/test split files.
    """

    main_dir = Path(main_dir)

    trainval_file = main_dir / "trainval.txt"
    test_file = main_dir / "test.txt"

    trainval_ids = read_split_file(trainval_file)
    test_ids = read_split_file(test_file)

    trainval_set = set(trainval_ids)
    test_set = set(test_ids)

    overlap = trainval_set.intersection(test_set)

    trainval_duplicates = [
        x
        for x, count in Counter(trainval_ids).items()
        if count > 1
    ]

    test_duplicates = [
        x
        for x, count in Counter(test_ids).items()
        if count > 1
    ]

    return {
        "trainval_count": len(trainval_ids),
        "trainval_unique": len(trainval_set),

        "test_count": len(test_ids),
        "test_unique": len(test_set),

        "trainval_duplicates": trainval_duplicates,
        "test_duplicates": test_duplicates,

        "overlap": sorted(overlap),
    }