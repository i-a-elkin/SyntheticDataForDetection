"""
Implementation of the converter to convert all JSON annotations in specified directories
to YOLO format and save the resulting label files in corresponding output directories.
"""

from utils_yolo import convert_to_yolo_format

if __name__ == "__main__":
    ANNOTATION_DIRS = [
        "./dataset/train/ann/",
        "./dataset/val/ann/",
        "./dataset/test/ann/",
    ]
    OUTPUT_DIRS = [
        "./dataset/train/labels/",
        "./dataset/val/labels/",
        "./dataset/test/labels/",
    ]
    CLASS_MAPPING = {6476598: 0}

    for annotation_dir, output_dir in zip(ANNOTATION_DIRS, OUTPUT_DIRS):
        convert_to_yolo_format(annotation_dir, output_dir, CLASS_MAPPING)
    print("Done!")
