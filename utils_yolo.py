"""
Implementation of utility functions for image generation 
and object detection with pretrained YOLO model.
"""

import os
import json
from datetime import datetime
from PIL import Image
import numpy as np


def generate_image(image_path, pipe, params):
    """
    Function to generate image variations using pre-trained StableDiffusion pipe.
    """
    image = Image.open(image_path).convert("RGB")
    image = image.resize((image.size[0] // 2, image.size[1] // 2))

    out = pipe(image=image, **params)

    return out["images"][0]


def boxes_with_high_confidence_yolo(
    image, model, confidence_threshold=0.6, filter_by_average=True
):
    """
    Function to check if the image contains boxes with high confidence
    using pretrained YOLO model.
    """

    results = model.predict(image, verbose=False, save=False, imgsz=640, conf=0.25)[0]
    boxes = results.boxes

    if boxes is None or boxes.shape[0] == 0:
        return False

    confidences = boxes.conf.cpu().numpy()

    if filter_by_average:
        if np.mean(confidences) < confidence_threshold:
            return False
    else:
        if np.all(confidences < confidence_threshold):
            return False

    return boxes


def save_image_and_boxes_yolo(image, boxes, image_name, output_dir):
    """
    Function to save the generated image and its corresponding boxes
    using pretrained YOLO model.
    """
    os.makedirs(output_dir, exist_ok=True)

    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")

    path_to_image = os.path.join(output_dir, f"{image_name}_{timestamp}.bmp")
    path_to_boxes = os.path.join(output_dir, f"{image_name}_{timestamp}.txt")

    image.save(path_to_image)

    with open(path_to_boxes, "w", encoding="utf-8") as f:
        for i in range(boxes.shape[0]):
            # Extracting box coordinates and class
            x, y, w, h = boxes.xywhn[i].cpu().numpy()
            cls = int(boxes.cls[i].cpu().numpy())

            f.write(f"{cls} {x} {y} {w} {h}\n")


def convert_to_yolo_format(annotation_dir, output_dir, class_mapping):
    """
    Converts annotations from a JSON format to the YOLO format.
    """
    os.makedirs(output_dir, exist_ok=True)

    for annotation_json in os.listdir(annotation_dir):

        with open(
            os.path.join(annotation_dir, annotation_json), "r", encoding="utf-8"
        ) as file_json:
            data = json.load(file_json)

        annotation_txt = (
            annotation_json.replace(".json", "").replace(".bmp", "") + ".txt"
        )
        labels = []

        for obj in data["objects"]:
            if obj["classId"] not in class_mapping:
                continue

            # Convert coordinates to YOLO format
            x_min, y_min = obj["points"]["exterior"][0]
            x_max, y_max = obj["points"]["exterior"][1]

            x_center = (x_min + x_max) / 2 / data["size"]["width"]
            y_center = (y_min + y_max) / 2 / data["size"]["height"]
            width = (x_max - x_min) / data["size"]["width"]
            height = (y_max - y_min) / data["size"]["height"]

            labels.append(
                f"{class_mapping[obj["classId"]]} {x_center} {y_center} {width} {height}\n"
            )

        with open(
            os.path.join(output_dir, annotation_txt),
            "w",
            encoding="utf-8",
        ) as file_txt:
            file_txt.writelines(labels)
