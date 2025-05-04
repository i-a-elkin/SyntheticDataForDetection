"""
Implementation of utility functions for object detection with pretrained Grounding DINO model.
"""

import os
from datetime import datetime
import numpy as np
import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection  # type: ignore


def boxes_with_high_confidence_dino(
    image,
    confidence_threshold=0.3,
    filter_by_average=True,
    text_labels=None,  # ["a ship"] or ["a ship", "a boat"]
    class_mapping=None,  # {"a ship": 0}
    box_threshold=0.25,
    text_threshold=0.25,
):
    """
    Function to check if the image contains boxes with high confidence
    using pretrained Grounding DINO model.
    """

    def convert_coordinates_to_yolo_format(box, image_size):
        """
        Converts coordinates from the format [x_min, y_min, x_max, y_max] to YOLO format.
        YOLO format: [x_center, y_center, width, height].
        """
        image_width, image_height = image_size
        x_min, y_min, x_max, y_max = box

        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(x_max, image_width)
        y_max = min(y_max, image_height)

        x_center = (x_min + x_max) / 2 / image_width
        y_center = (y_min + y_max) / 2 / image_height
        width = (x_max - x_min) / image_width
        height = (y_max - y_min) / image_height

        return [x_center, y_center, width, height]

    processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        "IDEA-Research/grounding-dino-base"
    ).to("cuda")

    inputs = processor(images=image, text=[text_labels], return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[image.size[::-1]],
    )[0]

    scores = results["scores"].tolist()
    if scores is None or len(scores) == 0:
        return False

    scores = np.array(scores)
    if filter_by_average:
        if np.mean(scores) < confidence_threshold:
            return False
    else:
        if np.all(scores < confidence_threshold):
            return False

    labels = results["labels"]

    boxes = results["boxes"].tolist()
    boxes = [convert_coordinates_to_yolo_format(box, image.size) for box in boxes]

    boxes = [
        {"label": class_mapping[label], "box": box} for label, box in zip(labels, boxes)
    ]

    return boxes


def save_image_and_boxes_dino(image, boxes, image_name, output_dir):
    """
    Function to save the generated image and its corresponding boxes
    using pretrained Grounding DINO model.
    """
    os.makedirs(output_dir, exist_ok=True)

    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")

    path_to_image = os.path.join(output_dir, f"{image_name}_{timestamp}.bmp")
    path_to_boxes = os.path.join(output_dir, f"{image_name}_{timestamp}.txt")

    image.save(path_to_image)

    with open(path_to_boxes, "w", encoding="utf-8") as f:
        for box in boxes:
            # Extracting box coordinates and class
            x, y, w, h = box["box"]
            cls = box["label"]

            f.write(f"{cls} {x} {y} {w} {h}\n")
