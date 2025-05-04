"""Implementation of YOLOv11 training pipeline."""

import os
import argparse
from ultralytics import YOLO  # type: ignore
from ultralytics import settings

settings.update({"datasets_dir": os.path.abspath(".")})


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="YOLO Training configuration")

    parser.add_argument(
        "--data",
        type=str,
        default="./dataset_ddpm_dino.yaml",
        help="Path to the dataset YAML file",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="./runs_ddpm_dino",
        help="Path to save training runs",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./yolo11s.pt",
        help="Path to pre-trained model (for example yolo11s.pt)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Base learning rate for the optimizer (for example 0.001)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1000,
        help="Number of training epochs",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    DATA = args.data
    PROJECT = args.project
    CHECKPOINT = args.checkpoint
    LR = args.lr
    EPOCHS = args.epochs
    AUGMENTATIONS = {
        "degrees": 10,
        "translate": 0.1,
        "scale": 0.5,
        "shear": 10,
        "fliplr": 0.5,
        "mosaic": 0.5,
        "erasing": 0.3,
        "hsv_h": 0.01,
        "hsv_s": 0.3,
        "hsv_v": 0.3,
    }

    model = YOLO(CHECKPOINT)
    train_results = model.train(
        data=DATA,
        epochs=EPOCHS,
        patience=EPOCHS,
        save_period=-1,
        batch=32,
        imgsz=640,
        device="cuda",
        optimizer="Adam",
        lr0=LR,
        freeze=0,
        project=PROJECT,
        name="ship_detection",
        exist_ok=True,
        augment=True,
        **AUGMENTATIONS,
    )
