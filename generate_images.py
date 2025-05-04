"""
Implementation of pretrained StableDiffusion pipelines for generating new images.
"""

import os
import random
import argparse

import torch

from diffusers import StableDiffusionImageVariationPipeline
from diffusers import StableDiffusionImg2ImgPipeline
from diffusers import StableDiffusionInstructPix2PixPipeline
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel

from ultralytics import YOLO  # type: ignore

from utils_yolo import generate_image
from utils_yolo import boxes_with_high_confidence_yolo
from utils_yolo import save_image_and_boxes_yolo

from utils_dino import boxes_with_high_confidence_dino
from utils_dino import save_image_and_boxes_dino


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Setting up image generation parameters"
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=800,
        help="Number of images to generate",
    )
    parser.add_argument(
        "--model_generator",
        type=str,
        choices=["ImageVariation", "Img2Img", "Pix2Pix", "DDPM"],
        default="Img2Img",
        help="Model generator to use for image generation",
    )
    parser.add_argument(
        "--checkpoint_ddpm",
        type=str,
        default="./finetuned_ddpm_ship_256",
        help="Path to the checkpoint for DDPM model",
    )
    parser.add_argument(
        "--initial_images",
        type=str,
        default="./dataset/train/images",
        help="Path to the initial images",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        default="./prompts.txt",
        help="Path to the prompts file",
    )
    parser.add_argument(
        "--model_detector",
        type=str,
        default="./runs/baseline/ship_detection/weights/best.pt",
        help="Path to the YOLO detection model or model id (for example DINO)",
    )
    parser.add_argument(
        "--dino_text_labels",
        type=str,
        default="a ship",
        help="Text labels for DINO detection model for example 'a ship' or 'a ship, a boat'",
    )
    parser.add_argument(
        "--dino_class_mapping",
        type=str,
        default="a ship: 0, ship: 0",
        help="Class mapping for DINO detection model for example 'a ship: 0, ship: 0'",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    NUM_IMAGES = args.num_images
    MODEL_GENERATOR = args.model_generator
    PATH_TO_INITIAL_IMAGES = args.initial_images
    PROMPTS = args.prompts
    DETECTION_MODEL_PATH = args.model_detector
    CHECKPOINT_DDPM = args.checkpoint_ddpm

    DINO_TEXT_LABELS = args.dino_text_labels
    DINO_TEXT_LABELS = DINO_TEXT_LABELS.split(", ")

    DINO_CLASS_MAPPING = args.dino_class_mapping  # for example "a ship: 0, ship: 0"
    DINO_CLASS_MAPPING = DINO_CLASS_MAPPING.split(", ")
    DINO_CLASS_MAPPING = {
        item.split(": ")[0]: int(item.split(": ")[1]) for item in DINO_CLASS_MAPPING
    }

    SAVE_GENERATED_IMAGES = f"./{MODEL_GENERATOR}_generated"

    if DETECTION_MODEL_PATH == "DINO":
        DETECTION_MODEL = "DINO"
        SAVE_GENERATED_IMAGES += "_dino"
    else:
        DETECTION_MODEL = YOLO(DETECTION_MODEL_PATH)
        SAVE_GENERATED_IMAGES += "_yolo"

    if MODEL_GENERATOR == "ImageVariation":
        STABLE_DIFFUSION_PIPELINE = (
            StableDiffusionImageVariationPipeline.from_pretrained(
                "lambdalabs/sd-image-variations-diffusers", revision="v2.0"
            ).to("cuda")
        )
        STABLE_DIFFUSION_PARAMS = {"guidance_scale": 15.0}
    elif MODEL_GENERATOR == "Img2Img":
        STABLE_DIFFUSION_PIPELINE = StableDiffusionImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
        ).to("cuda")
        STABLE_DIFFUSION_PARAMS = {"strength": 0.75, "guidance_scale": 15.0}
    elif MODEL_GENERATOR == "Pix2Pix":
        STABLE_DIFFUSION_PIPELINE = (
            StableDiffusionInstructPix2PixPipeline.from_pretrained(
                "timbrooks/instruct-pix2pix", torch_dtype=torch.float16
            ).to("cuda")
        )
        STABLE_DIFFUSION_PARAMS = {}
    elif MODEL_GENERATOR == "DDPM":
        model = UNet2DModel.from_pretrained(CHECKPOINT_DDPM)
        noise_scheduler = DDPMScheduler.from_pretrained(CHECKPOINT_DDPM)
        STABLE_DIFFUSION_PIPELINE = DDPMPipeline(  # type: ignore
            unet=model, scheduler=noise_scheduler
        ).to("cuda")
        STABLE_DIFFUSION_PARAMS = {"num_inference_steps": 50}
    else:
        raise ValueError(
            "Invalid model generator! Choose from ['ImageVariation', 'Img2Img', 'Pix2Pix', 'DDPM']"
        )

    with open(PROMPTS, "r", encoding="utf-8") as file:
        prompts = file.readlines()

    CNT = 0
    while CNT < NUM_IMAGES:
        for initial_image in os.listdir(PATH_TO_INITIAL_IMAGES):
            initial_image_name = initial_image.split(".")[0]
            initial_image_path = os.path.join(PATH_TO_INITIAL_IMAGES, initial_image)

            if MODEL_GENERATOR in ["Img2Img", "Pix2Pix"]:
                prompt = random.choice(prompts).strip()
                STABLE_DIFFUSION_PARAMS["prompt"] = prompt  # type: ignore

            if MODEL_GENERATOR in ["ImageVariation", "Img2Img", "Pix2Pix"]:
                new_image = generate_image(
                    initial_image_path,
                    STABLE_DIFFUSION_PIPELINE,
                    STABLE_DIFFUSION_PARAMS,
                )
            else:
                new_image = STABLE_DIFFUSION_PIPELINE(**STABLE_DIFFUSION_PARAMS).images[
                    0
                ]
                initial_image_name = "noise"  # pylint: disable=invalid-name

            if DETECTION_MODEL == "DINO":
                boxes = boxes_with_high_confidence_dino(
                    new_image,
                    confidence_threshold=0.5,
                    filter_by_average=True,
                    text_labels=DINO_TEXT_LABELS,
                    class_mapping=DINO_CLASS_MAPPING,
                    box_threshold=0.25,
                    text_threshold=0.25,
                )
                if boxes:
                    save_image_and_boxes_dino(
                        new_image, boxes, initial_image_name, SAVE_GENERATED_IMAGES
                    )
                    CNT += 1
                    print(
                        f"{CNT} out of {NUM_IMAGES}",
                        f"images generated by {MODEL_GENERATOR}",
                        "labels generated by DINO",
                    )
            else:
                boxes = boxes_with_high_confidence_yolo(
                    new_image,
                    DETECTION_MODEL,
                    confidence_threshold=0.5,
                    filter_by_average=True,
                )
                if boxes:
                    save_image_and_boxes_yolo(
                        new_image, boxes, initial_image_name, SAVE_GENERATED_IMAGES
                    )
                    CNT += 1
                    print(
                        f"{CNT} out of {NUM_IMAGES}",
                        f"images generated by {MODEL_GENERATOR}",
                        "labels generated by YOLO",
                    )

            if CNT == NUM_IMAGES:
                break

    print("Done!")
