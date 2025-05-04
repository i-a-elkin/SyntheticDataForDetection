"""Train a diffusion model on a custom dataset."""

import argparse
import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T  # type: ignore
from PIL import Image
from tqdm.auto import tqdm

from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel
from accelerate import Accelerator  # type: ignore
from transformers import get_cosine_schedule_with_warmup  # type: ignore


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="DDPM training configuration")

    parser.add_argument(
        "--data",
        type=str,
        default="./dataset/train/images",
        help="Path to the dataset folder containing images",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="./runs_ddpm_ship_256",
        help="Path to save training runs",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./finetuned_ddpm_ship_256",
        help="Path to pre-trained model checkpoint",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Base learning rate for the optimizer",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for training",
    )
    parser.add_argument(
        "--save_period",
        type=int,
        default=5,
        help="Period to generate and save images during training",
    )
    parser.add_argument(
        "--save_num_images",
        type=int,
        default=5,
        help="Number of images to save during training",
    )

    return parser.parse_args()


class ShipDataset(Dataset):
    """
    Class for loading images from a specified folder and applying augmentation.
    """

    def __init__(self, image_folder, image_size=256, extensions=(".jpg", ".bmp")):
        self.image_folder = image_folder
        self.extensions = extensions

        self.image_paths = []
        for fname in os.listdir(image_folder):
            if fname.lower().endswith(self.extensions):
                full_path = os.path.join(image_folder, fname)
                if os.path.isfile(full_path):
                    self.image_paths.append(full_path)

        if not self.image_paths:
            print(
                f"Warning: no images found in {image_folder} with extensions {self.extensions}"
            )

        self.transform = T.Compose(
            [
                T.Resize((image_size, image_size)),
                T.RandomRotation(degrees=12),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.2),
                T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        try:
            image = Image.open(image_path).convert("RGB")
        except (FileExistsError, FileNotFoundError) as e:
            print(f"Error reading file {image_path}: {e}")

        return self.transform(image)


def train_one_epoch(
    model, dataloader, noise_scheduler, optimizer, lr_scheduler, accelerator, epoch_idx
):
    """
    Train the model for one epoch.
    """
    model.train()
    total_loss = 0.0

    progress_bar = tqdm(dataloader, desc=f"Epoch [{epoch_idx+1}]", leave=False)

    for step, batch in enumerate(progress_bar):
        clean_images = batch
        # Generate random noise and timesteps
        noise = torch.randn_like(clean_images)
        timesteps = torch.randint(
            0,
            noise_scheduler.num_train_timesteps,
            (clean_images.shape[0],),
            device=clean_images.device,
        ).long()
        # Create noisy images using the noise scheduler
        noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

        # Predict the noise using the model
        noise_pred = model(noisy_images, timesteps).sample
        loss = nn.functional.mse_loss(noise_pred, noise)

        optimizer.zero_grad()
        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()

        if step % 50 == 0:
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / len(dataloader)
    accelerator.print(f"Epoch {epoch_idx + 1} is completed. Avg Loss: {avg_loss:.4f}")


def train(
    data,
    project,
    checkpoint,
    lr=1e-4,
    epochs=200,
    batch_size=4,
    save_period=5,
    save_num_images=5,
):
    """
    Main function to train the model.
    """
    dataset = ShipDataset(data, image_size=256)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = UNet2DModel.from_pretrained(checkpoint)
    noise_scheduler = DDPMScheduler.from_pretrained(checkpoint)

    num_training_steps = epochs * len(dataloader)
    num_warmup_steps = int(0.1 * num_training_steps)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    accelerator = Accelerator()
    model, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, dataloader, lr_scheduler
    )

    os.makedirs(os.path.join(project, "samples"), exist_ok=True)
    for epoch in range(epochs):
        train_one_epoch(
            model,
            dataloader,
            noise_scheduler,
            optimizer,
            lr_scheduler,
            accelerator,
            epoch,
        )

        if (epoch + 1) % save_period == 0:
            accelerator.print("Generating samples...")
            unwrapped_model = accelerator.unwrap_model(model)
            pipe = DDPMPipeline(unet=unwrapped_model, scheduler=noise_scheduler)
            for i in range(save_num_images):
                sample = pipe(num_inference_steps=50).images[0]  # type: ignore
                sample.save(
                    os.path.join(project, "samples", f"epoch{epoch + 1}_n{i + 1}.jpg")
                )

    os.makedirs(os.path.join(project, "last"), exist_ok=True)
    accelerator.print("Saving model...")
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(os.path.join(project, "last"))
    noise_scheduler.save_config(os.path.join(project, "last"))


if __name__ == "__main__":
    args = get_args()

    DATA = args.data
    PROJECT = args.project
    CHECKPOINT = args.checkpoint
    LR = args.lr
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    SAVE_PERIOD = args.save_period
    SAVE_NUM_IMAGES = args.save_num_images

    train(
        data=DATA,
        project=PROJECT,
        checkpoint=CHECKPOINT,
        lr=LR,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        save_period=SAVE_PERIOD,
        save_num_images=SAVE_NUM_IMAGES,
    )
