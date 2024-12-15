import os
import torch
from torch.utils.data import DataLoader
from torch import nn
import torchvision.transforms as transforms
from torchvision.io import read_image, ImageReadMode

import matplotlib.pyplot as plt
import numpy as np
from random import uniform

import seaborn as sns
import click


def select_device_name():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using device: {device}")
    return device


class GANImageGenerator(nn.Module):
    def __init__(self, latent_dim, image_channels, feature_maps=64):
        super(GANImageGenerator, self).__init__()

        self.net = nn.Sequential(
            # Latent vector to dense feature map
            nn.ConvTranspose2d(
                latent_dim, feature_maps * 8, 4, 1, 0, bias=False
            ),  # out: BS, feature_maps*8, 4, 4
            nn.BatchNorm2d(feature_maps * 8),
            nn.ReLU(True),
            # Feature map -> 2x upscaling
            nn.ConvTranspose2d(
                feature_maps * 8, feature_maps * 4, 4, 2, 1, bias=False
            ),  # out: BS, feature_maps*4, 8, 8
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True),
            # Feature map -> 2x upscaling
            nn.ConvTranspose2d(
                feature_maps * 4, feature_maps * 2, 4, 2, 1, bias=False
            ),  # out: BS, feature_maps*2, 16, 16
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),
            # Feature map -> 2x upscaling
            nn.ConvTranspose2d(
                feature_maps * 2, feature_maps, 4, 2, 1, bias=False
            ),  # out: BS, feature_maps, 32, 32
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(True),
            # Final layer to generate image
            nn.ConvTranspose2d(
                feature_maps, image_channels, 4, 2, 1, bias=False
            ),  # out: BS, image_channels, 64, 64
            nn.Tanh(),  # Output pixel range: [-1, 1]
        )

    def forward(self, x):
        return self.net(x)


@click.command()
@click.pass_context
def create_training_image(ctx):

    training_image_file = ctx.parent.obj["training_image_file"]

    point_count = 4096
    theta = np.array([uniform(0, 2 * np.pi) for _ in range(point_count)])

    # Data points outlining a heart shape
    x = 16 * (np.sin(theta) ** 3)
    y = (
        13 * np.cos(theta)
        - 5 * np.cos(2 * theta)
        - 2 * np.cos(3 * theta)
        - np.cos(4 * theta)
    )

    sns.set_theme(rc={"figure.figsize": (12, 12)})
    sns.set_style("white")
    sns_plot = sns.scatterplot(x=x, y=y, marker="o", color="black", linewidth=0)
    sns.despine(left=True, bottom=True)
    plt.xticks([])
    plt.yticks([])

    fig = sns_plot.get_figure()
    fig.savefig(training_image_file)

    print(f"Training image saved to {training_image_file}")


@click.command()
@click.pass_context
def train(ctx):

    # 1. CHECK INPUTS

    training_image_file = ctx.parent.obj["training_image_file"]
    try:
        with open(training_image_file):
            pass
    except FileNotFoundError:
        print(f"Training image file {training_image_file} not found")
        return

    generator_model_file = ctx.parent.obj["generator_model_file"]
    discriminator_model_file = ctx.parent.obj["discriminator_model_file"]

    train_progress_directory = ctx.parent.obj["train_progress_directory"]
    try:
        os.makedirs(train_progress_directory)
    except FileExistsError:
        pass

    # 2. SET HARDWARE DEVICE

    device_name = select_device_name()
    device = torch.device(device_name)
    torch.set_default_device(device_name)

    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((64, 64)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),  # Rescales to [-1, 1]
        ]
    )

    # 3. LOAD TRAINING IMAGE

    img = read_image(
        path=training_image_file,
        mode=ImageReadMode.UNCHANGED,
        apply_exif_orientation=False,
    )
    img = transform(img)
    img_pil = transforms.ToPILImage()(img)
    img_pil.show()

    # 4. PREPARE TENSORS AND DATA LOADER

    TRAIN_DATA_COUNT = 1024
    train_data = img.to(device)
    train_set = [
        # This is the true image so the label is always 1
        (train_data, 1)
        for i in range(TRAIN_DATA_COUNT)
    ]

    # Create the true positives data loader
    BATCH_SIZE = 32
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE)

    # Create the discriminator
    discriminator = nn.Sequential(
        nn.Conv2d(1, 64, 3),  # out: BS, 64, 62, 62
        nn.ReLU(),
        nn.MaxPool2d(2, 2),  # out: BS, 64, 31, 31
        nn.Dropout(0.3),
        nn.Conv2d(64, 32, 3),  # out: BS, 32, 29, 29
        nn.ReLU(),
        nn.MaxPool2d(2, 2),  # out: BS, 32, 14, 14
        nn.Dropout(0.3),
        nn.Flatten(),
        nn.Linear(32 * 14 * 14, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(64, 1),
        nn.Sigmoid(),
    )

    # Test the discriminator
    # input = torch.rand((1, 1, 64, 64))
    # d = discriminator(input)
    # print(d.shape)

    # Create the generator
    latent_dim = 1
    image_channels = 1
    generator = GANImageGenerator(latent_dim, image_channels)

    # Test the generator
    # latent_vector = torch.randn(1, latent_dim, 1, 1, device=device)  # Shape: [batch_size, latent_dim, 1, 1]
    # generated_image = generator(latent_vector)
    # print(generated_image.shape)  # Should be [1, 3, 64, 64] for a 64x64 RGB image

    # train loop
    LR = 0.001
    NUM_EPOCHS = 51
    loss_function = nn.BCELoss()
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters())
    optimizer_generator = torch.optim.Adam(generator.parameters())

    for epoch in range(NUM_EPOCHS):
        for n, (real_samples, _) in enumerate(train_loader):
            # Training the discriminator
            real_samples_labels = torch.ones((BATCH_SIZE, 1), device=device)
            latent_space_samples = torch.randn(
                BATCH_SIZE, latent_dim, 1, 1, device=device
            )
            generated_samples = generator(latent_space_samples)
            generated_samples_labels = torch.zeros((BATCH_SIZE, 1))
            all_samples = torch.cat((real_samples, generated_samples), dim=0)
            all_samples_labels = torch.cat(
                (real_samples_labels, generated_samples_labels), dim=0
            )

            # Data for training the discriminator
            if epoch % 2 == 0:
                discriminator.zero_grad()
                output_discriminator = discriminator(all_samples)
                loss_discriminator = loss_function(
                    output_discriminator, all_samples_labels
                )
                loss_discriminator.backward()
                optimizer_discriminator.step()

            if epoch % 2 == 1:
                # Data for training the generator
                # latent_space_samples = torch.randn((BATCH_SIZE, 1))
                latent_space_samples = torch.randn(BATCH_SIZE, latent_dim, 1, 1)

                # Training the generator
                generator.zero_grad()
                generated_samples = generator(latent_space_samples)
                output_discriminator_generated = discriminator(generated_samples)
                loss_generator = loss_function(
                    output_discriminator_generated, real_samples_labels
                )
                loss_generator.backward()
                optimizer_generator.step()

        # Show progress
        if epoch % 5 == 0 and epoch > 0:
            print(epoch)
            print(f"Epoch {epoch}, Discriminator Loss {loss_discriminator}")
            print(f"Epoch {epoch}, Generator Loss {loss_generator}")
            with torch.no_grad():
                latent_space_samples = torch.randn(1, latent_dim, 1, 1, device=device)
                generated_samples = generator(latent_space_samples)
            generated_image = transforms.ToPILImage()(generated_samples[0])
            generated_image.show()
            generated_image.save(
                f"{train_progress_directory}/image{str(epoch).zfill(4)}.png",
                format="PNG",
            )

        torch.save(generator.state_dict(), generator_model_file)
        torch.save(discriminator.state_dict(), discriminator_model_file)

    print("Training complete")
    print(f"Generator model saved to {generator_model_file}")
    print(f"Discriminator model saved to {discriminator_model_file}")


@click.command()
@click.pass_context
def sweep(ctx):
    # validate inputs
    generator_model_file = ctx.parent.obj["generator_model_file"]
    try:
        with open(generator_model_file):
            pass
    except FileNotFoundError:
        print(f"Generator model file {generator_model_file} not found")
        return

    sweep_directory = ctx.parent.obj["sweep_directory"]
    try:
        os.makedirs(sweep_directory)
    except FileExistsError:
        pass

    # load generator from disk
    latent_dim = 1
    image_channels = 1
    generator = GANImageGenerator(latent_dim, image_channels)
    generator.load_state_dict(torch.load(generator_model_file))
    generator.eval()

    # generate images for a sweep of latent space values
    values = np.linspace(-1, 1, 100, dtype=np.float64)
    tensors = [
        torch.tensor(value, dtype=torch.float32).view(1, 1, 1, 1) for value in values
    ]
    for i in range(len(values)):
        with torch.no_grad():
            latent_space_samples = tensors[i]
            generated_samples = generator(latent_space_samples)
        generated_image = transforms.ToPILImage()(generated_samples[0])
        # generated_image.show()
        generated_image.save(
            f"{sweep_directory}/image{str(i).zfill(4)}.png", format="PNG"
        )

    print(f"Latent-space sweep images saved to {sweep_directory}")
    print(f"You can make a video from these images using ffmpeg with the command:")
    print(
        f"ffmpeg -framerate 10 -i {sweep_directory}/image%04d.png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p sweep.mp4"
    )


@click.group()
@click.pass_context
@click.option(
    "--training-image-file", default="heart.png", help="path to training image file"
)
@click.option(
    "--train-progress-directory",
    default="train_progress",
    help="path to directory to save training progress images",
)
@click.option(
    "--generator-model-file",
    default="generator.pth",
    help="path to generator model file, to save or load from",
)
@click.option(
    "--discriminator-model-file",
    default="discriminator.pth",
    help="path to discriminator model file, to save or load from",
)
@click.option(
    "--sweep-directory", default="sweep", help="path to directory to save sweep images"
)
def cli(
    ctx,
    training_image_file,
    train_progress_directory,
    generator_model_file,
    discriminator_model_file,
    sweep_directory,
):
    ctx.obj = {
        "training_image_file": training_image_file,
        "train_progress_directory": train_progress_directory,
        "generator_model_file": generator_model_file,
        "discriminator_model_file": discriminator_model_file,
        "sweep_directory": sweep_directory,
    }


cli.add_command(train)
cli.add_command(create_training_image)
cli.add_command(sweep)

if __name__ == "__main__":
    cli()
