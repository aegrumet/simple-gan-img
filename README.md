# Basic Image GAN

A generative adversarial network that tries to mimic a simple greyscale image.

Based on Bert Gollnick's [GAN
exercises](https://github.com/DataScienceHamburg/PyTorchUltimateMaterial/tree/main/220_GAN)
for the [PyTorch Ultimate class](https://www.udemy.com/course/pytorch-ultimate/).

# Requirements

* Python 3
* Compatible versions of Python, PyTorch, and Torchvision - [this table](https://pypi.org/project/torchvision/) may come in handy.
* A machine that has a GPU or Apple Silicon, ideally

The package versions in requirements.txt were tested with Python 3.10.16. If you
use a different version, update requirements.txt accordingly.

# Basic usage

## Training image

A 64x64 greyscale training image of a heart is provided in
[heart.png](heart.png). This can also be regenerated by running

```sh
python gan_img.py create-training-image
```

## Training

To train the GAN, run

```sh
python gan_img.py train
```

This will run 50 training epochs and display a series of images.

The first image is the training image.

Each subsequent image is the generator output after 5 additional training epochs.
The generated images should converge to the training image.

At the end of training, the discriminator and model states are saved to files as
indicated by the program output.

## Sweep

To sweep through a range of latent space inputs and examine the resulting
outputs, run

```sh
python gan_img.py sweep
```

To run this, you must first train the model and save it to file.

With ffmpeg or a similar tool, it's possible to create a movie of the sweep.

As an example movie see [sweep.mp4](sweep.mp4) in this repository.

The sweep shows an interesting trainsition around the latent value of 0. It's on
my TO-DO list to understand why :-)