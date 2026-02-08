# Application Overview

## Purpose
This application computes the **Fréchet Inception Distance (FID)** between two
sets of images: a set of *real* images and a set of *generated* (fake) images.
The FID score is commonly used to evaluate how close generated images are to
real images by comparing their feature distributions using an InceptionV3
network.

## How It Works
- Loads PNG images from the `real_images/` and `fake_images/` folders.
- Resizes images to 299×299 and preprocesses them for InceptionV3.
- Uses a pretrained **InceptionV3** model (without the top classifier) to
  extract feature activations for both image sets.
- Computes the FID score using the mean and covariance statistics of the
  activations.

## Inputs
- `real_images/`: Folder containing real reference images.
- `fake_images/`: Folder containing generated images to compare.

## Output
- Prints a single line to the console: `FID: <score>`

## Running the App
From the project root:

```bash
uv run python main.py
```

## Notes
- Only `.png` files are processed by default.
- Images with alpha channels have the alpha channel removed before processing.
