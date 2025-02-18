"""Data transformation."""

import torch
import torchvision.transforms.functional as tf


class OctonionTransform:
    """Transform an MNIST image into an octonionic representation.

    Each pixel is mapped to an 8-dimensional vector (an octonion) as follows:
    - Channel 0: Normalized pixel value in [0, 1].
    - Channel 1: Normalized x coordinate (column index) in [0, 1].
    - Channel 2: Normalized y coordinate (row index) in [0, 1].
    - Channel 3 - 7: Random values drawn uniformly from [0, 1].

    Finally, each 8-dim vector is normalized to lie on the unit sphere.
    """

    def __call__(self, img):
        # If the input is not a tensor, convert it (this handles PIL.Image inputs).
        if not isinstance(img, torch.Tensor):
            img = tf.to_tensor(img)
        # For MNIST, img should have shape (C, H, W) with C==1.
        if img.dim() == 3 and img.shape[0] > 1:
            img = img[0:1, :, :]
        img = img.float()  # ensure the image is a float tensor

        # Get image dimensions.
        _, height, width = img.shape
        device = img.device

        # Create normalized coordinate grids.
        # grid_x: Each row will have values ranging from 0 to 1 (columns).
        grid_x = torch.linspace(0, 1, steps=width, device=device).repeat(height, 1)
        # grid_y: Each column will have values ranging from 0 to 1 (rows).
        grid_y = torch.linspace(0, 1, steps=height, device=device).unsqueeze(1).repeat(1, width)

        # Generate the remaining 5 random channels.
        random_channels = torch.rand(5, height, width, device=device)

        # Concatenate the components:
        # - Original pixel: shape (1, H, W)
        # - x coordinates: shape (1, H, W)
        # - y coordinates: shape (1, H, W)
        # - random channels: shape (5, H, W)
        octonion = torch.cat(
            [
                img,
                grid_x.unsqueeze(0),
                grid_y.unsqueeze(0),
                random_channels,
            ],
            dim=0,
        )  # Resulting shape: (8, H, W)

        # Normalize each pixel's 8-dim vector to lie on the unit sphere.
        # We compute the L2 norm per pixel (i.e. across the channel dimension).
        norm = torch.sqrt(torch.sum(octonion**2, dim=0, keepdim=True)) + 1e-8
        octonion_normalized = octonion / norm

        return octonion_normalized

    def __repr__(self):
        return self.__class__.__name__ + "()"
