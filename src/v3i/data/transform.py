"""Data transformation."""

import os

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.transforms.functional as tf


class MnistOctonionTransform:
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


def serialize_mnist(dataset: datasets.MNIST, output_file: str):
    """Converts a dataset into a dictionary of data and labels and serializes it to disk.

    Args:
        dataset: A torchvision dataset object.
        output_file: The file path where the serialized data will be saved.
    """
    data_list = []
    label_list = []

    # Loop through the dataset and collect transformed samples.
    for idx in range(len(dataset)):
        sample, label = dataset[idx]
        # Optionally, add a batch dimension for each sample.
        data_list.append(sample.unsqueeze(0))  # Shape becomes (1, 8, H, W)
        label_list.append(label)

    # Concatenate all samples into a single tensor.
    data_tensor = torch.cat(data_list, dim=0)  # Final shape: (N, 8, H, W)
    labels_tensor = torch.tensor(label_list)

    # Create a dictionary to hold the dataset.
    dataset_dict = {
        "data": data_tensor,
        "labels": labels_tensor,
    }

    # Make sure the output directory exists.
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Serialize using torch.save.
    torch.save(dataset_dict, output_file)
    print(f"Serialized dataset saved to {output_file}")


if __name__ == "__main__":
    # Define our composed transformation.
    transform = transforms.Compose([
        MnistOctonionTransform(),
    ])

    # Read the MNIST dataset from the data directory.
    mnist_dataset = datasets.MNIST(
        root="data/MNIST/raw",
        train=True,
        transform=transform,
        download=True,
    )

    # Define the path for serialized data.
    output_file = "data/MNIST/transformed/octonion_mnist_train.pt"

    # Serialize the dataset.
    serialize_mnist(mnist_dataset, output_file)
