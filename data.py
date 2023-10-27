import math
import torch
import torch.utils.data
from pathlib import Path
from torchvision import datasets, transforms
import multiprocessing

from .helpers import compute_mean_and_std, get_data_location
import matplotlib.pyplot as plt

def get_data_loaders(
    batch_size: int = 32, valid_size: float = 0.2, num_workers: int = -1, limit: int = -1
):
    # Set the number of workers to use in data loaders
    if num_workers == -1:
        num_workers = multiprocessing.cpu_count()

    # Initialize data loader dictionary
    data_loaders = {"train": None, "valid": None, "test": None}

    # Set the base path for the data
    base_path = Path(get_data_location())

    # Compute mean and standard deviation of the dataset
    mean, std = compute_mean_and_std()
    print(f"Dataset mean: {mean}, std: {std}")

    # Define data transformations for different data splits
    data_transforms = {
        "train": transforms.Compose([
            # Data augmentation for training
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(p=0.3), 
            transforms.RandomRotation(10),
            transforms.RandomChoice([
                transforms.ColorJitter(hue=0.1),
                transforms.ColorJitter(brightness=0.2),
                transforms.ColorJitter(saturation=0.2),
                transforms.ColorJitter(contrast=0.2),
            ]),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        "valid": transforms.Compose([
            # No data augmentation for validation
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std) 
        ]),
        "test": transforms.Compose([
            # No data augmentation for testing
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
    }

    # Create datasets for training, validation, and testing
    train_data = datasets.ImageFolder(
        base_path / "train",
        transform=data_transforms["train"]
    )
    valid_data = datasets.ImageFolder(
        base_path / "train",
        transform=data_transforms["valid"]
    )

    n_tot = len(train_data)
    indices = torch.randperm(n_tot)

    # Limit the number of data points if specified
    if limit > 0:
        indices = indices[:limit]
        n_tot = limit

    # Split the dataset into training and validation sets
    split = int(math.ceil(valid_size * n_tot))
    train_idx, valid_idx = indices[split:], indices[:split]

    # Define samplers for training and validation batches
    train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
    valid_sampler = torch.utils.data.SubsetRandomSampler(valid_idx) 

    # Create data loaders for training and validation
    data_loaders["train"] = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
    )
    data_loaders["valid"] = torch.utils.data.DataLoader(
        valid_data, batch_size=batch_size, sampler=valid_sampler, num_workers=num_workers
    )

    # Create data loader for testing
    test_data = datasets.ImageFolder(
        base_path / "test",
        transform=data_transforms["test"]
    )

    if limit > 0:
        indices = torch.arange(limit)
        test_sampler = torch.utils.data.SubsetRandomSampler(indices)
    else:
        test_sampler = None

    data_loaders["test"] = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, sampler=test_sampler, num_workers=num_workers
    )

    return data_loaders

def visualize_one_batch(data_loaders, max_n: int = 5):
    # Obtain one batch of training images
    dataiter  = iter(data_loaders["train"])
    images, labels  = dataiter.next()

    # Undo the normalization for visualization
    mean, std = compute_mean_and_std()
    invTrans = transforms.Compose(
        [
            transforms.Normalize(mean=[0.0, 0.0, 0.0], std=1 / std),
            transforms.Normalize(mean=-mean, std=[1.0, 1.0, 1.0]),
        ]
    )

    images = invTrans(images)

    # Get class names from the train data loader
    class_names  = data_loaders.dataset.classes

    # Convert from BGR to RGB and plot images
    images = torch.permute(images, (0, 2, 3, 1)).clip(0, 1)

    fig = plt.figure(figsize=(25, 4))
    for idx in range(max_n):
        ax = fig.add_subplot(1, max_n, idx + 1, xticks=[], yticks=[])
        ax.imshow(images[idx])
        ax.set_title(class_names[labels[idx].item()])


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    return get_data_loaders(batch_size=2, num_workers=0)


def test_data_loaders_keys(data_loaders):

    assert set(data_loaders.keys()) == {"train", "valid", "test"}, "The keys of the data_loaders dictionary should be train, valid and test"


def test_data_loaders_output_type(data_loaders):
    # Test the data loaders
    dataiter = iter(data_loaders["train"])
    images, labels = dataiter.next()

    assert isinstance(images, torch.Tensor), "images should be a Tensor"
    assert isinstance(labels, torch.Tensor), "labels should be a Tensor"
    assert images[0].shape[-1] == 224, "The tensors returned by your dataloaders should be 224x224. Did you " \
                                       "forget to resize and/or crop?"


def test_data_loaders_output_shape(data_loaders):
    dataiter = iter(data_loaders["train"])
    images, labels = dataiter.next()

    assert len(images) == 2, f"Expected a batch of size 2, got size {len(images)}"
    assert (
        len(labels) == 2
    ), f"Expected a labels tensor of size 2, got size {len(labels)}"


def test_visualize_one_batch(data_loaders):

    visualize_one_batch(data_loaders, max_n=2)