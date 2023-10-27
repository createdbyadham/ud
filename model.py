import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:
        super().__init__()
        # Define a CNN architecture with specific layer configurations.
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),  # 3-channel input, 16 output channels, 3x3 kernel with padding
            nn.BatchNorm2d(16),  # Batch normalization for the 16 channels
            nn.ReLU(),  # Rectified Linear Unit activation function
            nn.MaxPool2d(2, 2),  # Max-pooling layer with a 2x2 kernel

            nn.Conv2d(16, 32, 3, padding=1),  # 16 input channels, 32 output channels
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, 3, padding=1),  # 32 input channels, 64 output channels
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, padding=1),  # 64 input channels, 128 output channels
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, 3, padding=1),  # 128 input channels, 256 output channels
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Flatten(),  # Flatten the feature maps
            nn.Dropout(dropout),  # Dropout layer with specified dropout rate
            nn.Linear(7 * 7 * 256, 2048),  # Fully connected layer
            nn.ReLU(),

            nn.BatchNorm1d(2048),  # Batch normalization for the fully connected layer
            nn.Dropout(p=0.2),  # Additional dropout layer
            nn.Linear(2048, num_classes)  # Output layer with the specified number of classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward pass through the model
        return self.model(x)

######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    model = MyModel(num_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = dataiter.next()

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"