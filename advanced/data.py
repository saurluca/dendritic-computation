import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms


def load_mnist_data(
    dataset="mnist", subset_size=None, data_dir="./data", download=True
):
    """
    Download and load the MNIST or Fashion-MNIST dataset using PyTorch.

    Args:
        dataset (str): Dataset to load - either "mnist" or "fashion-mnist"
        subset_size (int): If specified, return only a subset of the data
        data_dir (str): Directory to save/load data
        download (bool): Whether to download if not present

    Returns:
        tuple: (X_train, y_train, X_test, y_test)
            X_train, X_test: Input features (normalized and standardized)
            y_train, y_test: Target labels (one-hot encoded)
    """
    # Validation
    if dataset not in ["mnist", "fashion-mnist"]:
        raise ValueError(
            f"Dataset must be one of ['mnist', 'fashion-mnist'], got '{dataset}'"
        )

    print(f"Loading {dataset.upper()} dataset...")

    # Define transforms for normalization only (we'll standardize separately)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # Converts to [0,1] and adds channel dimension
            transforms.Lambda(lambda x: x.view(-1)),  # Flatten to 784 dimensions
        ]
    )

    # Load appropriate dataset
    if dataset == "mnist":
        train_dataset = datasets.MNIST(
            data_dir, train=True, download=download, transform=transform
        )
        test_dataset = datasets.MNIST(
            data_dir, train=False, download=download, transform=transform
        )
    else:  # fashion-mnist
        train_dataset = datasets.FashionMNIST(
            data_dir, train=True, download=download, transform=transform
        )
        test_dataset = datasets.FashionMNIST(
            data_dir, train=False, download=download, transform=transform
        )

    # Convert to tensors
    X_train = torch.stack([train_dataset[i][0] for i in range(len(train_dataset))])
    y_train = torch.tensor([train_dataset[i][1] for i in range(len(train_dataset))])
    X_test = torch.stack([test_dataset[i][0] for i in range(len(test_dataset))])
    y_test = torch.tensor([test_dataset[i][1] for i in range(len(test_dataset))])

    # Calculate global mean and std from training data for standardization
    mean_val = X_train.mean()
    std_val = X_train.std()

    # Standardize to mean=0, std=1
    X_train = (X_train - mean_val) / std_val
    X_test = (X_test - mean_val) / std_val

    # Convert labels to one-hot encoding
    def to_one_hot(labels, n_classes=10):
        return F.one_hot(labels.long(), num_classes=n_classes).float()

    y_train = to_one_hot(y_train)
    y_test = to_one_hot(y_test)

    # Use subset if specified
    if subset_size is not None:
        X_train, y_train = X_train[:subset_size], y_train[:subset_size]
        # Keep proportional test size
        test_size = subset_size // 6
        X_test, y_test = X_test[:test_size], y_test[:test_size]

    print(f"Training data shape: {X_train.shape}, {y_train.shape}")
    print(f"Test data shape: {X_test.shape}, {y_test.shape}")
    print(f"Data standardized with mean={mean_val:.4f}, std={std_val:.4f}")

    return X_train, y_train, X_test, y_test


def create_data_loaders(
    X_train, y_train, X_test, y_test, batch_size=128, shuffle_train=True, num_workers=0
):
    """
    Create PyTorch DataLoaders for training and testing.

    Args:
        X_train, y_train: Training data and labels
        X_test, y_test: Test data and labels
        batch_size (int): Batch size for training
        shuffle_train (bool): Whether to shuffle training data
        num_workers (int): Number of workers for data loading

    Returns:
        tuple: (train_loader, test_loader)
    """
    # Create datasets
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, test_loader


def get_sample_image(X_train, y_train, class_idx=None, image_idx=0):
    """
    Get a sample image from the training data for visualization.

    Args:
        X_train: Training images tensor
        y_train: Training labels tensor (one-hot encoded)
        class_idx (int): If specified, get image from this class
        image_idx (int): Index of image to get from the class

    Returns:
        tuple: (image, label) where image is flattened and label is class index
    """
    if class_idx is not None:
        # Find indices of images belonging to the specified class
        class_indices = torch.where(torch.argmax(y_train, dim=1) == class_idx)[0]
        if len(class_indices) == 0:
            raise ValueError(f"No images found for class {class_idx}")
        if image_idx >= len(class_indices):
            raise ValueError(
                f"Image index {image_idx} out of range for class {class_idx}"
            )
        actual_idx = class_indices[image_idx]
    else:
        actual_idx = image_idx

    image = X_train[actual_idx]
    label = torch.argmax(y_train[actual_idx]).item()

    return image, label
