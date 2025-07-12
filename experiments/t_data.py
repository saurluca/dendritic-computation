import torch
import torchvision
import torchvision.transforms as transforms


def load_dataset(dataset="mnist", batch_size=256, augmentation=False):
    """Load MNIST, Fashion-MNIST, or CIFAR-10 dataset with normalization

    Args:
        dataset (str): Dataset to load - "mnist", "fashion-mnist", or "cifar10"
        batch_size (int): Batch size for data loaders

    Returns:
        tuple: (train_loader, test_loader, input_dim, num_classes)
    """

    # Dataset-specific configurations
    if dataset == "mnist":
        # MNIST mean and std
        mean, std = (0.1307,), (0.3081,)
        dataset_class = torchvision.datasets.MNIST
        input_dim = 28 * 28  # 784
        num_classes = 10
        print("Loading MNIST dataset...")
    elif dataset == "fashion-mnist":
        # Fashion-MNIST mean and std
        mean, std = (0.2860,), (0.3530,)
        dataset_class = torchvision.datasets.FashionMNIST
        input_dim = 28 * 28  # 784
        num_classes = 10
        print("Loading Fashion-MNIST dataset...")
    elif dataset == "cifar10":
        # CIFAR-10 mean and std per channel (RGB)
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        dataset_class = torchvision.datasets.CIFAR10
        input_dim = 32 * 32 * 3  # 3072
        num_classes = 10
        print("Loading CIFAR-10 dataset...")
    else:
        raise ValueError(
            f"Dataset must be 'mnist', 'fashion-mnist', or 'cifar10', got '{dataset}'"
        )

    # Define transforms
    if augmentation:
        # Strong augmentation for CIFAR-10
        train_transform = transforms.Compose(
            [
                # transforms.RandomCrop(32, padding=4),
                # transforms.RandomHorizontalFlip(p=0.5),
                # transforms.RandomRotation(15),
                # transforms.ColorJitter(
                #     brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                # ),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
                # transforms.RandomErasing(
                #     p=0.1, scale=(0.02, 0.33), ratio=(0.3, 3.3), value="random"
                # ),
            ]
        )
        test_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]
        )
    else:
        # No augmentation - same transform for train and test
        train_transform = test_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]
        )

    # Load datasets
    train_dataset = dataset_class(
        root="./data", train=True, download=True, transform=train_transform
    )
    test_dataset = dataset_class(
        root="./data", train=False, download=True, transform=test_transform
    )

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=8
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=8
    )

    return train_loader, test_loader, input_dim, num_classes
