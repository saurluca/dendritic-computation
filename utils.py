try:
    import cupy as cp

    # Test if CuPy can actually access CUDA and random number generator
    cp.cuda.Device(0).compute_capability
    cp.random.seed(1)  # Test if random number generator works
    print("Using CuPy (GPU acceleration)")
except (ImportError, Exception) as e:
    import numpy as cp

    print(f"CuPy not available or CUDA error ({type(e).__name__}), using NumPy (CPU)")
from sklearn.datasets import fetch_openml
import numpy as np


def load_mnist_data(
    rng=None,
    dataset="mnist",
    normalize=True,
    flatten=True,
    one_hot=True,
    subset_size=None,
    shuffle=False,
    data_augmentation=False,
):
    """
    Download and load the MNIST or Fashion-MNIST dataset.

    Args:
        dataset (str): Dataset to load - either "mnist" or "fashion-mnist"
        normalize (bool): If True, normalize pixel values using standard MNIST normalization
        flatten (bool): If True, flatten 28x28 images to 784-dimensional vectors
        one_hot (bool): If True, convert labels to one-hot encoding
        subset_size (int): If specified, return only a subset of the data
        data_augmentation (bool): If True, apply horizontal/vertical flips to training data only

    Returns:
        tuple: (X_train, y_train, X_test, y_test)
            X_train, X_test: Input features
            y_train, y_test: Target labels
    """
    # Map dataset names to OpenML dataset identifiers
    dataset_mapping = {"mnist": "mnist_784", "fashion-mnist": "Fashion-MNIST"}

    if dataset not in dataset_mapping:
        raise ValueError(
            f"Dataset must be one of {list(dataset_mapping.keys())}, got '{dataset}'"
        )

    dataset_name = dataset_mapping[dataset]
    print(f"Loading {dataset.upper()} dataset...")

    # Download dataset
    data = fetch_openml(
        dataset_name, version=1, as_frame=False, parser="auto", cache=True
    )

    # Shuffle both data and labels together to maintain correspondence
    n_samples = len(data.data)
    if shuffle:
        shuffle_indices = np.arange(n_samples)
        rng.shuffle(shuffle_indices)

    X, y = data.data, data.target.astype(int)
    if shuffle:
        X, y = X[shuffle_indices], y[shuffle_indices]

    # Split into train and test (last 10k samples for test, rest for train)
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]

    # Normalize pixel values using standard MNIST normalization
    if normalize:
        # Standard MNIST normalization values
        mnist_mean = 0.1307
        mnist_std = 0.3081

        # Convert to float32 and normalize to [0,1]
        X_train = X_train.astype(np.float32) / 255.0
        X_test = X_test.astype(np.float32) / 255.0

        # Apply standard normalization: (data - mean) / std
        X_train = (X_train - mnist_mean) / mnist_std
        X_test = (X_test - mnist_mean) / mnist_std

        # Convert to CuPy arrays
        X_train = cp.array(X_train)
        X_test = cp.array(X_test)
    else:
        X_train = cp.array(X_train)
        X_test = cp.array(X_test)

    # Apply data augmentation to training data only
    if data_augmentation:
        print(
            "Applying data augmentation to training data only (horizontal/vertical flips)..."
        )

        # Reshape to 2D images for augmentation
        X_train_img = X_train.reshape(-1, 28, 28)

        # Create augmented versions for training data only
        X_train_h_flip = cp.flip(X_train_img, axis=2)  # Horizontal flip
        X_train_v_flip = cp.flip(X_train_img, axis=1)  # Vertical flip
        X_train_hv_flip = cp.flip(cp.flip(X_train_img, axis=1), axis=2)  # Both flips

        # Concatenate all versions for training data
        X_train = cp.concatenate(
            [X_train_img, X_train_h_flip, X_train_v_flip, X_train_hv_flip], axis=0
        )

        # Replicate training labels 4 times
        y_train = cp.concatenate(
            [
                cp.array(y_train),
                cp.array(y_train),
                cp.array(y_train),
                cp.array(y_train),
            ],
            axis=0,
        )

        print(f"Augmented training data shape: {X_train.shape}")
        print(f"Test data shape (unchanged): {X_test.shape}")

    # Flatten images if needed
    if flatten:
        if data_augmentation:
            X_train = X_train.reshape(X_train.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)
    else:
        if not data_augmentation:
            X_train = X_train.reshape(-1, 28, 28)
        X_test = X_test.reshape(-1, 28, 28)

    # Convert labels to one-hot encoding
    if one_hot:

        def to_one_hot(labels, n_classes=10):
            one_hot_labels = cp.zeros((len(labels), n_classes))
            one_hot_labels[cp.arange(len(labels)), labels] = 1
            return one_hot_labels

        y_train = to_one_hot(cp.array(y_train))
        y_test = to_one_hot(cp.array(y_test))
    else:
        y_train = cp.array(y_train)
        y_test = cp.array(y_test)

    # Use subset if specified
    if subset_size is not None:
        X_train, y_train = X_train[:subset_size], y_train[:subset_size]
        X_test, y_test = (
            X_test[: subset_size // 6],
            y_test[: subset_size // 6],
        )  # Keep proportional test size

    print(f"Training data shape: {X_train.shape}, {y_train.shape}")
    print(f"Test data shape: {X_test.shape}, {y_test.shape}")

    return X_train, y_train, X_test, y_test


def load_cifar10_data(
    normalize=True,
    flatten=True,
    one_hot=True,
    subset_size=None,
    data_augmentation=False,
):
    """
    Download and load the CIFAR-10 dataset.
    Args:
        normalize (bool): If True, normalize pixel values using standard CIFAR-10 normalization
        flatten (bool): If True, keep images as 3072-dimensional vectors.
                        If False, reshape to (batch, 3, 32, 32).
        one_hot (bool): If True, convert labels to one-hot encoding
        subset_size (int): If specified, return only a subset of the data
        data_augmentation (bool): If True, apply horizontal/vertical flips to training data only
    Returns:
        tuple: (X_train, y_train, X_test, y_test)
    """
    dataset_name = "CIFAR_10"
    print(f"Loading {dataset_name} dataset...")

    # Download dataset
    data = fetch_openml(
        dataset_name, version=1, as_frame=False, parser="auto", cache=True
    )

    X, y = data.data, data.target.astype(int)

    print("finished downloading data")

    # Split into train and test (50k train, 10k test)
    X_train, X_test = X[:50000], X[50000:]
    y_train, y_test = y[:50000], y[50000:]

    # Normalize pixel values using standard CIFAR-10 normalization
    if normalize:
        # Standard CIFAR-10 normalization values (per channel)
        cifar10_mean = np.array([0.4914, 0.4822, 0.4465])
        cifar10_std = np.array([0.2470, 0.2435, 0.2616])

        # Convert to float32 and normalize to [0,1]
        X_train = X_train.astype(np.float32) / 255.0
        X_test = X_test.astype(np.float32) / 255.0

        # Reshape to (batch, height, width, channels) for per-channel normalization
        X_train = X_train.reshape(-1, 32, 32, 3)
        X_test = X_test.reshape(-1, 32, 32, 3)

        # Apply standard normalization per channel: (data - mean) / std
        X_train = (X_train - cifar10_mean) / cifar10_std
        X_test = (X_test - cifar10_mean) / cifar10_std

        # Convert to CuPy arrays
        X_train = cp.array(X_train)
        X_test = cp.array(X_test)
    else:
        X_train = cp.array(X_train)
        X_test = cp.array(X_test)

    # Apply data augmentation to training data only
    if data_augmentation:
        print(
            "Applying data augmentation to training data only (horizontal/vertical flips)..."
        )

        # Reshape to 3D images for augmentation (height, width, channels)
        if normalize:
            X_train_img = X_train  # Already reshaped above
        else:
            X_train_img = X_train.reshape(-1, 32, 32, 3)

        # Create augmented versions for training data only
        X_train_h_flip = cp.flip(X_train_img, axis=2)  # Horizontal flip
        X_train_v_flip = cp.flip(X_train_img, axis=1)  # Vertical flip
        X_train_hv_flip = cp.flip(cp.flip(X_train_img, axis=1), axis=2)  # Both flips

        # Concatenate all versions for training data
        X_train = cp.concatenate(
            [X_train_img, X_train_h_flip, X_train_v_flip, X_train_hv_flip], axis=0
        )

        # Replicate training labels 4 times
        y_train = cp.concatenate(
            [
                cp.array(y_train),
                cp.array(y_train),
                cp.array(y_train),
                cp.array(y_train),
            ],
            axis=0,
        )

        print(f"Augmented training data shape: {X_train.shape}")
        print(f"Test data shape (unchanged): {X_test.shape}")
    else:
        # Reshape if not doing data augmentation and normalize was True
        if normalize:
            pass  # Already reshaped above
        else:
            X_train = X_train.reshape(-1, 32, 32, 3)
            X_test = X_test.reshape(-1, 32, 32, 3)

    # Reshape if flattening is requested
    if flatten:
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)
    elif not normalize and not data_augmentation:
        # Reshape to (batch, channels, height, width) if not flattening and not normalized yet
        X_train = X_train.reshape(-1, 3, 32, 32)
        X_test = X_test.reshape(-1, 3, 32, 32)

    # Convert labels to one-hot encoding
    if one_hot:

        def to_one_hot(labels, n_classes=10):
            one_hot_labels = cp.zeros((len(labels), n_classes))
            one_hot_labels[cp.arange(len(labels)), labels] = 1
            return one_hot_labels

        y_train = to_one_hot(cp.array(y_train))
        y_test = to_one_hot(cp.array(y_test))
    else:
        y_train = cp.array(y_train)
        y_test = cp.array(y_test)

    # Use subset if specified
    if subset_size is not None:
        X_train, y_train = X_train[:subset_size], y_train[:subset_size]
        X_test, y_test = (
            X_test[: subset_size // 5],
            y_test[: subset_size // 5],
        )  # Keep proportional test size

    print(f"Training data shape: {X_train.shape}, {y_train.shape}")
    print(f"Test data shape: {X_test.shape}, {y_test.shape}")

    return X_train, y_train, X_test, y_test
