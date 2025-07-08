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
):
    """
    Download and load the MNIST or Fashion-MNIST dataset.

    Args:
        dataset (str): Dataset to load - either "mnist" or "fashion-mnist"
        normalize (bool): If True, normalize pixel values to [0, 1]
        flatten (bool): If True, flatten 28x28 images to 784-dimensional vectors
        one_hot (bool): If True, convert labels to one-hot encoding
        subset_size (int): If specified, return only a subset of the data

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

    # Normalize pixel values and convert to GPU arrays
    if normalize:
        # Convert to float32 first
        X_train = X_train.astype(np.float32) / 255.0
        X_test = X_test.astype(np.float32) / 255.0

        # Calculate global mean and std from training data
        mean_val = X_train.mean()
        std_val = X_train.std()

        # Standardize to mean=0, std=1
        X_train = (X_train - mean_val) / std_val
        X_test = (X_test - mean_val) / std_val

        # Convert to CuPy arrays
        X_train = cp.array(X_train)
        X_test = cp.array(X_test)
    else:
        X_train = cp.array(X_train)
        X_test = cp.array(X_test)

    # Flatten images if needed (they're already flattened in mnist_784)
    if not flatten:
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


def load_cifar10_data(normalize=True, flatten=True, one_hot=True, subset_size=None):
    """
    Download and load the CIFAR-10 dataset.
    Args:
        normalize (bool): If True, normalize pixel values to [0, 1]
        flatten (bool): If True, keep images as 3072-dimensional vectors.
                        If False, reshape to (batch, 3, 32, 32).
        one_hot (bool): If True, convert labels to one-hot encoding
        subset_size (int): If specified, return only a subset of the data
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

    # Normalize pixel values and convert to GPU arrays
    if normalize:
        # Convert to float32 first
        X_train = X_train.astype(np.float32) / 255.0
        X_test = X_test.astype(np.float32) / 255.0

        # Calculate global mean and std from training data
        mean_val = X_train.mean()
        std_val = X_train.std()

        # Standardize to mean=0, std=1
        X_train = (X_train - mean_val) / std_val
        X_test = (X_test - mean_val) / std_val

        # Convert to CuPy arrays
        X_train = cp.array(X_train)
        X_test = cp.array(X_test)
    else:
        X_train = cp.array(X_train)
        X_test = cp.array(X_test)

    # Reshape if not flattening (data is flat by default from fetch_openml)
    if not flatten:
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
