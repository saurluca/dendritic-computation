# %%
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.datasets import fetch_openml


def load_mnist_data(normalize=True, flatten=True, one_hot=True, subset_size=None):
    """
    Download and load the MNIST dataset.

    Args:
        normalize (bool): If True, normalize pixel values to [0, 1]
        flatten (bool): If True, flatten 28x28 images to 784-dimensional vectors
        one_hot (bool): If True, convert labels to one-hot encoding
        subset_size (int): If specified, return only a subset of the data

    Returns:
        tuple: (X_train, y_train, X_test, y_test)
            X_train, X_test: Input features
            y_train, y_test: Target labels
    """
    print("Loading MNIST dataset...")

    # Download MNIST dataset
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto", cache=True)
    X, y = mnist.data, mnist.target.astype(int)

    # Split into train and test (last 10k samples for test, rest for train)
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]

    # Normalize pixel values
    if normalize:
        X_train = X_train.astype(np.float32) / 255.0
        X_test = X_test.astype(np.float32) / 255.0

    # Flatten images if needed (they're already flattened in mnist_784)
    if not flatten:
        X_train = X_train.reshape(-1, 28, 28)
        X_test = X_test.reshape(-1, 28, 28)

    # Convert labels to one-hot encoding
    if one_hot:

        def to_one_hot(labels, n_classes=10):
            one_hot_labels = np.zeros((len(labels), n_classes))
            one_hot_labels[np.arange(len(labels)), labels] = 1
            return one_hot_labels

        y_train = to_one_hot(y_train)
        y_test = to_one_hot(y_test)

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


class Sigmoid:
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, x):
        return self._sigmoid(x)

    def backward(self, grad):
        return self._sigmoid(grad) * (1 - self._sigmoid(grad)) * grad

    def __call__(self, x):
        return self.forward(x)


class CrossEntropy:
    def __init__(self):
        self.softmax_output = None
        self.target = None

    def forward(self, logits, target):
        # Apply softmax
        exp_logits = np.exp(logits - np.max(logits))
        self.softmax_output = exp_logits / np.sum(exp_logits)
        self.target = target

        # Compute cross entropy loss
        return -np.sum(target * np.log(self.softmax_output + 1e-15))

    def backward(self):
        return self.softmax_output - self.target

    def __call__(self, logits, target):
        return self.forward(logits, target)


class ReLU:
    def forward(self, x):
        return np.maximum(0, x)

    def backward(self, grad):
        return np.where(grad > 0, grad, 0)

    def __call__(self, x):
        return self.forward(x)


class MSE:
    def __init__(self):
        self.pred = None
        self.target = None

    def forward(self, pred, target):
        self.pred, self.target = pred, target
        return np.mean((pred - target) ** 2)

    def backward(self):
        return np.mean(0.5 * (self.pred - self.target)).reshape(1)

    def __call__(self, pred, target):
        return self.forward(pred, target)


class SGD:
    def __init__(self, params, criterion, lr=0.01, momentum=0.9):
        self.params = params
        self.criterion = criterion
        self.lr = lr
        self.momentum = momentum
        self.updates = [
            [np.zeros_like(layer.W), np.zeros_like(layer.b)] for layer in self.params
        ]

    def zero_grad(self):
        for layer in self.params:
            layer.dW = 0.0
            layer.db = 0.0

    def step(self):
        for layer, update in zip(self.params, self.updates):
            update[0] = self.lr * layer.dW + self.momentum * update[0]
            update[1] = self.lr * layer.db + self.momentum * update[1]
            layer.W -= update[0]
            layer.b -= update[1]

    def __call__(self):
        return self.step()
    

class DendriteSGD:
    def __init__(self, params, criterion, lr=0.01, momentum=0.9):
        self.params = params
        self.criterion = criterion
        self.lr = lr
        self.momentum = momentum
        self.updates = [
            [np.zeros_like(layer.dendrite_W), np.zeros_like(layer.dendrite_b), np.zeros_like(layer.soma_W), np.zeros_like(layer.soma_b)] for layer in self.params
        ]

    def zero_grad(self):
        for layer in self.params:
            layer.dendrite_dW = 0.0
            layer.dendrite_db = 0.0
            layer.soma_dW = 0.0
            layer.soma_db = 0.0

    def step(self):
        for layer, update in zip(self.params, self.updates):
            update[0] = self.lr * layer.dendrite_dW + self.momentum * update[0]
            update[1] = self.lr * layer.dendrite_db + self.momentum * update[1]
            update[2] = self.lr * layer.soma_dW + self.momentum * update[2]
            update[3] = self.lr * layer.soma_db + self.momentum * update[3]
            layer.dendrite_W -= update[0]
            layer.dendrite_b -= update[1]
            layer.soma_W -= update[2]
            layer.soma_b -= update[3]

    def __call__(self):
        return self.step()


class Sequential:
    def __init__(self, layers):
        self.layers = layers

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def params(self):
        """Return a list of layers that have a Weight vectors"""
        params = []
        for layer in self.layers:
            if hasattr(layer, "W"):
                params.append(layer)
        return params

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class LinearLayer:
    """A fully connected, feed forward layer"""

    def __init__(self, in_dim, out_dim):
        self.W = np.random.randn(out_dim, in_dim) * np.sqrt(
            2.0 / (in_dim + out_dim)
        )  # xavier init
        self.b = np.zeros(out_dim)
        self.dW = 0.0
        self.db = 0.0
        self.x = None

    def forward(self, x):
        # print(f"x: {x}, self.W {self.W}, self.b {self.b}")
        self.x = x
        return self.W @ x + self.b

    def backward(self, grad):
        # print(f"shape of incoming grad {grad} \n shape of W {self.W.shape}")
        self.dW = np.outer(grad, self.x)
        self.db = grad
        grad = self.W.T @ grad
        return grad

    def __call__(self, x):
        return self.forward(x)


class DendriticLayer:
    """A sparse dendritic layer, consiting of dendrites and somas"""

    def __init__(
        self, in_dim, out_dim, strategy="random", n_dendrite_inputs=16, n_dendrites=3
    ):
        assert strategy == "random", "Invalid strategy"

        n_neurons = out_dim  # the number of neurons determins the size of the output
        n_soma_connections = (
            n_dendrites * n_neurons
        )  # number of possible connection from dendrites to somas
        # print(
        #     f"input dimension {in_dim}, n_neurons: {n_neurons}, n_soma_connections: {n_soma_connections}, n_dendrite_connections: {n_dendrite_connections}"
        # )

        self.dendrite_W = np.random.randn(n_soma_connections, in_dim)
        self.dendrite_b = np.zeros((n_soma_connections))
        self.dendrite_dW = 0.0
        self.dendrite_db = 0.0

        self.dendrite_activation = ReLU()

        self.soma_W = np.random.randn(n_neurons, n_soma_connections)
        self.soma_b = np.zeros(n_neurons)
        self.soma_dW = 0.0
        self.soma_db = 0.0

        self.soma_activation = ReLU()

        # inputs to save for backprop
        self.x_dendrite = None
        self.x_soma = None

        # sample soma mask:
        # [[1, 1, 0, 0]
        #  [0, 0, 1, 1]]
        # number of 1 per row is n_dendrites, rest 0. every column only has 1 entry
        # number of rows equals n_neurons, number of columns eqais n_soma_connections
        # it is a step pattern, so the first n_dendrites entries of the first row are one.
        self.soma_mask = np.zeros((n_neurons, n_soma_connections))
        for i in range(n_neurons):
            start_idx = i * n_dendrites
            end_idx = start_idx + n_dendrites
            self.soma_mask[i, start_idx:end_idx] = 1

        # mask out unneeded weights, thus making weights sparse
        self.soma_W = self.soma_W * self.soma_mask

        # sample dendrite mask
        # for each dendrite sample n_dendrite_inputs from the input array
        self.dendrite_mask = np.zeros((n_soma_connections, in_dim))
        for i in range(n_soma_connections):
            if strategy == "random":
                # sample without replacement from possible input for a given dendrite from the whole input
                input_idx = np.random.choice(
                    np.arange(in_dim), size=n_dendrite_inputs, replace=False
                )
            self.dendrite_mask[i, input_idx] = 1

        # mask out unneeded weights, thus making weights sparse
        self.dendrite_W = self.dendrite_W * self.dendrite_mask

    def forward(self, x):
        # pass through dendrites
        # print(f"x dendrite: {x.shape}, self.dendrite_W shape {self.dendrite_W.shape}")
        self.x_dendrite = x
        x = self.dendrite_W @ x + self.dendrite_b
        x = self.dendrite_activation(x)

        # pass through soma
        # print(f"x soma: {x.shape}, self.soma_W shape {self.soma_W.shape}")
        self.x_soma = x
        x = self.soma_W @ x + self.soma_b
        return self.soma_activation(x)

    def backward(self, grad):
        # print(f"shape of incoming grad {grad} \n shape of W {self.W.shape}")
        # self.dW = np.outer(grad, self.x)
        # self.db = grad
        # grad = self.W.T @ grad
        # return grad
        pass

    def __call__(self, x):
        return self.forward(x)


def train(
    X_train,
    y_train,
    model,
    criterion,
    optimiser,
    n_epochs=10,
):
    train_losses = []
    accuracy = []
    n_samples = len(X_train)
    for epoch in tqdm(range(n_epochs)):
        train_loss = 0.0
        correct_pred = 0.0
        # TODO shuffle train data ach epoch
        for X, target in zip(X_train, y_train):
            # forward pass
            pred = model(X)
            loss = criterion(pred, target)
            train_loss += loss
            # if most likely prediction eqauls target add to correct predictions
            correct_pred += np.argmax(pred) == np.argmax(target)

            # backward pass
            optimiser.zero_grad()
            grad = criterion.backward()
            model.backward(grad)
            optimiser.step()

            # print(f"y {target}, pred {pred}, loss {loss}")
        train_losses.append(train_loss)
        epoch_accuracy = correct_pred / n_samples
        accuracy.append(epoch_accuracy)
    return train_losses, accuracy


def plot_loss(losses):
    plt.plot(losses)
    plt.title("Loss over epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()


def plot_accuracy(accuracy):
    plt.plot(accuracy)
    plt.title("Accuarcy over epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuarcy")
    plt.show()


def main():
    # for repoducability
    np.random.seed(42)

    # config
    n_epochs = 2
    lr = 0.1
    in_dim = 28 * 28  # MNIST dimension
    n_classes = 10

    # load data
    X_train, y_train, X_test, y_test = load_mnist_data()
    
    model = DendriticLayer(in_dim, n_classes)
    criterion = CrossEntropy()
        
    # pred = model(X_train[1])
    # loss = criterion(pred, y_train[1])
    # print(f"prediction {pred}, true y {y_train[1]},\n loss {loss}")  
    
    optimiser = DendriteSGD([model], criterion, lr=lr, momentum=0.9)

    train_losses, accuracy = train(X_train, y_train, model, criterion, optimiser, n_epochs)
    plot_loss(train_losses)
    plot_accuracy(accuracy)
    
    # plot_predictions(outputs[-1], targets)
    print(f"final loss {train_losses[-1]}")
    print(f"final accuracy {accuracy[-1]}")


if __name__ == "main":
    main()


main()
