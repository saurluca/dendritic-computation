# %%
try:
    import cupy as cp
    # Test if CuPy can actually access CUDA and random number generator
    cp.cuda.Device(0).compute_capability
    cp.random.seed(1)  # Test if random number generator works
    print("Using CuPy (GPU acceleration)")
    GPU_AVAILABLE = True
except (ImportError, Exception) as e:
    import numpy as cp
    print(f"CuPy not available or CUDA error ({type(e).__name__}), using NumPy (CPU)")
    GPU_AVAILABLE = False

import numpy as np  # Keep for some specific operations that need to be on CPU
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
    mnist = fetch_openml(
        "mnist_784", version=1, as_frame=False, parser="auto", cache=True
    )
    X, y = mnist.data, mnist.target.astype(int)

    # Split into train and test (last 10k samples for test, rest for train)
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]

    # Normalize pixel values and convert to GPU arrays
    if normalize:
        X_train = cp.array(X_train.astype(np.float32) / 255.0)
        X_test = cp.array(X_test.astype(np.float32) / 255.0)
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


class Sigmoid:
    def _sigmoid(self, x):
        return 1 / (1 + cp.exp(-x))

    def forward(self, x):
        self.output = self._sigmoid(x)
        return self.output

    def backward(self, grad):
        # Derivative of sigmoid: sigmoid(x) * (1 - sigmoid(x))
        # where sigmoid(x) is the output from the forward pass
        return self.output * (1 - self.output) * grad
    
    def __call__(self, x):
        return self.forward(x)


class CrossEntropy:
    def __init__(self):
        self.softmax_output = None
        self.target = None
        self.batch_size = None

    def forward(self, logits, target):
        # Handle both single samples and batches
        if logits.ndim == 1:
            # Single sample case - reshape to batch of size 1
            logits = logits.reshape(1, -1)
            target = target.reshape(1, -1)
        
        # Apply softmax per sample (along axis=1)
        # Subtract max for numerical stability, then exponentiate
        exp_logits = cp.exp(logits - cp.max(logits, axis=1, keepdims=True))
        # Divide by sum of exponentiated logits per sample (along axis=1)
        self.softmax_output = exp_logits / cp.sum(exp_logits, axis=1, keepdims=True)
        
        self.target = target
        self.batch_size = logits.shape[0] # Store batch size
        
        # Compute cross entropy loss per sample, then average over the batch
        # Use a small epsilon for numerical stability with log(0)
        log_softmax = cp.log(self.softmax_output + 1e-15)
        # Only consider the log-probabilities of the true classes
        loss_per_sample = -cp.sum(target * log_softmax, axis=1) # Sum over classes for each sample

        # Return the average loss over the batch
        return cp.mean(loss_per_sample)

    def backward(self):
        grad = (self.softmax_output - self.target) / self.batch_size
        return grad

    def __call__(self, logits, target):
        return self.forward(logits, target)


class ReLU:
    def __init__(self):
        self.input = None
        
    def forward(self, x):
        self.input = x
        return cp.maximum(0, x)

    def backward(self, grad):
        return cp.where(self.input > 0, grad, 0)

    def __call__(self, x):
        return self.forward(x)
    

class LeakyReLU:
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.input = None
        
    def forward(self, x):
        self.input = x
        return cp.where(x > 0, x, self.alpha * x)
    
    def backward(self, grad):
        return cp.where(self.input > 0, grad, self.alpha * grad)
    
    def __call__(self, x):
        return self.forward(x)


class SGD:
    def __init__(self, params, criterion, lr=0.01, momentum=0.9):
        self.params = params
        self.criterion = criterion
        self.lr = lr
        self.momentum = momentum
        self.updates = [
            [cp.zeros_like(layer.W), cp.zeros_like(layer.b)] for layer in self.params
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
            [
                cp.zeros_like(layer.dendrite_W),
                cp.zeros_like(layer.dendrite_b),
                cp.zeros_like(layer.soma_W),
                cp.zeros_like(layer.soma_b),
            ]
            for layer in self.params
        ]

    def zero_grad(self):
        for layer in self.params:
            layer.dendrite_dW = 0.0
            layer.dendrite_db = 0.0
            layer.soma_dW = 0.0
            layer.soma_db = 0.0

    def step(self):
        for layer, update in zip(self.params, self.updates):
            # print(f"shape of db {layer.dendrite_db.shape} shape of b {layer.dendrite_b.shape}")
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
            if hasattr(layer, "W") or hasattr(layer, "soma_W"):
                params.append(layer)
        return params

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class LinearLayer:
    """A fully connected, feed forward layer"""

    def __init__(self, in_dim, out_dim):
        self.W = cp.random.randn(out_dim, in_dim) * cp.sqrt(
            2.0 / (in_dim)
        )  # He init
        self.b = cp.zeros(out_dim)
        self.dW = 0.0
        self.db = 0.0
        self.x = None

    def forward(self, x):
        # print(f"x: {x}, self.W {self.W}, self.b {self.b}")
        self.x = x
        return x @ self.W.T + self.b

    def backward(self, grad):
        # print(f"shape of incoming grad {grad} \n shape of W {self.W.shape}")
        # self.dW = cp.outer(grad, self.x)
        self.dW = grad.T @ self.x
        self.db = grad.sum(axis=0)
        grad = grad @ self.W
        return grad

    def num_params(self):
        return self.W.size + self.b.size

    def __call__(self, x):
        return self.forward(x)


class DendriticLayer:
    """A sparse dendritic layer, consiting of dendrites and somas"""

    def __init__(
        self, in_dim, out_dim, strategy="random", n_dendrite_inputs=16, n_dendrites=3
    ):
        assert strategy == "random", "Invalid strategy"

        n_neurons = out_dim  # the number of neurons determins the size of the output
        n_soma_connections = (n_dendrites * n_neurons) 
        # print(
        #     f"input dimension {in_dim}, n_neurons: {n_neurons}, n_soma_connections: {n_soma_connections}, n_dendrite_connections: {n_dendrite_connections}"
        # )

        self.dendrite_W = cp.random.randn(n_soma_connections, in_dim) * cp.sqrt(
            2.0 / (in_dim)
        )
        self.dendrite_b = cp.zeros((n_soma_connections))
        self.dendrite_dW = 0.0
        self.dendrite_db = 0.0

        self.dendrite_activation = LeakyReLU()

        self.soma_W = cp.random.randn(n_neurons, n_soma_connections) * cp.sqrt(
            2.0 / (n_soma_connections)
        )
        self.soma_b = cp.zeros(n_neurons)
        self.soma_dW = 0.0
        self.soma_db = 0.0

        self.soma_activation = LeakyReLU()

        # inputs to save for backprop
        self.dendrite_x = None
        self.soma_x = None

        # sample soma mask:
        # [[1, 1, 0, 0]
        #  [0, 0, 1, 1]]
        # number of 1 per row is n_dendrites, rest 0. every column only has 1 entry
        # number of rows equals n_neurons, number of columns eqais n_soma_connections
        # it is a step pattern, so the first n_dendrites entries of the first row are one.
        self.soma_mask = cp.zeros((n_neurons, n_soma_connections))
        for i in range(n_neurons):
            start_idx = i * n_dendrites
            end_idx = start_idx + n_dendrites
            self.soma_mask[i, start_idx:end_idx] = 1

        # mask out unneeded weights, thus making weights sparse
        self.soma_W = self.soma_W * self.soma_mask

        # sample dendrite mask
        # for each dendrite sample n_dendrite_inputs from the input array
        self.dendrite_mask = cp.zeros((n_soma_connections, in_dim))
        for i in range(n_soma_connections):
            if strategy == "random":
                # sample without replacement from possible input for a given dendrite from the whole input
                input_idx = cp.random.choice(
                    cp.arange(in_dim), size=n_dendrite_inputs, replace=False
                )
            self.dendrite_mask[i, input_idx] = 1

        # mask out unneeded weights, thus making weights sparse
        self.dendrite_W = self.dendrite_W * self.dendrite_mask

    def forward(self, x):
        # pass through dendrites
        # print(f"x dendrite: {x.shape}, self.dendrite_W shape {self.dendrite_W.shape}")
        self.dendrite_x = x
        x = self.dendrite_W @ x + self.dendrite_b
        x = self.dendrite_activation(x)

        # pass through soma
        # print(f"x soma: {x.shape}, self.soma_W shape {self.soma_W.shape}")
        self.soma_x = x
        x = self.soma_W @ x + self.soma_b
        return self.soma_activation(x)

    def backward(self, grad):
        # print(f"shape of incoming grad {grad} \n shape of W {self.W.shape}")
        grad = self.soma_activation.backward(grad)
        
        # soma back pass, multiply with mask to keep only valid gradients
        self.soma_dW = cp.outer(grad, self.soma_x) * self.soma_mask
        self.soma_db = grad
        soma_grad = self.soma_W.T @ grad 
        
        soma_grad = self.dendrite_activation.backward(soma_grad)

        # dendrite back pass
        self.dendrite_dW = cp.outer(soma_grad, self.dendrite_x) * self.dendrite_mask
        self.dendrite_db = soma_grad
        dendrite_grad = self.dendrite_W.T @ soma_grad
        return dendrite_grad
    
    def num_params(self):
        print(f"\nparameters: dendrite_mask: {cp.sum(self.dendrite_mask)}, dendrite_b: {self.dendrite_b.size}, soma_W: {cp.sum(self.soma_mask)}, soma_b: {self.soma_b.size}")
        return int(cp.sum(self.dendrite_mask) + self.dendrite_b.size + cp.sum(self.soma_mask) + self.soma_b.size)
 
    def __call__(self, x):
        return self.forward(x)


def create_batches(X, y, batch_size=128, shuffle=True, drop_last=True):
    n_samples = len(X)
    # shuffle data
    if shuffle:
        indices = cp.arange(n_samples)
        cp.random.shuffle(indices)
        X = X[indices]
        y = y[indices]
    
    for i in range(0, n_samples, batch_size):
        if drop_last and i + batch_size > n_samples:
            break
        X_batch = X[i:i+batch_size]
        y_batch = y[i:i+batch_size]
        yield X_batch, y_batch
    return X_batch, y_batch



def train(
    X_train,
    y_train,
    model,
    criterion,
    optimiser,
    n_epochs=2,
    batch_size=128,
):
    train_losses = []
    accuracy = []
    n_samples = len(X_train)
    for epoch in tqdm(range(n_epochs)):
        train_loss = 0.0
        correct_pred = 0.0
        for X, target in create_batches(X_train, y_train, batch_size, shuffle=True):
            # forward pass
            pred = model(X)
            loss = criterion(pred, target)
            train_loss += loss
            # if most likely prediction equals target add to correct predictions
            batch_correct = cp.sum(cp.argmax(pred, axis=1) == cp.argmax(target, axis=1))
            correct_pred += batch_correct

            # backward pass
            optimiser.zero_grad()
            grad = criterion.backward()
            model.backward(grad)
            optimiser.step()

            # print(f"y {target}, pred {pred}, loss {loss}")
        normalised_train_loss = train_loss / n_samples
        train_losses.append(float(normalised_train_loss))  # Convert to float for plotting
        epoch_accuracy = correct_pred / n_samples
        accuracy.append(float(epoch_accuracy))  # Convert to float for plotting
    return train_losses, accuracy


def evaluate(
    X_test,
    y_test,
    model,
    criterion,
    batch_size=128,
):
    n_samples = len(X_test)
    test_loss = 0.0
    correct_pred = 0.0
    for X, target in create_batches(X_test, y_test, batch_size, shuffle=False, drop_last=False):
        # forward pass
        pred = model(X)
        loss = criterion(pred, target)
        test_loss += loss
        # if most likely prediction eqauls target add to correct predictions
        batch_correct = cp.sum(cp.argmax(pred, axis=1) == cp.argmax(target, axis=1))
        correct_pred += batch_correct
    normalised_test_loss = test_loss / n_samples
    accuracy = correct_pred / n_samples
    return float(normalised_test_loss), float(accuracy)  # Convert to float for printing


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
    cp.random.seed(42)

    # config
    n_epochs = 10
    lr = 0.01
    batch_size = 128
    in_dim = 28 * 28  # MNIST dimension
    n_classes = 10

    # load data
    # subset_size=100
    X_train, y_train, X_test, y_test = load_mnist_data(subset_size=10000)

    # criterion = CrossEntropy()
    # model = Sequential([
    #     DendriticLayer(in_dim, n_classes, n_dendrite_inputs=16, n_dendrites=32)
    # ])
    # optimiser = DendriteSGD(model.params(), criterion, lr=lr, momentum=0.9)
    
    v_criterion = CrossEntropy()
    v_model = Sequential([
        LinearLayer(in_dim, n_classes),
        ReLU()
    ])
    v_optimiser = SGD(v_model.params(), v_criterion, lr=lr, momentum=0.9)

    # train model
    # train_losses, train_accuracy = train(
    #     X_train, y_train, model, criterion, optimiser, n_epochs, batch_size
    # )
    # # run model evaluation
    # test_loss, test_accuracy = evaluate(X_test, y_test, model, criterion)


    # train vani
    v_train_losses, v_train_accuracy = train(
        X_train, y_train, v_model, v_criterion, v_optimiser, n_epochs, batch_size
    )
    # run model evaluation
    v_test_loss, v_test_accuracy = evaluate(X_test, y_test, v_model, v_criterion)
    
    # # plot accuracy of vanilla model vs dendritic model
    plt.plot(v_train_accuracy, label="Vanilla")
    # plt.plot(train_accuracy, label="Dendritic")
    plt.title("Accuracy over epochs")
    plt.legend()
    plt.show()
    
    # # plot both models in comparison
    plt.plot(v_train_losses, label="Vanilla")
    # plt.plot(train_losses, label="Dendritic")
    plt.title("Loss over epochs")
    plt.legend()
    plt.show()
    
    # print(f"final train loss dendritic model {round(train_losses[-1], 4)} vs vanilla {round(v_train_losses[-1], 4)}")
    # print(f"final test loss dendritic model {round(test_loss, 4)} vs vanilla {round(v_test_loss, 4)}")
    # print(f"final train accuracy dendritic model {round(train_accuracy[-1], 4)} vs vanilla {round(v_train_accuracy[-1], 4)}")
    # print(f"final test accuracy dendritic model {round(test_accuracy, 4)} vs vanilla {round(v_test_accuracy, 4)}")

    # print(f"number of dendritic params: {model.params()[0].num_params()}")
    # print(f"number of vanilla params: {v_model.params()[0].num_params()}")

# if __name__ == "main":
#     main()


main()
