# %%
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class Sigmoid:
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, x):
        return self._sigmoid(x)

    def backward(self, grad):
        return self._sigmoid(grad) * (1 - self._sigmoid(grad)) * grad

    def __call__(self, x):
        return self.forward(x)


class ReLU:
    def forward(self, x):
        return np.maximum(0, x)

    def backward(self, grad):
        return np.where(grad > 0, 1, 0) * grad

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
        n_soma_connections = n_dendrites * n_neurons # number of possible connection from dendrites to somas
        n_dendrite_connections = n_soma_connections * in_dim # number of possible connections from input to dendrites
        print(f"input dimension {in_dim}, n_neurons: {n_neurons}, n_soma_connections: {n_soma_connections}, n_dendrite_connections: {n_dendrite_connections}")
    
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
                input_idx = np.random.choice(np.arange(in_dim), size=n_dendrite_inputs, replace=False) 
            self.dendrite_mask[i, input_idx] = 1
        
        # mask out unneeded weights, thus making weights sparse
        self.dendrite_W = self.dendrite_W * self.dendrite_mask
        
        
    def forward(self, x):
        # pass through dendrites
        print(f"x dendrite: {x.shape}, self.dendrite_W shape {self.dendrite_W.shape}")
        self.x_dendrite = x
        x = self.dendrite_W @ x + self.dendrite_b
        x = self.dendrite_activation(x)
    
        # pass through soma
        print(f"x soma: {x.shape}, self.soma_W shape {self.soma_W.shape}")
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
    train_data,
    model,
    criterion,
    optimiser,
    n_epochs=10,
):
    train_losses = []
    outputs = []
    for epoch in tqdm(range(n_epochs)):
        train_loss = 0.0
        outputs_epoch = []
        for X, target in train_data:
            # forward pass
            pred = model(X)
            loss = criterion(pred, target)
            train_loss += loss
            outputs_epoch.append(pred)

            # backward pass
            optimiser.zero_grad()
            grad = criterion.backward()
            model.backward(grad)
            optimiser.step()

            # print(f"y {target}, pred {pred}, loss {loss}")
        train_losses.append(train_loss)
        outputs.append(outputs_epoch)
    return train_losses, outputs


def plot_loss(losses):
    plt.plot(losses)
    plt.title("Loss over epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()


def plot_predictions(outputs, targets):
    plt.scatter(targets, outputs)
    plt.title("Predictions vs Targets")
    plt.xlabel("Targets")
    plt.ylabel("Predictions")
    plt.show()


def main():
    np.random.seed(42)

    # config
    n_epochs = 10
    lr = 0.1

    # setup dummy data
    # n_samples = 200
    # inputs = np.random.uniform(-1, 1, size=(n_samples, 3))
    # true_w = np.array([1.5, -2.0, 0.5])
    # true_b = -0.1
    # targets = Sigmoid()._sigmoid(inputs @ true_w + true_b)
    # train_data = list(zip(inputs, targets))

    # model = Sequential(
    #     [
    #         LinearLayer(in_dim=3, out_dim=1),
    #         Sigmoid(),
    #         # LinearLayer(in_dim=2, out_dim=1),
    #         # Sigmoid()
    #     ]
    # )
    
    n_samples = 32 * 32
    inputs = np.random.uniform(-1, 1, size=(n_samples))
    # true_w = np.array([1.5, -2.0, 0.5])
    # true_b = -0.1
    # targets = Sigmo"""  """id()._sigmoid(inputs @ true_w + true_b)

    model = DendriticLayer(32 * 32, 3)
    
    pred = model(inputs)
    print(pred)
    
    

    # criterion = MSE()
    # optimiser = SGD(model.params(), criterion, lr=lr, momentum=0.9)

    # train_losses, outputs = train(train_data, model, criterion, optimiser, n_epochs)
    # plot_loss(train_losses)
    # plot_predictions(outputs[-1], targets)

    # print(f"final loss {train_losses[-1]}")

    # # print out final model params
    # final_params = model.params()[0]
    # print(
    #     f"true W {true_w} model w {final_params.W} \n true b {true_b}, model b {final_params.b}"
    # )


if __name__ == "main":
    main()


main()
