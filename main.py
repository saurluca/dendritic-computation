# %%
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Callable, List, Tuple, Optional, Union


class Sigmoid:
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self._sigmoid(x)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return self._sigmoid(grad) * (1 - self._sigmoid(grad)) * grad

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


class ReLU:
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return np.where(grad > 0, 1, 0) * grad

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


class MSE:
    def __init__(self) -> None:
        self.pred: Optional[np.ndarray] = None
        self.target: Optional[np.ndarray] = None

    def forward(self, pred: np.ndarray, target: np.ndarray) -> float:
        self.pred, self.target = pred, target
        return np.mean((pred - target) ** 2)

    def backward(self) -> np.ndarray:
        return np.mean(0.5 * (self.pred - self.target)).reshape(1)

    def __call__(self, pred: np.ndarray, target: np.ndarray) -> float:
        return self.forward(pred, target)


class SGD:
    def __init__(
        self, params: list, criterion: Callable, lr: float = 0.01, momentum: float = 0.9
    ):
        self.params = params
        self.criterion = criterion
        self.lr = lr
        self.momentum = momentum
        self.updates = [
            [np.zeros_like(layer.W), np.zeros_like(layer.b)] for layer in self.params
        ]

    def zero_grad(self) -> None:
        for layer in self.params:
            layer.dW = 0.0
            layer.db = 0.0

    def step(self) -> None:
        for layer, update in zip(self.params, self.updates):
            update[0] = self.lr * layer.dW + self.momentum * update[0]
            update[1] = self.lr * layer.db + self.momentum * update[1]
            layer.W -= update[0]
            layer.b -= update[1]

    def __call__(self) -> None:
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

    def __call__(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer(x)
        return x


class LinearLayer:
    def __init__(self, in_dim: int, out_dim: int) -> None:
        self.W: np.ndarray = np.random.randn(out_dim, in_dim) * np.sqrt(
            2.0 / (in_dim + out_dim)
        )  # xavier init
        self.b: np.ndarray = np.zeros(out_dim)
        self.dW: Union[float, np.ndarray] = 0.0
        self.db: Union[float, np.ndarray] = 0.0
        self.x: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        # print(f"x: {x}, self.W {self.W}, self.b {self.b}")
        self.x = x
        return self.W @ x + self.b

    def backward(self, grad: np.ndarray) -> np.ndarray:
        # print(f"shape of incoming grad {grad} \n shape of W {self.W.shape}")
        self.dW = np.outer(grad, self.x)
        self.db = grad
        grad = self.W.T @ grad
        return grad

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


def train(
    train_data: List[Tuple[np.ndarray, np.ndarray]],
    model: Sequential,
    criterion: MSE,
    optimiser: SGD,
    n_epochs: int = 10,
) -> Tuple[List[float], List[List[np.ndarray]]]:
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


def main() -> None:
    np.random.seed(42)

    # config
    n_epochs = 10
    lr = 0.1

    # setup dummy data
    n_samples = 200
    inputs = np.random.uniform(-1, 1, size=(n_samples, 3))
    true_w = np.array([1.5, -2.0, 0.5])
    true_b = -0.1
    targets = Sigmoid()._sigmoid(inputs @ true_w + true_b)
    train_data = list(zip(inputs, targets))

    model = Sequential(
        [
            LinearLayer(in_dim=3, out_dim=1),
            Sigmoid(),
            # LinearLayer(in_dim=2, out_dim=1),
            # Sigmoid()
        ]
    )
    criterion = MSE()
    optimiser = SGD(model.params(), criterion, lr=lr, momentum=0.9)

    train_losses, outputs = train(train_data, model, criterion, optimiser, n_epochs)
    plot_loss(train_losses)
    plot_predictions(outputs[-1], targets)

    print(f"final loss {train_losses[-1]}")

    # print out final model params
    final_params = model.params()[0]
    print(
        f"true W {true_w} model w {final_params.W} \n true b {true_b}, model b {final_params.b}"
    )


if __name__ == "main":
    main()


main()
