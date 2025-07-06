# %%
try:
    import cupy as cp

    # Test if CuPy can actually access CUDA and random number generator
    cp.cuda.Device(0).compute_capability
    cp.random.seed(1)  # Test if random number generator works
    print("Using CuPy (GPU acceleration)")
except (ImportError, Exception) as e:
    import numpy as cp

    print(f"CuPy not available or CUDA error ({type(e).__name__}), using NumPy (CPU)")

from matplotlib import pyplot as plt
from modules import Adam, CrossEntropy, LeakyReLU, Sequential
from utils import load_mnist_data, load_cifar10_data
from training import compare_models, plot_dendritic_weights, plot_dendritic_weights_single_image, print_network_entropy, train, train_one_model, plot_dendritic_weights_full_model

class MinimalDendriticLayer:
    """A sparse dendritic layer, consiting of dendrites and somas"""

    def __init__(
        self,
        in_dim,
        n_dendrites,
        n_dendrite_inputs=16,
        synaptic_resampling=True,
        percentage_resample=0.005,
        scaling_resampling_percentage=False,
        steps_to_resample=100,
        stop_after_n_mask_updates=100,
        dynamic_steps_size=False,
        use_bias=False,
    ):
        self.n_dendrites = n_dendrites
        # dynamicly resample
        self.synaptic_resampling = synaptic_resampling
        self.percentage_resample = percentage_resample
        self.steps_to_resample = steps_to_resample
        self.scaling_resampling_percentage = scaling_resampling_percentage
        self.dynamic_steps_size = dynamic_steps_size
        self.stop_after_n_mask_updates = stop_after_n_mask_updates
        # to keep track of resampling
        self.num_mask_updates = 1
        self.update_steps = 0

        self.in_dim = in_dim
        self.n_dendrite_inputs = n_dendrite_inputs
        self.dendrite_W = cp.random.randn(n_dendrites, in_dim) * cp.sqrt(
            2.0 / (in_dim)
        )  # He init, for ReLU
        self.dendrite_b = cp.zeros((n_dendrites))  # Always create bias array for Adam optimizer compatibility
        self.dendrite_dW = 0.0
        self.dendrite_db = 0.0

        # Add dummy soma attributes for Adam optimizer compatibility
        self.soma_W = cp.zeros((n_dendrites, n_dendrites))  # dummy
        self.soma_b = cp.zeros(n_dendrites)  # dummy
        self.soma_dW = 0.0
        self.soma_db = 0.0

        # sample dendrite mask
        # for each dendrite sample n_dendrite_inputs from the input array
        self.dendrite_mask = cp.zeros((n_dendrites, in_dim))
        for i in range(n_dendrites):
            # sample without replacement from possible input for a given dendrite from the whole input
            input_idx = cp.random.choice(
                cp.arange(in_dim), size=n_dendrite_inputs, replace=False
            )
            self.dendrite_mask[i, input_idx] = 1

        # mask out unneeded weights, thus making weights sparse
        self.dendrite_W = self.dendrite_W * self.dendrite_mask

    def forward(self, x):
        # dendrites forward pass
        self.dendrite_x = x
        x = x @ self.dendrite_W.T
        # if self.dendrite_b is not None:
            # x += self.dendrite_b
        
        # dummy soma forward pass for consistency with main.py
        self.soma_x = x
        return x

    def backward(self, grad):
        # dendrite back pass
        self.dendrite_dW = grad.T @ self.dendrite_x * self.dendrite_mask
        self.dendrite_db = grad.sum(axis=0)
        dendrite_grad = grad @ self.dendrite_W
        
        # dummy soma gradients for Adam optimizer compatibility
        self.soma_dW = cp.zeros_like(self.soma_W)
        self.soma_db = cp.zeros_like(self.soma_b)

        if self.synaptic_resampling:
            self.update_steps += 1

            
            # if enough steps have passed, resample
            if self.dynamic_steps_size:
                resample_bool = self.update_steps >= 100 + 5 * self.num_mask_updates
                # resample_bool = self.update_steps >= cp.exp((self.num_mask_updates + 20) / 10 ) + 20
            else:
                # resample_bool = self.update_steps >= 20 + 10 * self.num_mask_updates
                resample_bool = self.update_steps >= self.steps_to_resample 
                # and self.num_mask_updates < 500
                # if self.update_steps == 500:
                    # print("LAST UPDATE")
            if resample_bool:
                # reset step counter
                self.update_steps = 0
                self.resample_dendrites()
                
                
        return dendrite_grad

    def resample_dendrites(self):
        # --- Part 1: Connection Removal ---
        if self.scaling_resampling_percentage:
            resampling_percentage = 1 / (1 + 0.1 * self.num_mask_updates)
        else:
            resampling_percentage = self.percentage_resample

        n_to_remove_per_dendrite = int(self.n_dendrite_inputs * resampling_percentage)
        if n_to_remove_per_dendrite == 0:
            print("No dendrites to remove")
            return

        num_dendrites = self.dendrite_mask.shape[0]

        # For magnitude, we remove the smallest. Set inactive connections to infinity so they are not picked.
        metric = cp.abs(self.dendrite_W)
        metric[self.dendrite_mask == 0] = cp.inf
        sorted_indices = cp.argsort(metric, axis=1)
        cols_to_remove = sorted_indices[:, :n_to_remove_per_dendrite]

        # Create corresponding row indices and flatten for the swap logic
        rows_to_remove = cp.arange(num_dendrites)[:, cp.newaxis]
        removed_dendrite_indices = rows_to_remove.repeat(n_to_remove_per_dendrite, axis=1).flatten()
        removed_input_indices = cols_to_remove.flatten()

        n_connections_to_remove = removed_dendrite_indices.size
        
        # --- Part 2: One-shot Resampling Attempt ---
        num_inputs_per_dendrite = self.dendrite_x.shape[1]

        newly_selected_input_indices = cp.random.randint(
            0, num_inputs_per_dendrite, size=n_connections_to_remove, dtype=int
        )

        # --- Part 3: Conflict Detection ---
        conflict_with_existing = self.dendrite_mask[removed_dendrite_indices, newly_selected_input_indices] == 1
        
        num_dendrites = self.dendrite_mask.shape[0]
        proposed_flat_indices = removed_dendrite_indices * num_inputs_per_dendrite + newly_selected_input_indices
        counts = cp.bincount(proposed_flat_indices.astype(int), minlength=num_dendrites * num_inputs_per_dendrite)
        is_duplicate_flat = counts[proposed_flat_indices.astype(int)] > 1
        
        is_problematic = conflict_with_existing | is_duplicate_flat
        is_successful = ~is_problematic

        # --- Part 4: Apply Successful Swaps ---
        dendrites_to_swap = removed_dendrite_indices[is_successful]
        old_inputs_to_remove = removed_input_indices[is_successful]
        new_inputs_to_add = newly_selected_input_indices[is_successful]

        if dendrites_to_swap.size > 0:
            self.dendrite_mask[dendrites_to_swap, old_inputs_to_remove] = 0
            self.dendrite_mask[dendrites_to_swap, new_inputs_to_add] = 1

            self.dendrite_W[dendrites_to_swap, new_inputs_to_add] = (
                cp.random.randn(dendrites_to_swap.shape[0]) * cp.sqrt(2.0 / self.in_dim)
            )
        
        # print(f"num of dendrite successful swaps: {dendrites_to_swap.size}")
        self.num_mask_updates += 1
        
        # --- Part 5: Verification ---
        connections_per_dendrite = cp.sum(self.dendrite_mask, axis=1)
        assert cp.all(connections_per_dendrite == self.n_dendrite_inputs), \
            f"Resampling failed: not all dendrites have {self.n_dendrite_inputs} connections."

    def num_params(self):
        print(
            f"\nparameters: dendrite_mask: {cp.sum(self.dendrite_mask)}, dendrite_b: {self.dendrite_b.size}"
        )
        return int(
            cp.sum(self.dendrite_mask)
            + self.dendrite_b.size
        )

    def __call__(self, x):
        return self.forward(x)


def main():
    cp.random.seed(197023)

    # data config
    dataset = "mnist"  # "mnist", "fashion-mnist", "cifar10"

    # config
    n_epochs = 15 # 15 MNIST, 20 Fashion-MNIST
    lr = 0.003  # 0.003
    weight_decay = 0.01 #0.001
    batch_size = 256
    grad_clip = 0.1

    in_dim = 28 * 28  # Image dimensions (28x28 MNIST, 32x32x3 CIFAR-10)

    # dendriticmodel config
    n_dendrite_inputs = 2
    n_dendrites = 10  # 10 classes for MNIST
    X_train, y_train, X_test, y_test = load_mnist_data(dataset=dataset)

    criterion = CrossEntropy()
    model = Sequential(
        [
            MinimalDendriticLayer(
                in_dim,
                n_dendrites,
                n_dendrite_inputs=n_dendrite_inputs,
                synaptic_resampling=True,
                percentage_resample=0.5,
                steps_to_resample=5,
                stop_after_n_mask_updates=10000,
            ),
        ]
    )
    optimiser = Adam(model.params(), criterion, lr=lr, weight_decay=weight_decay, grad_clip=grad_clip)

    print(f"model params: {model.params()}")

    train_one_model(
        X_train,
        y_train,
        X_test,
        y_test,
        model,
        criterion,
        optimiser,
        n_epochs=n_epochs,
        batch_size=batch_size,
    )
    
    # plot_dendritic_weights_full_model(model, X_test[0])
    # for i in range(10):
        # plot_dendritic_weights_single_image(model, X_test[0], neuron_idx=i)

if __name__ == "__main__":
    main()

