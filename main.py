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

from modules import Adam, CrossEntropy, LeakyReLU, Sequential
from utils import load_mnist_data
from training import compare_models
import sys


class LinearLayer:
    """A fully connected, feed forward layer"""

    def __init__(self, in_dim, out_dim):
        self.W = cp.random.randn(out_dim, in_dim) * cp.sqrt(
            2.0 / (in_dim)
        )  # He init, for ReLU
        self.b = cp.zeros(out_dim)
        self.dW = 0.0
        self.db = 0.0
        self.x = None

    def forward(self, x):
        self.x = x
        return x @ self.W.T + self.b

    def backward(self, grad):
        self.dW = grad.T @ self.x
        self.db = grad.sum(axis=0)
        grad = grad @ self.W
        return grad

    def num_params(self):
        return self.W.size + self.b.size
    
    def var_params(self):
        return cp.var(self.W) + cp.var(self.b)

    def __call__(self, x):
        return self.forward(x)


class DendriticLayer:
    """A sparse dendritic layer, consiting of dendrites and somas"""

    def __init__(
        self,
        in_dim,
        n_neurons,
        strategy="random",
        n_dendrite_inputs=16,
        n_dendrites=4,
        synaptic_resampling=True,
        percentage_resample=0.005,
        prob_of_resampling=0.05,
        resampling_criterion="gradient",
        scaling_resampling_percentage=False,
        resample_with_permutation=False,
        steps_to_resample=100,
    ):
        assert strategy in ("random", "local-receptive-fields", "fully-connected"), (
            "Invalid strategy"
        )
        assert resampling_criterion in ("gradient", "magnitude"), (
            "Invalid resampling_criterion"
        )
        assert not synaptic_resampling or strategy == "random", (
            "synaptic_resampling is only supported for random strategy"
        )

        n_soma_connections = n_dendrites * n_neurons

        # dynamicly resample
        self.synaptic_resampling = synaptic_resampling
        self.percentage_resample = percentage_resample
        self.resampling_criterion = resampling_criterion  # gradient, magnitude
        self.prob_of_resampling = prob_of_resampling
        self.steps_to_resample = steps_to_resample
        self.scaling_resampling_percentage = scaling_resampling_percentage
        self.resample_with_permutation = resample_with_permutation

        # to keep track of resampling
        self.num_mask_updates = 1
        self.update_steps = 0

        self.in_dim = in_dim
        self.n_dendrite_inputs = n_dendrite_inputs
        self.dendrite_W = cp.random.randn(n_soma_connections, in_dim) * cp.sqrt(
            2.0 / (in_dim)
        )  # He init, for ReLU
        self.dendrite_b = cp.zeros((n_soma_connections))
        self.dendrite_dW = 0.0
        self.dendrite_db = 0.0
        self.dendrite_activation = LeakyReLU()

        self.soma_W = cp.random.randn(n_neurons, n_soma_connections) * cp.sqrt(
            2.0 / (n_soma_connections)
        )  # He init, for ReLU
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
            elif strategy == "local-receptive-fields":
                # According to the description: "16 inputs are chosen from the 4 × 4 neighborhood"
                input_idx = self.create_local_receptive_field_mask()
            elif strategy == "fully-connected":
                # sample all inputs for a given dendrite
                input_idx = cp.arange(in_dim)
            self.dendrite_mask[i, input_idx] = 1

        # mask out unneeded weights, thus making weights sparse
        self.dendrite_W = self.dendrite_W * self.dendrite_mask

    
    def create_local_receptive_field_mask(self):
        assert self.n_dendrite_inputs == 16, (
                    "local-receptive-fields strategy requires exactly 16 dendrite inputs for a 4x4 neighborhood"
                )

        image_size = int(cp.sqrt(self.in_dim))  # 28 for MNIST

        # Choose center pixel such that 4x4 neighborhood fits within image bounds
        # For 4x4 grid centered at (center_row, center_col), we need:
        # - Grid spans from (center_row-1, center_col-1) to (center_row+2, center_col+2)
        # - So center_row must be in [1, image_size-3] and center_col must be in [1, image_size-3]
        # This ensures the full 4x4 grid is within [0, image_size-1] bounds
        min_center = 1
        max_center = image_size - 3  # 25 for 28x28 image

        center_row = cp.random.randint(min_center, max_center + 1)
        center_col = cp.random.randint(min_center, max_center + 1)

        # Create 4x4 neighborhood around center pixel
        # The 4x4 grid will be positioned such that center is at position (1,1) in the grid
        input_indices = []
        for dr in range(-1, 3):  # -1, 0, 1, 2 (4 rows)
            for dc in range(-1, 3):  # -1, 0, 1, 2 (4 cols)
                row = center_row + dr
                col = center_col + dc
                idx = row * image_size + col
                input_indices.append(idx)

        input_idx = cp.array(input_indices)
        return input_idx
    
    def forward(self, x):
        # dendrites forward pass
        self.dendrite_x = x
        x = x @ self.dendrite_W.T + self.dendrite_b
        x = self.dendrite_activation(x)

        # soma forward pass
        self.soma_x = x
        x = x @ self.soma_W.T + self.soma_b
        return self.soma_activation(x)

    def backward(self, grad):
        grad = self.soma_activation.backward(grad)

        # soma back pass, multiply with mask to keep only valid gradients
        self.soma_dW = grad.T @ self.soma_x * self.soma_mask
        self.soma_db = grad.sum(axis=0)
        soma_grad = grad @ self.soma_W

        soma_grad = self.dendrite_activation.backward(soma_grad)

        # dendrite back pass
        self.dendrite_dW = soma_grad.T @ self.dendrite_x * self.dendrite_mask
        self.dendrite_db = soma_grad.sum(axis=0)
        dendrite_grad = soma_grad @ self.dendrite_W

        if self.synaptic_resampling:
            self.update_steps += 1    
            
            # if enough steps have passed, resample
            if self.update_steps >= self.steps_to_resample:
                # reset step counter
                self.update_steps = 0
                self.resample_dendrites_new()

            # resample based on probability
            # if not cp.random.random() < self.prob_of_resampling / (self.num_mask_updates * 2):
                # self.num_mask_updates += 1
                # return dendrite_grad

        return dendrite_grad

    def resample_dendrites(self):
        # Calculate total number of connections to remove across entire network
        if self.scaling_resampling_percentage:
            resampling_percentage = self.percentage_resample ** (1 / self.num_mask_updates)
        else:
            resampling_percentage = self.percentage_resample
        total_active_connections = int(cp.sum(self.dendrite_mask))
        n_connections_to_remove = int(total_active_connections * resampling_percentage)

        if n_connections_to_remove == 0:
            print("no connections to remove, skipping resampling")
            return
        
        # print(f"resampling {resampling_percentage*100}% of dendritic inputs")

        # Find and remove top connections based on gradient or magnitude
        if self.resampling_criterion == "gradient":
            active_gradients = cp.abs(self.dendrite_dW) * self.dendrite_mask
            metric = active_gradients.flatten()
            # remove the largest gradient connections
            flat_indices = cp.argsort(metric)[-n_connections_to_remove:]
        elif self.resampling_criterion == "magnitude":
            metric = cp.abs(self.dendrite_W).flatten()
            # remove the smallest magnitude connections
            flat_indices = cp.argsort(metric)[:n_connections_to_remove]

        dendrite_indices, input_indices = cp.unravel_index(
            flat_indices, self.dendrite_dW.shape
        )
        self.dendrite_mask[dendrite_indices, input_indices] = 0

        # Count connections lost per dendrite and resample
        unique_dendrites, counts = cp.unique(dendrite_indices, return_counts=True)

        # Pre-generate random candidates for resampling
        # print(f"shape of dendrite_x: {self.dendrite_x.shape}")
        n_inputs = self.dendrite_x.shape[1]
        max_resample = int(cp.max(counts))
        # print(f"max_resample: {max_resample}, shape of counts: {counts.shape}")
        # TODO implement random draw with replacement, careful not in same dendrite, othwerise duplicates
        random_pool = cp.random.permutation(n_inputs)[: min(n_inputs, max_resample * 4)]

        for dendrite_idx, n_to_resample in zip(unique_dendrites, counts):
            available_mask = self.dendrite_mask[dendrite_idx] == 0
            available_candidates = random_pool[available_mask[random_pool]]

            # if not enough candidates, skip
            if len(available_candidates) < n_to_resample:
                print("Not enoguh candidates avialbe, continuing")
                continue

            new_inputs = cp.random.permutation(available_candidates)[:n_to_resample]
            # new_inputs_all.append(new_inputs)
            self.dendrite_mask[dendrite_idx, new_inputs] = 1
            # Reinitialize weights for newly added connections using He initialization
            self.dendrite_W[dendrite_idx, new_inputs] = cp.random.randn(
                len(new_inputs)
            ) * cp.sqrt(2.0 / self.in_dim)
            
            
    def resample_dendrites_new(self):
        # Initial assertion to ensure correctness before resampling
        connections_per_dendrite = cp.sum(self.dendrite_mask, axis=1)
        assert cp.all(connections_per_dendrite == self.n_dendrite_inputs), \
            f"Resampling failed: before resampling not all dendrites have {self.n_dendrite_inputs} connections."

        # Calculate resampling percentage, potentially scaled
        if self.scaling_resampling_percentage:
            resampling_percentage = self.percentage_resample ** (1 / self.num_mask_updates)
        else:
            resampling_percentage = self.percentage_resample

        # --- Part 1: Balanced Connection Removal from Each Dendrite ---
        n_to_remove_per_dendrite = int(self.n_dendrite_inputs * resampling_percentage)
        if n_to_remove_per_dendrite == 0:
            return  # Nothing to remove
        
        # print(f"n_to_remove_per_dendrite: {n_to_remove_per_dendrite}")

        num_dendrites = self.dendrite_mask.shape[0]

        if self.resampling_criterion == "gradient":
            # For gradient, we remove the largest. Inactive connections are 0, so they won't be picked.
            metric = cp.abs(self.dendrite_dW) * self.dendrite_mask
            sorted_indices = cp.argsort(metric, axis=1)
            cols_to_remove = sorted_indices[:, -n_to_remove_per_dendrite:]
        elif self.resampling_criterion == "magnitude":
            # For magnitude, we remove the smallest. Set inactive connections to infinity so they are not picked.
            metric = cp.abs(self.dendrite_W)
            metric[self.dendrite_mask == 0] = cp.inf
            sorted_indices = cp.argsort(metric, axis=1)
            cols_to_remove = sorted_indices[:, :n_to_remove_per_dendrite]
        else:
            raise ValueError(f"Unsupported resampling_criterion: {self.resampling_criterion}")

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

        # print(f"shape of dendrite_mask: {self.dendrite_mask.shape}")
        # print(f"num of dendrite successful swaps: {cp.count_nonzero(new_inputs_to_add)}")
        # print(f"total num of dendrite connecitons: {cp.count_nonzero(self.dendrite_mask)}")
        # print(f"num of dendrite failed swaps: {cp.count_nonzero(is_problematic)}")
        
        # # --- Part 5: Verification ---
        connections_per_dendrite = cp.sum(self.dendrite_mask, axis=1)
        assert cp.all(connections_per_dendrite == self.n_dendrite_inputs), \
            f"Resampling failed: not all dendrites have {self.n_dendrite_inputs} connections."

        # raise Exception("Stop here")

    def num_params(self):
        print(
            f"\nparameters: dendrite_mask: {cp.sum(self.dendrite_mask)}, dendrite_b: {self.dendrite_b.size}, soma_W: {cp.sum(self.soma_mask)}, soma_b: {self.soma_b.size}"
        )
        return int(
            cp.sum(self.dendrite_mask)
            + self.dendrite_b.size
            + cp.sum(self.soma_mask)
            + self.soma_b.size
        )
        
    def var_params(self):
        return cp.var(self.dendrite_W) + cp.var(self.dendrite_b) + cp.var(self.soma_W) + cp.var(self.soma_b)

    def __call__(self, x):
        return self.forward(x)


def main():
    # for repoducability
    cp.random.seed(32)

    # config
    n_epochs = 20  # 15 MNIST, 20 Fashion-MNIST
    lr = 0.001  # 0.07 - SGD
    v_lr = 0.001  # 0.015 - SGD
    weight_decay = 0.0 #0.001
    batch_size = 128
    in_dim = 28 * 28  # Image dimensions (28x28 for both MNIST and Fashion-MNIST)
    n_classes = 10

    # dendriticmodel config
    n_dendrite_inputs = 16
    n_dendrites = 16
    n_neurons = 10
    strategy = "random"  # ["random", "local-receptive-fields", "fully-connected"]

    # vanilla model config
    # n_vanilla_neurons_1 = 12
    # n_vanilla_neurons_2 = 12

    # data config
    dataset = "fashion-mnist"  # Choose between "mnist" or "fashion-mnist"
    subset_size = None

    X_train, y_train, X_test, y_test = load_mnist_data(
        dataset=dataset, subset_size=subset_size
    )

    criterion = CrossEntropy()
    model = Sequential(
        [
            DendriticLayer(
                in_dim,
                n_neurons,
                n_dendrite_inputs=n_dendrite_inputs,
                n_dendrites=n_dendrites,
                strategy=strategy,
                synaptic_resampling=True,
                percentage_resample=0.2,
                # prob_of_resampling=0.5,
                steps_to_resample=50,
                resampling_criterion="magnitude",
                scaling_resampling_percentage=True,
                resample_with_permutation=False,
            ),
            LeakyReLU(),
            LinearLayer(n_neurons, n_classes),
        ]
    )
    optimiser = Adam(model.params(), criterion, lr=lr, weight_decay=weight_decay)

    v_criterion = CrossEntropy()
    # v_model = Sequential(
    #     [
    #         LinearLayer(in_dim, n_vanilla_neurons_1),
    #         LeakyReLU(),
    #         LinearLayer(n_vanilla_neurons_1, n_vanilla_neurons_2),
    #         LeakyReLU(),
    #         LinearLayer(n_vanilla_neurons_2, n_classes),
    #     ]
    # )
    v_model = Sequential(
        [
            DendriticLayer(
                in_dim,
                n_neurons,
                n_dendrite_inputs=n_dendrite_inputs,
                n_dendrites=n_dendrites,
                strategy=strategy,
                synaptic_resampling=False,
            ),
            LeakyReLU(),
            LinearLayer(n_neurons, n_classes),
        ]
    )
    v_optimiser = Adam(v_model.params(), v_criterion, lr=v_lr, weight_decay=weight_decay)

    print(f"number of model_1 params: {model.num_params()}")
    print(f"number of model_2 params: {v_model.num_params()}")

    # raise Exception("Stop here")

    compare_models(
        model,
        v_model,
        optimiser,
        v_optimiser,
        X_train,
        y_train,
        X_test,
        y_test,
        criterion,
        n_epochs=n_epochs,
        batch_size=batch_size,
        model_name_1="Dendritic",
        model_name_2="Vanilla",
        track_variance=True,
    )


main()
