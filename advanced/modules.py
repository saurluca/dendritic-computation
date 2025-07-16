import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DendriticLayer(nn.Module):
    """
    A sparse dendritic layer consisting of dendrites and somas.

    This layer implements a biologically-inspired neural network component where:
    - Each neuron has multiple dendrites
    - Each dendrite receives sparse inputs from the input space
    - Dendrites process inputs and feed into soma
    - Optional synaptic resampling mimics biological synaptic plasticity

    Args:
        in_dim (int): Input dimension
        n_neurons (int): Number of neurons in the layer
        n_dendrite_inputs (int): Number of inputs each dendrite receives
        n_dendrites (int): Number of dendrites per neuron
        synaptic_resampling (bool): Whether to enable synaptic resampling
        percentage_resample (float): Percentage of connections to resample each time
        steps_to_resample (int): Number of backward passes before resampling
    """

    def __init__(
        self,
        in_dim,
        n_neurons,
        n_dendrite_inputs,
        n_dendrites,
        synaptic_resampling=True,
        percentage_resample=0.25,
        steps_to_resample=128,
    ):
        super(DendriticLayer, self).__init__()

        self.in_dim = in_dim
        self.n_dendrite_inputs = n_dendrite_inputs
        self.n_neurons = n_neurons
        self.n_dendrites = n_dendrites
        self.n_soma_connections = n_dendrites * n_neurons
        self.n_synaptic_connections = n_dendrite_inputs * n_dendrites * n_neurons

        # Synaptic resampling parameters
        self.synaptic_resampling = synaptic_resampling
        self.percentage_resample = percentage_resample
        self.steps_to_resample = steps_to_resample
        self.num_mask_updates = 1
        self.update_steps = 0

        # Dendrite weights and biases
        self.dendrite_W = nn.Parameter(
            torch.randn(self.n_soma_connections, in_dim) * np.sqrt(2.0 / in_dim)
        )
        self.dendrite_b = nn.Parameter(torch.zeros(self.n_soma_connections))

        # Soma weights and biases
        self.soma_W = nn.Parameter(
            torch.randn(n_neurons, self.n_soma_connections)
            * np.sqrt(2.0 / self.n_soma_connections)
        )
        self.soma_b = nn.Parameter(torch.zeros(n_neurons))

        # Activation functions
        self.dendrite_activation = nn.LeakyReLU(negative_slope=0.1)
        self.soma_activation = nn.LeakyReLU(negative_slope=0.1)

        # Create masks (these are not parameters, just buffers)
        self._create_masks()

    def _create_masks(self):
        """Create sparse connection masks for dendrites and soma"""
        device = self.dendrite_W.device

        # Soma mask: step pattern where each neuron connects to its own dendrites
        soma_mask = torch.zeros(
            (self.n_neurons, self.n_soma_connections), device=device
        )
        for i in range(self.n_neurons):
            start_idx = i * self.n_dendrites
            end_idx = start_idx + self.n_dendrites
            soma_mask[i, start_idx:end_idx] = 1
        self.register_buffer("soma_mask", soma_mask)

        # Dendrite mask: each dendrite samples random inputs
        dendrite_mask = torch.zeros(
            (self.n_soma_connections, self.in_dim), device=device
        )
        for i in range(self.n_soma_connections):
            # Sample without replacement
            input_idx = torch.randperm(self.in_dim)[: self.n_dendrite_inputs]
            dendrite_mask[i, input_idx] = 1
        self.register_buffer("dendrite_mask", dendrite_mask)

        # Apply masks to weights
        with torch.no_grad():
            self.dendrite_W.data *= self.dendrite_mask
            self.soma_W.data *= self.soma_mask

    def forward(self, x):
        """Forward pass through dendrites then soma"""
        # Dendrite forward pass
        dendrite_out = F.linear(
            x, self.dendrite_W * self.dendrite_mask, self.dendrite_b
        )
        dendrite_out = self.dendrite_activation(dendrite_out)

        # Soma forward pass
        soma_out = F.linear(dendrite_out, self.soma_W * self.soma_mask, self.soma_b)
        soma_out = self.soma_activation(soma_out)

        return soma_out

    def resample_dendrites(self):
        """
        Implements synaptic resampling by replacing weak dendritic connections with new random ones.

        This method mimics synaptic plasticity in biological neurons, where weak or unused
        synaptic connections are pruned and replaced with new connections to explore different
        input patterns. The resampling helps prevent overfitting and maintains exploration
        capabilities during training.

        Algorithm Overview:
        1. **Connection Removal**: Identifies the weakest connections (lowest weight magnitude)
           for removal based on `self.percentage_resample`
        2. **One-shot Resampling**: Randomly samples new input connections for each removed connection
        3. **Conflict Detection**: Checks for conflicts with existing connections and duplicate
           assignments within the same dendrite
        4. **Successful Swaps**: Applies only valid swaps that don't create conflicts
        5. **Verification**: Ensures the dendritic structure integrity is maintained

        The method operates efficiently by:
        - Using vectorized operations for all dendrites simultaneously
        - Implementing one-shot resampling rather than iterative attempts
        - Only applying successful swaps to avoid invalid states
        - Maintaining sparsity through the dendrite_mask

        Side Effects:
            - Updates self.dendrite_mask to reflect new connection patterns
            - Reinitializes weights for new connections using He initialization
            - Zeros out weights and gradients for removed connections
            - Increments self.num_mask_updates counter

        Raises:
            AssertionError: If resampling violates dendritic structure constraints
                - Each dendrite must maintain exactly n_dendrite_inputs connections
                - Total active connections must equal n_synaptic_connections

        Returns:
            None: Method modifies the layer's state in-place

        Note:
            - Called automatically during backward pass if synaptic_resampling=True
            - Early returns if percentage_resample results in 0 connections to remove
            - Biologically inspired by synaptic pruning and neuroplasticity
        """
        # Calculate number of connections to remove per dendrite
        n_to_remove_per_dendrite = int(
            self.n_dendrite_inputs * self.percentage_resample
        )
        if n_to_remove_per_dendrite == 0:
            return

        num_dendrites = self.dendrite_mask.shape[0]

        with torch.no_grad():
            # --- Part 1: Connection Removal ---
            # Use weight magnitude as metric, mask inactive connections with infinity
            metric = torch.abs(self.dendrite_W)
            metric[self.dendrite_mask == 0] = float("inf")

            # Sort by magnitude and select weakest connections
            sorted_indices = torch.argsort(metric, dim=1)
            cols_to_remove = sorted_indices[:, :n_to_remove_per_dendrite]

            # Create row indices for flattening
            rows_to_remove = torch.arange(num_dendrites, device=self.dendrite_W.device)[
                :, None
            ]
            removed_dendrite_indices = rows_to_remove.repeat(
                1, n_to_remove_per_dendrite
            ).flatten()
            removed_input_indices = cols_to_remove.flatten()

            # --- Part 2: One-shot Resampling Attempt ---
            n_connections_to_resample = removed_dendrite_indices.size(0)
            newly_selected_input_indices = torch.randint(
                0,
                self.in_dim,
                size=(n_connections_to_resample,),
                device=self.dendrite_W.device,
                dtype=torch.long,
            )

            # --- Part 3: Conflict Detection ---
            # Check conflicts with existing connections
            conflict_with_existing = (
                self.dendrite_mask[
                    removed_dendrite_indices, newly_selected_input_indices
                ]
                == 1
            )

            # Check for duplicates within the same dendrite
            proposed_flat_indices = (
                removed_dendrite_indices * self.in_dim + newly_selected_input_indices
            )

            # Count occurrences (using bincount)
            counts = torch.bincount(
                proposed_flat_indices, minlength=num_dendrites * self.in_dim
            )
            is_duplicate_flat = counts[proposed_flat_indices] > 1

            # Flag problematic connections
            is_problematic = conflict_with_existing | is_duplicate_flat
            is_successful = ~is_problematic

            # --- Part 4: Apply Successful Swaps ---
            if not is_successful.any():
                return

            dendrites_to_swap = removed_dendrite_indices[is_successful]
            old_inputs_to_remove = removed_input_indices[is_successful]
            new_inputs_to_add = newly_selected_input_indices[is_successful]

            # Update mask
            self.dendrite_mask[dendrites_to_swap, old_inputs_to_remove] = 0
            self.dendrite_mask[dendrites_to_swap, new_inputs_to_add] = 1

            # Initialize new weights with He initialization
            self.dendrite_W[dendrites_to_swap, new_inputs_to_add] = torch.randn(
                dendrites_to_swap.size(0), device=self.dendrite_W.device
            ) * np.sqrt(2.0 / self.in_dim)

            # Apply mask to ensure only active connections have non-zero weights
            self.dendrite_W.data *= self.dendrite_mask

            self.num_mask_updates += 1

            # --- Part 5: Verification ---
            n_dendritic_mask_connections = torch.sum(self.dendrite_mask, dim=1)
            assert torch.all(n_dendritic_mask_connections == self.n_dendrite_inputs), (
                f"Resampling failed: not all dendrites have {self.n_dendrite_inputs} connections per dendrite."
            )
            assert (
                torch.sum(self.dendrite_mask).item() == self.n_synaptic_connections
            ), (
                f"Resampling failed: not all dendrites have {self.n_synaptic_connections} connections in total."
            )

    def num_params(self):
        """Return the number of parameters in the layer"""
        dendrite_params = torch.sum(self.dendrite_mask).item() + self.dendrite_b.numel()
        soma_params = torch.sum(self.soma_mask).item() + self.soma_b.numel()
        return int(dendrite_params + soma_params)

    def extra_repr(self):
        """String representation of the layer"""
        return (
            f"in_dim={self.in_dim}, n_neurons={self.n_neurons}, "
            f"n_dendrites={self.n_dendrites}, n_dendrite_inputs={self.n_dendrite_inputs}, "
            f"synaptic_resampling={self.synaptic_resampling}"
        )


class DendriticLayerBackwardHook:
    """
    Custom backward hook to implement synaptic resampling after backward pass.
    This is needed because PyTorch doesn't allow modifying parameters during backward.
    """

    def __init__(self, layer):
        self.layer = layer

    def __call__(self, module, grad_input, grad_output):
        if self.layer.training and self.layer.synaptic_resampling:
            self.layer.update_steps += 1
            if self.layer.update_steps >= self.layer.steps_to_resample:
                self.layer.update_steps = 0
                # Schedule resampling for after backward pass
                self.layer._schedule_resampling = True


# Register the hook when layer is created
def register_dendritic_hooks(layer):
    """Register backward hooks for synaptic resampling"""
    if isinstance(layer, DendriticLayer):
        hook = DendriticLayerBackwardHook(layer)
        layer.register_backward_hook(hook)
        layer._schedule_resampling = False
