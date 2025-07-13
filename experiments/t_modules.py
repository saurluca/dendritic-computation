import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt

class DendriticLayer(nn.Module):
    """A sparse dendritic layer consisting of dendrites only (no soma)"""

    def __init__(
        self,
        in_dim,
        output_dim,
        n_dendrite_inputs,
        synaptic_resampling=True,
        percentage_resample=0.25,
        steps_to_resample=128,
        dendrite_bias=True,
    ):
        super(DendriticLayer, self).__init__()

        self.in_dim = in_dim
        self.n_dendrite_inputs = n_dendrite_inputs
        self.output_dim = output_dim
        self.n_synaptic_connections = n_dendrite_inputs * output_dim

        # Synaptic resampling parameters
        self.synaptic_resampling = synaptic_resampling
        self.percentage_resample = percentage_resample
        self.steps_to_resample = steps_to_resample
        self.num_mask_updates = 1
        self.update_steps = 0

        # Dendrite layer (input -> output directly)
        self.dendrite_linear = nn.Linear(in_dim, output_dim, bias=dendrite_bias)
        self.dendrite_activation = nn.LeakyReLU(0.1)

        # Initialize weights with He initialization
        nn.init.kaiming_normal_(
            self.dendrite_linear.weight, mode="fan_in", nonlinearity="leaky_relu"
        )
        if dendrite_bias:
            nn.init.zeros_(self.dendrite_linear.bias)

        # Create masks
        self._create_masks()

        # Apply masks to weights
        self._apply_masks()

        # Register backward hook for synaptic resampling
        if self.synaptic_resampling:
            self.register_backward_hook(self._backward_hook)

    def _create_masks(self):
        """Create sparse connectivity masks for dendrites"""

        # Dendrite mask: each neuron connects to n_dendrite_inputs random inputs
        # Shape: (output_dim, in_dim)
        dendrite_mask = torch.zeros(self.output_dim, self.in_dim)
        for i in range(self.output_dim):
            # Sample without replacement from possible inputs
            input_idx = torch.randperm(self.in_dim)[: self.n_dendrite_inputs]
            dendrite_mask[i, input_idx] = 1

        # Register as buffer so it automatically moves with the model
        self.register_buffer("dendrite_mask", dendrite_mask)

    def _apply_masks(self):
        """Apply masks to weight matrices"""
        with torch.no_grad():
            # Apply dendrite mask (transpose because nn.Linear uses transposed weight matrix)
            self.dendrite_linear.weight.data *= self.dendrite_mask

    def forward(self, x):
        # Automatically flatten input if it's not already flattened
        if x.dim() > 2:
            x = x.view(x.size(0), -1)  # Flatten to (batch_size, features)

        # Dendrite forward pass (direct output)
        x = self.dendrite_linear(x)
        x = self.dendrite_activation(x)
        return x

    def _backward_hook(self, module, grad_input, grad_output):
        """Hook called during backward pass for synaptic resampling"""
        self.update_steps += 1

        if self.update_steps >= self.steps_to_resample:
            self.update_steps = 0
            self._resample_dendrites()

    def _resample_dendrites(self):
        """Implement synaptic resampling by replacing weak dendritic connections"""
        # Calculate number of connections to remove per neuron
        n_to_remove_per_neuron = int(self.n_dendrite_inputs * self.percentage_resample)
        if n_to_remove_per_neuron == 0:
            return

        with torch.no_grad():
            # Get weight magnitudes (masked weights only)
            weights = self.dendrite_linear.weight.data  # Shape: (output_dim, in_dim)
            masked_weights = weights * self.dendrite_mask
            metric = torch.abs(masked_weights)

            # Set inactive connections to infinity so they are not picked
            metric[self.dendrite_mask == 0] = float("inf")

            # Sort by magnitude and get indices of weakest connections
            sorted_indices = torch.argsort(metric, dim=1)
            cols_to_remove = sorted_indices[:, :n_to_remove_per_neuron]

            # Create row indices for removal
            num_neurons = self.dendrite_mask.shape[0]
            mask_device = self.dendrite_mask.device  # Use device from mask buffer
            rows_to_remove = torch.arange(num_neurons, device=mask_device)[:, None]
            rows_to_remove = rows_to_remove.repeat(1, n_to_remove_per_neuron).flatten()
            cols_to_remove = cols_to_remove.flatten()

            # Sample new connections
            n_connections_to_resample = len(cols_to_remove)
            newly_selected_cols = torch.randint(
                0, self.in_dim, (n_connections_to_resample,), device=mask_device
            )

            # Check for conflicts with existing connections
            conflict_with_existing = (
                self.dendrite_mask[rows_to_remove, newly_selected_cols] == 1
            )

            # Check for duplicates within the same neuron
            proposed_flat_indices = rows_to_remove * self.in_dim + newly_selected_cols
            unique_indices, counts = torch.unique(
                proposed_flat_indices, return_counts=True
            )
            duplicate_mask = torch.zeros_like(proposed_flat_indices, dtype=torch.bool)
            for idx in unique_indices[counts > 1]:
                duplicate_mask[proposed_flat_indices == idx] = True

            # Only apply successful swaps (no conflicts or duplicates)
            is_successful = ~(conflict_with_existing | duplicate_mask)

            if is_successful.sum() == 0:
                return

            # Apply successful swaps
            successful_rows = rows_to_remove[is_successful]
            old_cols = cols_to_remove[is_successful]
            new_cols = newly_selected_cols[is_successful]

            # Update mask
            self.dendrite_mask[successful_rows, old_cols] = 0
            self.dendrite_mask[successful_rows, new_cols] = 1

            # Reinitialize new weights
            new_weights = torch.randn(
                len(successful_rows), device=mask_device
            ) * torch.sqrt(torch.tensor(2.0 / self.in_dim, device=mask_device))
            weights[successful_rows, new_cols] = new_weights

            # Apply mask to ensure only active connections have weights
            weights *= self.dendrite_mask

            self.num_mask_updates += 1

    def num_params(self):
        """Return number of logically active parameters"""
        active_dendrite_params = self.dendrite_mask.sum().item()
        dendrite_bias_params = (
            self.dendrite_linear.bias.numel()
            if self.dendrite_linear.bias is not None
            else 0
        )

        total = active_dendrite_params + dendrite_bias_params
        print(
            f"Active parameters: dendrite_W: {active_dendrite_params}, dendrite_b: {dendrite_bias_params}"
        )
        return int(total)


class PatchEmbedding(nn.Module):
    """Converts image patches into embeddings"""

    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        # x shape: (batch_size, channels, height, width)
        x = self.proj(x)  # (batch_size, embed_dim, n_patches_h, n_patches_w)
        x = x.flatten(2)  # (batch_size, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (batch_size, n_patches, embed_dim)
        return x


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism"""

    def __init__(self, embed_dim, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads

        assert embed_dim % n_heads == 0, "embed_dim must be divisible by n_heads"

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.scale = self.head_dim**-0.5

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape

        # Generate Q, K, V
        qkv = self.qkv(x)  # (batch_size, seq_len, embed_dim * 3)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch_size, n_heads, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention computation
        attn = (
            q @ k.transpose(-2, -1)
        ) * self.scale  # (batch_size, n_heads, seq_len, seq_len)
        attn = torch.softmax(attn, dim=-1)

        # Apply attention to values
        out = attn @ v  # (batch_size, n_heads, seq_len, head_dim)
        out = out.transpose(1, 2)  # (batch_size, seq_len, n_heads, head_dim)
        out = out.reshape(
            batch_size, seq_len, embed_dim
        )  # (batch_size, seq_len, embed_dim)

        return self.proj(out)


class TransformerBlock(nn.Module):
    """Transformer encoder block with attention and MLP (optionally dendritic)"""

    def __init__(
        self, embed_dim, n_heads, mlp_ratio=4.0, dropout=0.1, use_dendritic=False
    ):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, n_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.use_dendritic = use_dendritic

        mlp_hidden_dim = int(embed_dim * mlp_ratio)

        if use_dendritic:
            # Simplified dendritic layer configuration
            output_dim = embed_dim  # Output dimension matches
            n_dendrite_inputs = 32  # Fixed as requested

            self.mlp = nn.Sequential(
                DendriticLayer(
                    in_dim=embed_dim,
                    output_dim=output_dim,
                    n_dendrite_inputs=n_dendrite_inputs,
                    synaptic_resampling=True,
                    percentage_resample=0.15,
                    steps_to_resample=64,
                ),
                # nn.Dropout(dropout)
            )
        else:
            # Standard feedforward MLP
            self.mlp = nn.Sequential(
                nn.Linear(embed_dim, mlp_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(mlp_hidden_dim, embed_dim),
                nn.Dropout(dropout),
            )

    def forward(self, x):
        # Attention with residual connection
        x = x + self.attn(self.norm1(x))
        # MLP with residual connection
        mlp_input = self.norm2(x)
        if self.use_dendritic:
            # For dendritic layer, we need to flatten and reshape
            batch_size, seq_len, embed_dim = mlp_input.shape
            mlp_input_flat = mlp_input.view(
                -1, embed_dim
            )  # (batch_size * seq_len, embed_dim)
            mlp_output_flat = self.mlp(mlp_input_flat)
            mlp_output = mlp_output_flat.view(batch_size, seq_len, embed_dim)
        else:
            mlp_output = self.mlp(mlp_input)
        x = x + mlp_output
        return x


class VisionTransformer(nn.Module):
    """Basic Vision Transformer implementation"""

    def __init__(
        self,
        img_size,
        patch_size,
        in_channels,
        n_classes,
        embed_dim=192,
        depth=6,
        n_heads=8,
        mlp_ratio=4.0,
        dropout=0.1,
        use_dendritic=False,
    ):
        super(VisionTransformer, self).__init__()

        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        n_patches = self.patch_embed.n_patches
        self.use_dendritic = use_dendritic

        # Class token and positional embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(embed_dim, n_heads, mlp_ratio, dropout, use_dendritic)
                for _ in range(depth)
            ]
        )

        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, n_classes)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        # Use normal initialization since trunc_normal_ is not available
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        batch_size = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)  # (batch_size, n_patches, embed_dim)

        # Add class token
        cls_tokens = self.cls_token.expand(
            batch_size, -1, -1
        )  # (batch_size, 1, embed_dim)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch_size, n_patches + 1, embed_dim)

        # Add positional embeddings
        x = x + self.pos_embed
        x = self.dropout(x)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        # Classification
        x = self.norm(x)
        cls_token_final = x[:, 0]  # Use class token for classification
        return self.head(cls_token_final)

    def num_params(self):
        if self.use_dendritic:
            # For dendritic ViT, calculate logical active parameters
            total_pytorch = sum(p.numel() for p in self.parameters())

            # Calculate active parameters from dendritic layers
            total_active = 0
            for block in self.blocks:
                if hasattr(block.mlp[0], "num_params"):  # DendriticLayer
                    total_active += block.mlp[0].num_params()
                else:
                    total_active += sum(p.numel() for p in block.mlp.parameters())

            # Add non-dendritic parameters
            non_dendritic_params = (
                sum(p.numel() for p in self.patch_embed.parameters())
                + self.cls_token.numel()
                + self.pos_embed.numel()
                + sum(p.numel() for p in self.norm.parameters())
                + sum(p.numel() for p in self.head.parameters())
            )

            # Add attention layers
            for block in self.blocks:
                non_dendritic_params += (
                    sum(p.numel() for p in block.norm1.parameters())
                    + sum(p.numel() for p in block.attn.parameters())
                    + sum(p.numel() for p in block.norm2.parameters())
                )

            total_active += non_dendritic_params

            print(
                f"Dendritic ViT - PyTorch allocated: {total_pytorch}, Logical active: {total_active}"
            )
            return total_active
        else:
            total = sum(p.numel() for p in self.parameters())
            print(f"Standard ViT parameters: {total}")
            return total


class DropLinear(nn.Module):
    """
    Fully connected layer, that drops out neurons over time.
    Either based on smallest weight magnitude. Or by generating a random mask,
    and testing if model performance improves, if not it will be kept.
    Also possible to undo last mask change if performance drops.
    """

    def __init__(
        self,
        in_dim,
        out_dim,
        steps_to_resample=16,
        target_params=100,
        training_steps=1000,
        stop_percentage=0.9,
        drop_distribution="exponential",
        drop=True,
    ):
        super(DropLinear, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.steps_to_resample = steps_to_resample
        self.target_params = target_params
        self.training_steps = training_steps
        self.stop_percentage = stop_percentage
        self.drop_distribution = drop_distribution
        self.drop = drop

        # use he initialization
        self.weight = nn.Parameter(
            torch.randn(out_dim, in_dim) * torch.sqrt(torch.tensor(2.0 / in_dim))
        )
        self.bias = nn.Parameter(torch.zeros(out_dim))

        # mask for the linear layer
        self.register_buffer("mask", torch.ones(in_dim, dtype=torch.bool))
        # TODO, later make possible to undo last mask change
        # self.register_buffer("prev_mask", torch.ones(in_dim, dtype=torch.bool))
        self.update_steps = 0
        self.mask_update_count = 0

        # Calculate and store the dropping schedule
        if self.drop:
            self.dropping_schedule = self.calculate_dropping_schedule()

    def update_mask(self):
        # Check if we still have schedule steps remaining
        if self.mask_update_count >= len(self.dropping_schedule):
            return

        # Get number of parameters to drop from schedule
        n_to_drop = self.dropping_schedule[self.mask_update_count]

        # Calculate number of currently active weights
        active_indices = torch.where(self.mask)[0]
        n_active = len(active_indices)

        # Ensure we don't drop more than available
        n_to_drop = min(n_to_drop, n_active)

        if n_to_drop <= 0 or n_active <= 0:
            self.mask_update_count += 1
            self.update_steps = 0
            return

        # Only consider weights that are currently active
        weights = self.weight.data
        active_weight_magnitudes = torch.abs(weights).sum(dim=0)[active_indices]
        sorted_indices = torch.argsort(active_weight_magnitudes)
        # Get indices of smallest active weights
        smallest_active_indices = active_indices[sorted_indices[:n_to_drop]]
        # Set mask to 0 for smallest active weights (cumulative)
        self.mask[smallest_active_indices] = 0

        self.mask_update_count += 1
        self.update_steps = 0

    def calculate_dropping_schedule(self, verbose=True):
        """
        Based on self.target_params, calculate a list of parameter counts to drop at each step.
        Returns a list with total_update_steps entries.
        """
        last_step = int(self.training_steps * self.stop_percentage)
        total_update_steps = int(last_step / self.steps_to_resample)

        if total_update_steps == 0:
            return []

        starting_params = self.num_active_params()
        total_params_to_drop = starting_params - self.target_params

        if total_params_to_drop <= 0:
            return [0] * total_update_steps

        # Convert to input dimension drops (since we drop input features)
        total_inputs_to_drop = total_params_to_drop // self.out_dim

        schedule = []

        if self.drop_distribution == "linear":
            # Linear distribution: same number each step
            inputs_per_step = total_inputs_to_drop // total_update_steps
            remainder = total_inputs_to_drop % total_update_steps

            for i in range(total_update_steps):
                # Distribute remainder across first few steps
                extra = 1 if i < remainder else 0
                schedule.append(inputs_per_step + extra)

        elif self.drop_distribution == "exponential":
            # Exponential distribution: more early, less later
            # Use exponential decay: drop_rate = base_rate * exp(-decay_rate * step)
            decay_rate = 0.5  # Controls how fast the decay happens
            weights = []

            for i in range(total_update_steps):
                weight = math.exp(-decay_rate * i / total_update_steps)
                weights.append(weight)

            total_weight = sum(weights)

            for i in range(total_update_steps):
                normalized_weight = weights[i] / total_weight
                inputs_to_drop = int(normalized_weight * total_inputs_to_drop)
                schedule.append(inputs_to_drop)
        else:
            raise ValueError(f"Invalid drop distribution: {self.drop_distribution}")

        # Ensure we don't exceed total inputs to drop
        total_scheduled = sum(schedule)
        if total_scheduled > total_inputs_to_drop:
            # Reduce from the end
            excess = total_scheduled - total_inputs_to_drop
            for i in range(len(schedule) - 1, -1, -1):
                if excess <= 0:
                    break
                reduction = min(schedule[i], excess)
                schedule[i] -= reduction
                excess -= reduction
        elif total_scheduled < total_inputs_to_drop:
            # Add to the end
            remaining = total_inputs_to_drop - total_scheduled
            for i in range(len(schedule)):
                if remaining <= 0:
                    break
                schedule[i] += 1
                remaining -= 1

        if verbose:
            # plot dropping schedule
            plt.title(f"Dropping schedule for {self.drop_distribution} distribution")
            plt.xlabel("Step")
            plt.ylabel("Number of parameters to drop")
            plt.plot(schedule)
            plt.show()

        return schedule

    def forward(self, x):
        self.update_steps += 1
        if self.training and self.drop and self.update_steps >= self.steps_to_resample:
            self.update_mask()

        # Apply mask to weights (mask is applied to input dimension)
        if self.drop:
            masked_weight = self.weight * self.mask.unsqueeze(
                0
            )  # Broadcast mask to weight shape
        else:
            masked_weight = self.weight
        return torch.nn.functional.linear(x, masked_weight, self.bias)

    def num_params(self):
        return self.in_dim * self.out_dim + self.out_dim

    def num_active_params(self):
        return self.mask.sum() * self.out_dim + self.out_dim
