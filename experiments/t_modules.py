import torch
import torch.nn as nn


class DendriticLayer(nn.Module):
    """A sparse dendritic layer consisting of dendrites only (no soma)"""

    def __init__(
        self,
        in_dim,
        n_neurons,
        n_dendrite_inputs,
        synaptic_resampling=True,
        percentage_resample=0.25,
        steps_to_resample=128,
        dendrite_bias=True,
    ):
        super(DendriticLayer, self).__init__()

        self.in_dim = in_dim
        self.n_dendrite_inputs = n_dendrite_inputs
        self.n_neurons = n_neurons
        self.n_synaptic_connections = n_dendrite_inputs * n_neurons

        # Synaptic resampling parameters
        self.synaptic_resampling = synaptic_resampling
        self.percentage_resample = percentage_resample
        self.steps_to_resample = steps_to_resample
        self.num_mask_updates = 1
        self.update_steps = 0

        # Dendrite layer (input -> output directly)
        self.dendrite_linear = nn.Linear(in_dim, n_neurons, bias=dendrite_bias)
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
        # Shape: (n_neurons, in_dim)
        dendrite_mask = torch.zeros(self.n_neurons, self.in_dim)
        for i in range(self.n_neurons):
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
            weights = self.dendrite_linear.weight.data  # Shape: (n_neurons, in_dim)
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
            n_neurons = embed_dim  # Output dimension matches
            n_dendrite_inputs = 32  # Fixed as requested

            print(
                f"Dendritic MLP config: {n_dendrite_inputs} inputs, {n_neurons} neurons"
            )

            self.mlp = nn.Sequential(
                DendriticLayer(
                    in_dim=embed_dim,
                    n_neurons=n_neurons,
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

