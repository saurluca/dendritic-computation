# %%
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import math

# Set device for data loading
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class DendriticLayer(nn.Module):
    """A sparse dendritic layer consisting of dendrites and somas"""
    
    def __init__(
        self,
        in_dim,
        n_neurons,
        n_dendrite_inputs,
        n_dendrites,
        synaptic_resampling=True,
        percentage_resample=0.25,
        steps_to_resample=128,
        dendrite_bias=True,
        soma_bias=True,
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
        
        # Dendrite layer (input -> dendrites)
        self.dendrite_linear = nn.Linear(in_dim, self.n_soma_connections, bias=dendrite_bias)
        self.dendrite_activation = nn.LeakyReLU(0.1)
        
        # Soma layer (dendrites -> output)
        self.soma_linear = nn.Linear(self.n_soma_connections, n_neurons, bias=soma_bias)
        self.soma_activation = nn.LeakyReLU(0.1)
        
        # Initialize weights with He initialization
        nn.init.kaiming_normal_(self.dendrite_linear.weight, mode='fan_in', nonlinearity='leaky_relu')
        if dendrite_bias:
            nn.init.zeros_(self.dendrite_linear.bias)
        nn.init.kaiming_normal_(self.soma_linear.weight, mode='fan_in', nonlinearity='leaky_relu')
        if soma_bias:
            nn.init.zeros_(self.soma_linear.bias)
        
        # Create masks
        self._create_masks()
        
        # Apply masks to weights
        self._apply_masks()
        
        # Register backward hook for synaptic resampling
        if self.synaptic_resampling:
            self.register_backward_hook(self._backward_hook)
    
    def _create_masks(self):
        """Create sparse connectivity masks for dendrites and soma"""
        
        # Create masks on CPU first, then register as buffers so they move with the model
        # Soma mask: step pattern where each neuron connects to its specific dendrites
        # Shape: (n_neurons, n_soma_connections)
        soma_mask = torch.zeros(self.n_neurons, self.n_soma_connections)
        for i in range(self.n_neurons):
            start_idx = i * self.n_dendrites
            end_idx = start_idx + self.n_dendrites
            soma_mask[i, start_idx:end_idx] = 1
        
        # Dendrite mask: each dendrite connects to n_dendrite_inputs random inputs
        # Shape: (n_soma_connections, in_dim)
        dendrite_mask = torch.zeros(self.n_soma_connections, self.in_dim)
        for i in range(self.n_soma_connections):
            # Sample without replacement from possible inputs
            input_idx = torch.randperm(self.in_dim)[:self.n_dendrite_inputs]
            dendrite_mask[i, input_idx] = 1
        
        # Register as buffers so they automatically move with the model
        self.register_buffer('soma_mask', soma_mask)
        self.register_buffer('dendrite_mask', dendrite_mask)
    
    def _apply_masks(self):
        """Apply masks to weight matrices"""
        with torch.no_grad():
            # Apply dendrite mask (transpose because nn.Linear uses transposed weight matrix)
            self.dendrite_linear.weight.data *= self.dendrite_mask
            # Apply soma mask (transpose because nn.Linear uses transposed weight matrix)
            self.soma_linear.weight.data *= self.soma_mask
    
    def forward(self, x):
        # Dendrite forward pass
        x = self.dendrite_linear(x)
        x = self.dendrite_activation(x)
        
        # Soma forward pass
        x = self.soma_linear(x)
        x = self.soma_activation(x)
        
        return x
    
    def _backward_hook(self, module, grad_input, grad_output):
        """Hook called during backward pass for synaptic resampling"""
        self.update_steps += 1
        
        if self.update_steps >= self.steps_to_resample:
            self.update_steps = 0
            self._resample_dendrites()
    
    def _resample_dendrites(self):
        """Implement synaptic resampling by replacing weak dendritic connections"""
        # print("resampling dendrites")
        # Calculate number of connections to remove per dendrite
        n_to_remove_per_dendrite = int(self.n_dendrite_inputs * self.percentage_resample)
        if n_to_remove_per_dendrite == 0:
            return
        
        with torch.no_grad():
            # Get weight magnitudes (masked weights only)
            weights = self.dendrite_linear.weight.data  # Shape: (n_soma_connections, in_dim)
            masked_weights = weights * self.dendrite_mask
            metric = torch.abs(masked_weights)
            
            # Set inactive connections to infinity so they are not picked
            metric[self.dendrite_mask == 0] = float('inf')
            
            # Sort by magnitude and get indices of weakest connections
            sorted_indices = torch.argsort(metric, dim=1)
            cols_to_remove = sorted_indices[:, :n_to_remove_per_dendrite]
            
            # Create row indices for removal
            num_dendrites = self.dendrite_mask.shape[0]
            mask_device = self.dendrite_mask.device  # Use device from mask buffer
            rows_to_remove = torch.arange(num_dendrites, device=mask_device)[:, None]
            rows_to_remove = rows_to_remove.repeat(1, n_to_remove_per_dendrite).flatten()
            cols_to_remove = cols_to_remove.flatten()
            
            # Sample new connections
            n_connections_to_resample = len(cols_to_remove)
            newly_selected_cols = torch.randint(0, self.in_dim, (n_connections_to_resample,), device=mask_device)
            
            # Check for conflicts with existing connections
            conflict_with_existing = self.dendrite_mask[rows_to_remove, newly_selected_cols] == 1
            
            # Check for duplicates within the same dendrite
            proposed_flat_indices = rows_to_remove * self.in_dim + newly_selected_cols
            unique_indices, counts = torch.unique(proposed_flat_indices, return_counts=True)
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
            new_weights = torch.randn(len(successful_rows), device=mask_device) * torch.sqrt(torch.tensor(2.0 / self.in_dim, device=mask_device))
            weights[successful_rows, new_cols] = new_weights
            
            # Apply mask to ensure only active connections have weights
            weights *= self.dendrite_mask
            
            self.num_mask_updates += 1
    
    def num_params(self):
        """Return number of logically active parameters (like notebook.py)"""
        active_dendrite_params = self.dendrite_mask.sum().item()
        dendrite_bias_params = self.dendrite_linear.bias.numel() if self.dendrite_linear.bias is not None else 0
        active_soma_params = self.soma_mask.sum().item()
        soma_bias_params = self.soma_linear.bias.numel() if self.soma_linear.bias is not None else 0
        
        total = active_dendrite_params + dendrite_bias_params + active_soma_params + soma_bias_params
        print(f"Active parameters: dendrite_W: {active_dendrite_params}, dendrite_b: {dendrite_bias_params}, "
              f"soma_W: {active_soma_params}, soma_b: {soma_bias_params}")
        return int(total)


class DendriticNet(nn.Module):
    """Complete neural network with dendritic layer"""
    
    def __init__(self, in_dim, n_neurons, n_dendrite_inputs, n_dendrites, n_classes, output_bias=True, **kwargs):
        super(DendriticNet, self).__init__()
        
        self.dendritic_layer = DendriticLayer(
            in_dim, n_neurons, n_dendrite_inputs, n_dendrites, **kwargs
        )
        self.output_layer = nn.Linear(n_neurons, n_classes, bias=output_bias)
        
        # Initialize output layer
        nn.init.kaiming_normal_(self.output_layer.weight, mode='fan_in', nonlinearity='leaky_relu')
        if output_bias:
            nn.init.zeros_(self.output_layer.bias)
    
    def forward(self, x):
        x = self.dendritic_layer(x)
        x = self.output_layer(x)
        return x
    
    def num_params(self):
        dendritic_params = self.dendritic_layer.num_params()
        output_params = sum(p.numel() for p in self.output_layer.parameters())
        total = dendritic_params + output_params
        print(f"Total parameters: {total} (Dendritic: {dendritic_params}, Output: {output_params})")
        return total


class PatchEmbedding(nn.Module):
    """Converts image patches into embeddings"""
    
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
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
        self.scale = self.head_dim ** -0.5
        
    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x)  # (batch_size, seq_len, embed_dim * 3)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch_size, n_heads, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention computation
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (batch_size, n_heads, seq_len, seq_len)
        attn = torch.softmax(attn, dim=-1)
        
        # Apply attention to values
        out = attn @ v  # (batch_size, n_heads, seq_len, head_dim)
        out = out.transpose(1, 2)  # (batch_size, seq_len, n_heads, head_dim)
        out = out.reshape(batch_size, seq_len, embed_dim)  # (batch_size, seq_len, embed_dim)
        
        return self.proj(out)


class TransformerBlock(nn.Module):
    """Transformer encoder block with attention and MLP (optionally dendritic)"""
    
    def __init__(self, embed_dim, n_heads, mlp_ratio=4.0, dropout=0.1, use_dendritic=False):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, n_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.use_dendritic = use_dendritic
        
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        
        if use_dendritic:
            # Calculate dendritic parameters to match FF layer parameter count
            # FF params: embed_dim * mlp_hidden_dim * 2 + mlp_hidden_dim + embed_dim  
            ff_params = embed_dim * mlp_hidden_dim * 2 + mlp_hidden_dim + embed_dim
            
            # CORRECTED: DendriticLayer logical active parameter count (like notebook.py):
            # Active dendrite weights: n_dendrite_inputs * n_dendrites * n_neurons
            # Dendrite biases: n_dendrites * n_neurons  
            # Active soma weights: n_dendrites * n_neurons (each neuron connects to n_dendrites)
            # Soma biases: n_neurons
            #
            # Total logical active: (n_dendrite_inputs * n_dendrites * n_neurons) + 
            #                      (n_dendrites * n_neurons) + 
            #                      (n_dendrites * n_neurons) + 
            #                      n_neurons
            #                    = n_dendrites * n_neurons * (n_dendrite_inputs + 1 + 1) + n_neurons
            #                    = n_dendrites * n_neurons * (n_dendrite_inputs + 2) + n_neurons
            
            n_neurons = embed_dim  # Output dimension matches
            n_dendrite_inputs = 32  # Fixed as requested
            
            # Solve: n_dendrites * n_neurons * (n_dendrite_inputs + 2) + n_neurons = ff_params
            # n_dendrites * n_neurons * (n_dendrite_inputs + 2) = ff_params - n_neurons
            # n_dendrites = (ff_params - n_neurons) / (n_neurons * (n_dendrite_inputs + 2))
            n_dendrites = max(1, (ff_params - n_neurons) // (n_neurons * (n_dendrite_inputs + 2)))
            
            # Calculate actual logical active parameters for verification
            actual_active_params = n_dendrites * n_neurons * (n_dendrite_inputs + 2) + n_neurons
            
            print(f"Dendritic MLP config: {n_dendrite_inputs} inputs, {n_dendrites} dendrites, {n_neurons} neurons")
            print(f"Target FF params: {ff_params}, Actual active params: {actual_active_params}")
            print(f"Active parameter difference: {actual_active_params - ff_params}")
            
            self.mlp = nn.Sequential(
                DendriticLayer(
                    in_dim=embed_dim,
                    n_neurons=n_neurons,
                    n_dendrite_inputs=n_dendrite_inputs,
                    n_dendrites=n_dendrites,
                    synaptic_resampling=True,  # No resampling for ViT
                    percentage_resample=0.15,
                    steps_to_resample=64
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
                nn.Dropout(dropout)
            )
        
    def forward(self, x):
        # Attention with residual connection
        x = x + self.attn(self.norm1(x))
        # MLP with residual connection  
        mlp_input = self.norm2(x)
        if self.use_dendritic:
            # For dendritic layer, we need to flatten and reshape
            batch_size, seq_len, embed_dim = mlp_input.shape
            mlp_input_flat = mlp_input.view(-1, embed_dim)  # (batch_size * seq_len, embed_dim)
            mlp_output_flat = self.mlp(mlp_input_flat)
            mlp_output = mlp_output_flat.view(batch_size, seq_len, embed_dim)
        else:
            mlp_output = self.mlp(mlp_input)
        x = x + mlp_output
        return x


class VisionTransformer(nn.Module):
    """Basic Vision Transformer implementation"""
    
    def __init__(self, img_size, patch_size, in_channels, n_classes, embed_dim=192, depth=6, n_heads=8, mlp_ratio=4.0, dropout=0.1, use_dendritic=False):
        super(VisionTransformer, self).__init__()
        
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        n_patches = self.patch_embed.n_patches
        self.use_dendritic = use_dendritic
        
        # Class token and positional embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads, mlp_ratio, dropout, use_dendritic)
            for _ in range(depth)
        ])
        
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
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch_size, 1, embed_dim)
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
                if hasattr(block.mlp[0], 'num_params'):  # DendriticLayer
                    total_active += block.mlp[0].num_params()
                else:
                    total_active += sum(p.numel() for p in block.mlp.parameters())
            
            # Add non-dendritic parameters
            non_dendritic_params = (
                sum(p.numel() for p in self.patch_embed.parameters()) +
                self.cls_token.numel() + self.pos_embed.numel() +
                sum(p.numel() for p in self.norm.parameters()) +
                sum(p.numel() for p in self.head.parameters())
            )
            
            # Add attention layers
            for block in self.blocks:
                non_dendritic_params += (
                    sum(p.numel() for p in block.norm1.parameters()) +
                    sum(p.numel() for p in block.attn.parameters()) +
                    sum(p.numel() for p in block.norm2.parameters())
                )
            
            total_active += non_dendritic_params
            
            print(f"Dendritic ViT - PyTorch allocated: {total_pytorch}, Logical active: {total_active}")
            return total_active
        else:
            total = sum(p.numel() for p in self.parameters())
            print(f"Standard ViT parameters: {total}")
            return total


def load_dataset(dataset="mnist", batch_size=256):
    """Load MNIST, Fashion-MNIST, or CIFAR-10 dataset with normalization
    
    Args:
        dataset (str): Dataset to load - "mnist", "fashion-mnist", or "cifar10"
        batch_size (int): Batch size for data loaders
        
    Returns:
        tuple: (train_loader, test_loader, input_dim, num_classes)
    """
    
    # Dataset-specific configurations
    if dataset == "mnist":
        # MNIST mean and std
        mean, std = (0.1307,), (0.3081,)
        dataset_class = torchvision.datasets.MNIST
        input_dim = 28 * 28  # 784
        num_classes = 10
        print("Loading MNIST dataset...")
    elif dataset == "fashion-mnist":
        # Fashion-MNIST mean and std
        mean, std = (0.2860,), (0.3530,)
        dataset_class = torchvision.datasets.FashionMNIST
        input_dim = 28 * 28  # 784
        num_classes = 10
        print("Loading Fashion-MNIST dataset...")
    elif dataset == "cifar10":
        # CIFAR-10 mean and std per channel (RGB)
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        dataset_class = torchvision.datasets.CIFAR10
        input_dim = 32 * 32 * 3  # 3072
        num_classes = 10
        print("Loading CIFAR-10 dataset...")
    else:
        raise ValueError(f"Dataset must be 'mnist', 'fashion-mnist', or 'cifar10', got '{dataset}'")
    
    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    # Load datasets
    train_dataset = dataset_class(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = dataset_class(
        root='./data', train=False, download=True, transform=transform
    )
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=8
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=8
    )
    
    return train_loader, test_loader, input_dim, num_classes


def train_model(model, train_loader, test_loader, model_name="Model", n_epochs=20, learning_rate=0.002):
    """Train a neural network (either dendritic or ViT)"""
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Check if this is a dendritic model
    is_dendritic = hasattr(model, 'dendritic_layer')
    
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    
    total_batches = len(train_loader) * n_epochs
    
    with tqdm(total=total_batches, desc=f"Training {model_name}") as pbar:
        for epoch in range(n_epochs):
            model.train()
            train_loss = 0.0
            correct = 0
            total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                
                # Flatten images only for dendritic model
                if is_dendritic:
                    data = data.view(data.size(0), -1)  # Flatten for dendritic
                
                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, target)
                loss.backward()
                
                # Apply masks only for dendritic model
                if is_dendritic:
                    with torch.no_grad():
                        if model.dendritic_layer.dendrite_linear.weight.grad is not None:
                            model.dendritic_layer.dendrite_linear.weight.grad *= model.dendritic_layer.dendrite_mask
                        if model.dendritic_layer.soma_linear.weight.grad is not None:
                            model.dendritic_layer.soma_linear.weight.grad *= model.dendritic_layer.soma_mask
                
                optimizer.step()
                
                # Apply masks to weights after optimizer step (dendritic only)
                if is_dendritic:
                    model.dendritic_layer._apply_masks()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                
                pbar.set_postfix({
                    'Epoch': f'{epoch+1}/{n_epochs}',
                    'Batch': f'{batch_idx+1}/{len(train_loader)}',
                    'Loss': f'{loss.item():.4f}'
                })
                pbar.update(1)
            
            # Calculate epoch metrics
            train_loss /= len(train_loader)
            train_acc = correct / total
            
            # Evaluate on test set
            test_loss, test_acc = evaluate_model(model, test_loader, criterion, is_dendritic)
            
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            test_losses.append(test_loss)
            test_accuracies.append(test_acc)
            
            print(f'Epoch {epoch+1}/{n_epochs}: '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                  f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
    
    return train_losses, train_accuracies, test_losses, test_accuracies


def evaluate_model(model, test_loader, criterion, is_dendritic=None):
    """Evaluate the model on test data"""
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    # Auto-detect if dendritic if not provided
    if is_dendritic is None:
        is_dendritic = hasattr(model, 'dendritic_layer')
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            # Flatten images only for dendritic model
            if is_dendritic:
                data = data.view(data.size(0), -1)  # Flatten for dendritic
            
            outputs = model(data)
            loss = criterion(outputs, target)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    test_loss /= len(test_loader)
    test_acc = correct / total
    
    return test_loss, test_acc


def plot_results(results_dict):
    """Plot training results for multiple models"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    # Plot losses
    for i, (model_name, (train_losses, train_accuracies, test_losses, test_accuracies)) in enumerate(results_dict.items()):
        color = colors[i % len(colors)]
        ax1.plot(train_losses, label=f'{model_name} Train', color=color, linestyle='--', alpha=0.7)
        ax1.plot(test_losses, label=f'{model_name} Test', color=color)
    
    ax1.set_title('Loss over Epochs')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracies
    for i, (model_name, (train_losses, train_accuracies, test_losses, test_accuracies)) in enumerate(results_dict.items()):
        color = colors[i % len(colors)]
        ax2.plot(train_accuracies, label=f'{model_name} Train', color=color, linestyle='--', alpha=0.7)
        ax2.plot(test_accuracies, label=f'{model_name} Test', color=color)
    
    ax2.set_title('Accuracy over Epochs')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    plt.savefig(f'{model_name}_results.png')


def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Configuration
    dataset = "cifar10"  # "mnist", "fashion-mnist", or "cifar10"
    n_epochs = 8
    learning_rate = 0.005
    batch_size = 256
    
    # Load data (get input dimensions and classes from dataset)
    train_loader, test_loader, in_dim, n_classes = load_dataset(dataset, batch_size)
    
    # Get dataset-specific parameters
    if dataset == "mnist" or dataset == "fashion-mnist":
        img_size = 28
        in_channels = 1
        # Dendritic model config
        n_dendrite_inputs = 32
        n_dendrites = 23
        n_neurons = 10
        # ViT config
        patch_size = 4  # 7x7 patches for 28x28 images
        embed_dim = 128
        depth = 4
        n_heads = 8
    elif dataset == "cifar10":
        img_size = 32
        in_channels = 3
        # Dendritic model config  
        n_dendrite_inputs = 128
        n_dendrites = 64
        n_neurons = 32
        # ViT config
        patch_size = 4  # 8x8 patches for 32x32 images
        embed_dim = 192
        depth = 4
        n_heads = 8
    
    # Create models
    print(f"Creating models for {dataset.upper()} dataset...")
    
    # Comment out basic dendritic model for this experiment
    # # 1. Dendritic Neural Network
    # dendritic_model = DendriticNet(
    #     in_dim=in_dim,
    #     n_neurons=n_neurons,
    #     n_dendrite_inputs=n_dendrite_inputs,
    #     n_dendrites=n_dendrites,
    #     n_classes=n_classes,
    #     synaptic_resampling=True,
    #     percentage_resample=0.5,
    #     steps_to_resample=128
    # ).to(device)
    

    # 1 Dendritic Vision Transformer (Dendritic layers)
    vit_dendritic_model = VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=in_channels,
        n_classes=n_classes,
        embed_dim=embed_dim,
        depth=depth,
        n_heads=n_heads,
        dropout=0.1,
        use_dendritic=True
    ).to(device)
    
    # 2. Standard Vision Transformer (FF layers)
    vit_ff_model = VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=in_channels,
        n_classes=n_classes,
        embed_dim=embed_dim,
        depth=depth,
        n_heads=n_heads,
        dropout=0.1,
        use_dendritic=False
    ).to(device)
    
    
    print(f"Standard ViT created with {vit_ff_model.num_params()} parameters")
    print(f"Dendritic ViT created with {vit_dendritic_model.num_params()} parameters")
    
    # Train both ViT models
    results = {}
    
        
    print("\n" + "="*60)
    print("Training Dendritic Vision Transformer...")
    print("="*60)
    vit_dendritic_results = train_model(
        vit_dendritic_model, train_loader, test_loader,
        model_name="ViT-Dendritic", n_epochs=n_epochs, learning_rate=0.002
    )
    results["ViT-Dendritic"] = vit_dendritic_results
    
    print("\n" + "="*60)
    print("Training Standard Vision Transformer (FF layers)...")
    print("="*60)
    vit_ff_results = train_model(
        vit_ff_model, train_loader, test_loader, 
        model_name="ViT-FF", n_epochs=n_epochs, learning_rate=0.002
    )
    results["ViT-FF"] = vit_ff_results

    
    # Print final comparison
    print("\n" + "="*60)
    print("FINAL RESULTS COMPARISON")
    print("="*60)
    
    for model_name, (train_losses, train_accuracies, test_losses, test_accuracies) in results.items():
        print(f"\n{model_name}:")
        print(f"  Train Accuracy: {train_accuracies[-1]*100:.1f}%")
        print(f"  Test Accuracy: {test_accuracies[-1]*100:.1f}%")
        print(f"  Train Loss: {train_losses[-1]:.4f}")
        print(f"  Test Loss: {test_losses[-1]:.4f}")
    
    # Plot comparison
    print("\nGenerating comparison plots...")
    plot_results(results)


if __name__ == "__main__":
    main() 