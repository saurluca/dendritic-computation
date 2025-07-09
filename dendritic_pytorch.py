# %%
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

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
        self.dendrite_linear = nn.Linear(in_dim, self.n_soma_connections)
        self.dendrite_activation = nn.LeakyReLU(0.1)
        
        # Soma layer (dendrites -> output)
        self.soma_linear = nn.Linear(self.n_soma_connections, n_neurons)
        self.soma_activation = nn.LeakyReLU(0.1)
        
        # Initialize weights with He initialization
        nn.init.kaiming_normal_(self.dendrite_linear.weight, mode='fan_in', nonlinearity='leaky_relu')
        # nn.init.zeros_(self.dendrite_linear.bias)
        nn.init.kaiming_normal_(self.soma_linear.weight, mode='fan_in', nonlinearity='leaky_relu')
        # nn.init.zeros_(self.soma_linear.bias)
        
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
        """Return number of active parameters"""
        active_dendrite_params = self.dendrite_mask.sum().item()
        dendrite_bias_params = self.dendrite_linear.bias.numel()
        active_soma_params = self.soma_mask.sum().item()
        soma_bias_params = self.soma_linear.bias.numel()
        
        total = active_dendrite_params + dendrite_bias_params + active_soma_params + soma_bias_params
        print(f"Parameters: dendrite_W: {active_dendrite_params}, dendrite_b: {dendrite_bias_params}, "
              f"soma_W: {active_soma_params}, soma_b: {soma_bias_params}")
        return int(total)


class DendriticNet(nn.Module):
    """Complete neural network with dendritic layer"""
    
    def __init__(self, in_dim, n_neurons, n_dendrite_inputs, n_dendrites, n_classes, **kwargs):
        super(DendriticNet, self).__init__()
        
        self.dendritic_layer = DendriticLayer(
            in_dim, n_neurons, n_dendrite_inputs, n_dendrites, **kwargs
        )
        self.output_layer = nn.Linear(n_neurons, n_classes)
        
        # Initialize output layer
        nn.init.kaiming_normal_(self.output_layer.weight, mode='fan_in', nonlinearity='leaky_relu')
        # nn.init.zeros_(self.output_layer.bias)
    
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


def train_model(model, train_loader, test_loader, n_epochs=20, learning_rate=0.002):
    """Train the dendritic neural network"""
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    
    total_batches = len(train_loader) * n_epochs
    
    with tqdm(total=total_batches, desc="Training") as pbar:
        for epoch in range(n_epochs):
            model.train()
            train_loss = 0.0
            correct = 0
            total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                data = data.view(data.size(0), -1)  # Flatten images
                
                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, target)
                loss.backward()
                
                # Apply masks after gradient computation but before optimizer step
                with torch.no_grad():
                    if model.dendritic_layer.dendrite_linear.weight.grad is not None:
                        model.dendritic_layer.dendrite_linear.weight.grad *= model.dendritic_layer.dendrite_mask
                    if model.dendritic_layer.soma_linear.weight.grad is not None:
                        model.dendritic_layer.soma_linear.weight.grad *= model.dendritic_layer.soma_mask
                
                optimizer.step()
                
                # Apply masks to weights after optimizer step
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
            test_loss, test_acc = evaluate_model(model, test_loader, criterion)
            
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            test_losses.append(test_loss)
            test_accuracies.append(test_acc)
            
            print(f'Epoch {epoch+1}/{n_epochs}: '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                  f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
    
    return train_losses, train_accuracies, test_losses, test_accuracies


def evaluate_model(model, test_loader, criterion):
    """Evaluate the model on test data"""
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = data.view(data.size(0), -1)  # Flatten images
            
            outputs = model(data)
            loss = criterion(outputs, target)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    test_loss /= len(test_loader)
    test_acc = correct / total
    
    return test_loss, test_acc


def plot_results(train_losses, train_accuracies, test_losses, test_accuracies):
    """Plot training results"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot losses
    ax1.plot(train_losses, label='Train Loss', color='blue', linestyle='--')
    ax1.plot(test_losses, label='Test Loss', color='blue')
    ax1.set_title('Loss over Epochs')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracies
    ax2.plot(train_accuracies, label='Train Accuracy', color='green', linestyle='--')
    ax2.plot(test_accuracies, label='Test Accuracy', color='green')
    ax2.set_title('Accuracy over Epochs')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Configuration
    dataset = "cifar10"  # "mnist", "fashion-mnist", or "cifar10"
    n_epochs = 20
    learning_rate = 0.002
    batch_size = 256
    
    # Model configuration (adjust based on dataset complexity)
    if dataset == "cifar10":
        n_dendrite_inputs = 128  # More inputs for CIFAR-10's higher complexity
        n_dendrites = 64
        n_neurons = 32
    else:  # MNIST or Fashion-MNIST
        n_dendrite_inputs = 32
        n_dendrites = 23
        n_neurons = 10
    
    # Load data (get input dimensions and classes from dataset)
    train_loader, test_loader, in_dim, n_classes = load_dataset(dataset, batch_size)
    
    # Create model
    model = DendriticNet(
        in_dim=in_dim,
        n_neurons=n_neurons,
        n_dendrite_inputs=n_dendrite_inputs,
        n_dendrites=n_dendrites,
        n_classes=n_classes,
        synaptic_resampling=True,
        percentage_resample=0.5,
        steps_to_resample=128
    ).to(device)
    
    print(f"Model created with {model.num_params()} parameters")
    
    # Train model
    print("\nStarting training...")
    train_losses, train_accuracies, test_losses, test_accuracies = train_model(
        model, train_loader, test_loader, n_epochs, learning_rate
    )
    
    # Print final results
    print(f"\nFinal Results:")
    print(f"Train Accuracy: {train_accuracies[-1]*100:.1f}%")
    print(f"Test Accuracy: {test_accuracies[-1]*100:.1f}%")
    print(f"Train Loss: {train_losses[-1]:.4f}")
    print(f"Test Loss: {test_losses[-1]:.4f}")
    
    # Plot results
    plot_results(train_losses, train_accuracies, test_losses, test_accuracies)


if __name__ == "__main__":
    main() 