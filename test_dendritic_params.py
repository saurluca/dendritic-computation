import torch
import torch.nn as nn
import math

# Copy the DendriticLayer class for testing
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
        
        # Dendrite layer (input -> dendrites)
        self.dendrite_linear = nn.Linear(in_dim, self.n_soma_connections)
        self.dendrite_activation = nn.LeakyReLU(0.1)
        
        # Soma layer (dendrites -> output)
        self.soma_linear = nn.Linear(self.n_soma_connections, n_neurons)
        self.soma_activation = nn.LeakyReLU(0.1)
        
        # Create masks
        self._create_masks()
        
        # Apply masks to weights
        self._apply_masks()
    
    def _create_masks(self):
        """Create sparse connectivity masks for dendrites and soma"""
        
        # Soma mask: step pattern where each neuron connects to its specific dendrites
        soma_mask = torch.zeros(self.n_neurons, self.n_soma_connections)
        for i in range(self.n_neurons):
            start_idx = i * self.n_dendrites
            end_idx = start_idx + self.n_dendrites
            soma_mask[i, start_idx:end_idx] = 1
        
        # Dendrite mask: each dendrite connects to n_dendrite_inputs random inputs
        dendrite_mask = torch.zeros(self.n_soma_connections, self.in_dim)
        for i in range(self.n_soma_connections):
            input_idx = torch.randperm(self.in_dim)[:self.n_dendrite_inputs]
            dendrite_mask[i, input_idx] = 1
        
        # Register as buffers
        self.register_buffer('soma_mask', soma_mask)
        self.register_buffer('dendrite_mask', dendrite_mask)
    
    def _apply_masks(self):
        """Apply masks to weight matrices"""
        with torch.no_grad():
            self.dendrite_linear.weight.data *= self.dendrite_mask
            self.soma_linear.weight.data *= self.soma_mask
    
    def pytorch_total_params(self):
        """Count all PyTorch parameters (including masked ones)"""
        total = sum(p.numel() for p in self.parameters())
        return total
    
    def pytorch_breakdown(self):
        """Breakdown of PyTorch parameters"""
        dendrite_w = self.dendrite_linear.weight.numel()
        dendrite_b = self.dendrite_linear.bias.numel()
        soma_w = self.soma_linear.weight.numel()
        soma_b = self.soma_linear.bias.numel()
        
        print(f"PyTorch Parameter Breakdown:")
        print(f"  dendrite_linear.weight: {dendrite_w}")
        print(f"  dendrite_linear.bias: {dendrite_b}")
        print(f"  soma_linear.weight: {soma_w}")
        print(f"  soma_linear.bias: {soma_b}")
        print(f"  Total: {dendrite_w + dendrite_b + soma_w + soma_b}")
        
        return dendrite_w, dendrite_b, soma_w, soma_b
    
    def logical_active_params(self):
        """Count only logically active parameters (like original notebook.py)"""
        active_dendrite_w = self.dendrite_mask.sum().item()
        dendrite_b = self.dendrite_linear.bias.numel()
        active_soma_w = self.soma_mask.sum().item()
        soma_b = self.soma_linear.bias.numel()
        
        total = active_dendrite_w + dendrite_b + active_soma_w + soma_b
        
        print(f"Logical Active Parameter Breakdown:")
        print(f"  active dendrite weights: {active_dendrite_w}")
        print(f"  dendrite biases: {dendrite_b}")
        print(f"  active soma weights: {active_soma_w}")
        print(f"  soma biases: {soma_b}")
        print(f"  Total: {total}")
        
        return int(total)
    
    def forward(self, x):
        x = self.dendrite_linear(x)
        x = self.dendrite_activation(x)
        x = self.soma_linear(x)
        x = self.soma_activation(x)
        return x


def test_parameter_calculations():
    """Test different parameter calculation methods"""
    
    print("="*60)
    print("DENDRITIC LAYER PARAMETER CALCULATION TEST")
    print("="*60)
    
    # Test configurations
    configs = [
        {
            "name": "Small Config",
            "in_dim": 128,
            "n_neurons": 32,
            "n_dendrite_inputs": 16,
            "n_dendrites": 4
        },
        {
            "name": "Medium Config", 
            "in_dim": 192,
            "n_neurons": 192,
            "n_dendrite_inputs": 32,
            "n_dendrites": 8
        }
    ]
    
    for config in configs:
        print(f"\n{config['name']}:")
        print(f"  in_dim={config['in_dim']}, n_neurons={config['n_neurons']}")
        print(f"  n_dendrite_inputs={config['n_dendrite_inputs']}, n_dendrites={config['n_dendrites']}")
        
        # Create dendritic layer
        layer = DendriticLayer(
            in_dim=config['in_dim'],
            n_neurons=config['n_neurons'],
            n_dendrite_inputs=config['n_dendrite_inputs'],
            n_dendrites=config['n_dendrites']
        )
        
        # Count parameters using different methods
        pytorch_total = layer.pytorch_total_params()
        print(f"\nPyTorch Total Parameters: {pytorch_total}")
        
        print()
        layer.pytorch_breakdown()
        
        print()
        logical_active = layer.logical_active_params()
        
        print(f"\nComparison:")
        print(f"  PyTorch total: {pytorch_total}")
        print(f"  Logical active: {logical_active}")
        print(f"  Difference: {pytorch_total - logical_active}")
        print(f"  Sparsity: {1 - logical_active/pytorch_total:.3f}")


def test_ff_vs_dendritic_matching():
    """Test parameter matching between FF and dendritic layers"""
    
    print("\n" + "="*60)
    print("FF vs DENDRITIC PARAMETER MATCHING TEST")
    print("="*60)
    
    embed_dim = 192
    mlp_ratio = 4.0
    mlp_hidden_dim = int(embed_dim * mlp_ratio)
    
    # Standard FF layer parameters
    ff_params = embed_dim * mlp_hidden_dim * 2 + mlp_hidden_dim + embed_dim
    print(f"\nStandard FF layer parameters: {ff_params}")
    print(f"  First linear: {embed_dim} -> {mlp_hidden_dim} = {embed_dim * mlp_hidden_dim + mlp_hidden_dim}")
    print(f"  Second linear: {mlp_hidden_dim} -> {embed_dim} = {mlp_hidden_dim * embed_dim + embed_dim}")
    
    # Test different dendritic configurations
    n_dendrite_inputs = 32  # Fixed as requested
    n_neurons = embed_dim   # Output dimension matches
    
    print(f"\nTesting dendritic configurations with n_dendrite_inputs={n_dendrite_inputs}, n_neurons={n_neurons}")
    
    for n_dendrites in [1, 2, 4, 8, 16]:
        layer = DendriticLayer(
            in_dim=embed_dim,
            n_neurons=n_neurons,
            n_dendrite_inputs=n_dendrite_inputs,
            n_dendrites=n_dendrites
        )
        
        pytorch_total = layer.pytorch_total_params()
        logical_active = layer.logical_active_params()
        
        print(f"\nn_dendrites={n_dendrites}:")
        print(f"  PyTorch total: {pytorch_total}")
        print(f"  Logical active: {logical_active}")
        print(f"  Difference from FF: {pytorch_total - ff_params}")
        print(f"  Active difference from FF: {logical_active - ff_params}")


if __name__ == "__main__":
    test_parameter_calculations()
    test_ff_vs_dendritic_matching() 