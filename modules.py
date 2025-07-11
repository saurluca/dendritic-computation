try:
    import cupy as cp

    # Test if CuPy can actually access CUDA and random number generator
    cp.cuda.Device(0).compute_capability
    cp.random.seed(1)  # Test if random number generator works
    print("Using CuPy (GPU acceleration)")
except (ImportError, Exception) as e:
    import numpy as cp

    print(f"CuPy not available or CUDA error ({type(e).__name__}), using NumPy (CPU)")


class CrossEntropy:
    def __init__(self):
        self.softmax_output = None
        self.target = None
        self.batch_size = None

    def forward(self, logits, target):
        # Handle both single samples and batches
        if logits.ndim == 1:
            # Single sample case - reshape to batch of size 1
            logits = logits.reshape(1, -1)
            target = target.reshape(1, -1)

        # Apply softmax per sample (along axis=1)
        # Subtract max for numerical stability, then exponentiate
        exp_logits = cp.exp(logits - cp.max(logits, axis=1, keepdims=True))
        # Divide by sum of exponentiated logits per sample (along axis=1)
        self.softmax_output = exp_logits / cp.sum(exp_logits, axis=1, keepdims=True)

        self.target = target
        self.batch_size = logits.shape[0]  # Store batch size

        # Compute cross entropy loss per sample, then average over the batch
        # Use a small epsilon for numerical stability with log(0)
        log_softmax = cp.log(self.softmax_output + 1e-15)
        # Only consider the log-probabilities of the true classes
        loss_per_sample = -cp.sum(
            target * log_softmax, axis=1
        )  # Sum over classes for each sample

        # Return the average loss over the batch
        return cp.mean(loss_per_sample)

    def backward(self):
        grad = (self.softmax_output - self.target) / self.batch_size
        return grad

    def __call__(self, logits, target):
        return self.forward(logits, target)


class ReLU:
    def __init__(self):
        self.input = None

    def forward(self, x):
        self.input = x
        return cp.maximum(0, x)

    def backward(self, grad):
        return cp.where(self.input > 0, grad, 0)

    def __call__(self, x):
        return self.forward(x)


class LeakyReLU:
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.input = None

    def forward(self, x):
        self.input = x
        return cp.where(x > 0, x, self.alpha * x)

    def backward(self, grad):
        return cp.where(self.input > 0, grad, self.alpha * grad)

    def __call__(self, x):
        return self.forward(x)


class ELU:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.input = None

    def forward(self, x):
        self.input = x
        return cp.where(x > 0, x, self.alpha * (cp.exp(x) - 1))

    def backward(self, grad):
        return cp.where(self.input > 0, grad, self.alpha * cp.exp(self.input) * grad)

    def __call__(self, x):
        return self.forward(x)


class Dropout:
    def __init__(self, p=0.5):
        """
        Dropout layer that randomly sets input elements to zero with probability p.

        Args:
            p (float): Probability of setting an element to zero. Default is 0.5.
        """
        self.p = p
        self.training = True
        self.mask = None

    def train(self):
        """Set the layer to training mode."""
        self.training = True

    def eval(self):
        """Set the layer to evaluation mode."""
        self.training = False

    def forward(self, x):
        if self.training and self.p > 0:
            # Generate random mask: 1 where we keep the value, 0 where we drop
            self.mask = cp.random.rand(*x.shape) > self.p
            # Apply mask and scale by 1/(1-p) to maintain expected values
            return x * self.mask / (1 - self.p)
        else:
            # During evaluation, just pass through
            self.mask = None
            return x

    def backward(self, grad):
        if self.mask is not None:
            # Apply the same mask to gradients and scale
            return grad * self.mask / (1 - self.p)
        else:
            # During evaluation, just pass gradients through
            return grad

    def __call__(self, x):
        return self.forward(x)


class BatchNormalization:
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        """
        Batch Normalization layer.
        
        Args:
            num_features (int): Number of features in the input
            eps (float): Small constant for numerical stability. Default is 1e-5.
            momentum (float): Momentum for running statistics. Default is 0.1.
            affine (bool): Whether to use learnable affine parameters (gamma, beta). Default is True.
        """
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.training = True
        
        # Learnable parameters
        if self.affine:
            self.gamma = cp.ones(num_features)  # Scale parameter
            self.beta = cp.zeros(num_features)  # Shift parameter
            self.dgamma = 0.0
            self.dbeta = 0.0
        else:
            self.gamma = None
            self.beta = None
            
        # Running statistics (for inference)
        self.running_mean = cp.zeros(num_features)
        self.running_var = cp.ones(num_features)
        
        # Cache for backward pass
        self.input_normalized = None
        self.input_centered = None
        self.std = None
        self.batch_mean = None
        self.batch_var = None
        self.batch_size = None
        
    def train(self):
        """Set the layer to training mode."""
        self.training = True
        
    def eval(self):
        """Set the layer to evaluation mode."""
        self.training = False
        
    def forward(self, x):
        """
        Forward pass of batch normalization.
        
        Args:
            x: Input tensor of shape (batch_size, num_features) or (batch_size, ...)
            
        Returns:
            Normalized output tensor
        """
        # Handle different input shapes
        original_shape = x.shape
        if x.ndim > 2:
            # Flatten all dimensions except the feature dimension
            x = x.reshape(-1, self.num_features)
        
        self.batch_size = x.shape[0]
        
        if self.training:
            # Training mode: use batch statistics
            self.batch_mean = cp.mean(x, axis=0)
            self.batch_var = cp.var(x, axis=0)
            
            # Update running statistics
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * self.batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * self.batch_var
            
            # Normalize
            self.input_centered = x - self.batch_mean
            self.std = cp.sqrt(self.batch_var + self.eps)
            self.input_normalized = self.input_centered / self.std
        else:
            # Inference mode: use running statistics
            self.input_centered = x - self.running_mean
            self.std = cp.sqrt(self.running_var + self.eps)
            self.input_normalized = self.input_centered / self.std
            
        # Apply affine transformation if enabled
        if self.affine:
            output = self.gamma * self.input_normalized + self.beta
        else:
            output = self.input_normalized
            
        # Reshape back to original shape if needed
        if len(original_shape) > 2:
            output = output.reshape(original_shape)
            
        return output
        
    def backward(self, grad):
        """
        Backward pass of batch normalization.
        
        Args:
            grad: Gradient from the next layer
            
        Returns:
            Gradient with respect to input
        """
        # Handle different gradient shapes
        original_shape = grad.shape
        if grad.ndim > 2:
            grad = grad.reshape(-1, self.num_features)
            
        if self.affine:
            # Gradients with respect to gamma and beta
            self.dgamma = cp.sum(grad * self.input_normalized, axis=0)
            self.dbeta = cp.sum(grad, axis=0)
            
            # Scale gradient by gamma
            grad = grad * self.gamma
            
        if self.training:
            # Training mode: compute gradients with respect to input
            # This is the standard batch normalization backward pass
            
            # Gradient with respect to normalized input
            grad_normalized = grad
            
            # Gradient with respect to variance
            grad_var = cp.sum(grad_normalized * self.input_centered, axis=0) * (-0.5) * cp.power(self.batch_var + self.eps, -1.5)
            
            # Gradient with respect to mean
            grad_mean = cp.sum(grad_normalized * (-1.0 / self.std), axis=0) + grad_var * cp.mean(-2.0 * self.input_centered, axis=0)
            
            # Gradient with respect to input
            grad_input = (grad_normalized / self.std) + (grad_var * 2.0 * self.input_centered / self.batch_size) + (grad_mean / self.batch_size)
        else:
            # Inference mode: simpler gradient computation
            grad_input = grad / self.std
            
        # Reshape back to original shape if needed
        if len(original_shape) > 2:
            grad_input = grad_input.reshape(original_shape)
            
        return grad_input
        
    def num_params(self):
        """Return the number of parameters in this layer."""
        if self.affine:
            return self.gamma.size + self.beta.size
        else:
            return 0
            
    def var_params(self):
        """Return the variance of parameters in this layer."""
        if self.affine:
            return cp.var(self.gamma) + cp.var(self.beta)
        else:
            return 0.0
            
    def __call__(self, x):
        return self.forward(x)


class SGD:
    def __init__(self, params, criterion, lr=0.01, momentum=0.9):
        self.params = params
        self.criterion = criterion
        self.lr = lr
        self.momentum = momentum

        # Initialize updates based on layer type
        self.updates = []
        for layer in self.params:
            if hasattr(layer, "dendrite_W"):  # DendriticLayer
                layer_updates = [cp.zeros_like(layer.dendrite_W)]
                if hasattr(layer, "dendrite_bias") and layer.dendrite_bias:
                    layer_updates.append(cp.zeros_like(layer.dendrite_b))
                if hasattr(layer, "soma_enabled") and layer.soma_enabled:
                    layer_updates.append(cp.zeros_like(layer.soma_W))
                    if hasattr(layer, "soma_bias") and layer.soma_bias:
                        layer_updates.append(cp.zeros_like(layer.soma_b))
                self.updates.append(layer_updates)
            elif hasattr(layer, "gamma") and hasattr(layer, "beta"):  # BatchNormalization
                self.updates.append([cp.zeros_like(layer.gamma), cp.zeros_like(layer.beta)])
            else:  # LinearLayer
                layer_updates = [cp.zeros_like(layer.W)]
                if hasattr(layer, "bias") and layer.bias:
                    layer_updates.append(cp.zeros_like(layer.b))
                self.updates.append(layer_updates)

    def zero_grad(self):
        for layer in self.params:
            if hasattr(layer, "dendrite_W"):  # DendriticLayer
                layer.dendrite_dW = 0.0
                if hasattr(layer, "dendrite_bias") and layer.dendrite_bias:
                    layer.dendrite_db = 0.0
                if hasattr(layer, "soma_enabled") and layer.soma_enabled:
                    layer.soma_dW = 0.0
                    if hasattr(layer, "soma_bias") and layer.soma_bias:
                        layer.soma_db = 0.0
            elif hasattr(layer, "gamma") and hasattr(layer, "beta"):  # BatchNormalization
                layer.dgamma = 0.0
                layer.dbeta = 0.0
            else:  # LinearLayer
                layer.dW = 0.0
                if hasattr(layer, "bias") and layer.bias:
                    layer.db = 0.0

    def step(self):
        for layer, update in zip(self.params, self.updates):
            if hasattr(layer, "dendrite_W"):  # DendriticLayer
                idx = 0
                # Update dendrite weights
                update[idx] = self.lr * layer.dendrite_dW + self.momentum * update[idx]
                layer.dendrite_W -= update[idx]
                idx += 1
                
                # Update dendrite bias if enabled
                if hasattr(layer, "dendrite_bias") and layer.dendrite_bias:
                    update[idx] = self.lr * layer.dendrite_db + self.momentum * update[idx]
                    layer.dendrite_b -= update[idx]
                    idx += 1
                
                # Update soma weights and bias if enabled
                if hasattr(layer, "soma_enabled") and layer.soma_enabled:
                    update[idx] = self.lr * layer.soma_dW + self.momentum * update[idx]
                    layer.soma_W -= update[idx]
                    idx += 1
                    
                    if hasattr(layer, "soma_bias") and layer.soma_bias:
                        update[idx] = self.lr * layer.soma_db + self.momentum * update[idx]
                        layer.soma_b -= update[idx]
                    
                # Apply dendrite mask to ensure sparsity is maintained
                layer.dendrite_W = layer.dendrite_W * layer.dendrite_mask
            elif hasattr(layer, "gamma") and hasattr(layer, "beta"):  # BatchNormalization
                update[0] = self.lr * layer.dgamma + self.momentum * update[0]
                update[1] = self.lr * layer.dbeta + self.momentum * update[1]
                layer.gamma -= update[0]
                layer.beta -= update[1]
            else:  # LinearLayer
                idx = 0
                # Update weights
                update[idx] = self.lr * layer.dW + self.momentum * update[idx]
                layer.W -= update[idx]
                idx += 1
                
                # Update bias if enabled
                if hasattr(layer, "bias") and layer.bias:
                    update[idx] = self.lr * layer.db + self.momentum * update[idx]
                    layer.b -= update[idx]

    def __call__(self):
        return self.step()


class Adam:
    def __init__(
        self,
        params,
        criterion,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
        grad_clip=None,
    ):
        self.params = params
        self.criterion = criterion
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0  # Global time step, increments once per batch
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip

        # Initialize moment estimates based on layer type
        self.m = []
        self.v = []
        for layer in self.params:
            if hasattr(layer, "dendrite_W"):  # DendriticLayer
                layer_m = [cp.zeros_like(layer.dendrite_W)]
                layer_v = [cp.zeros_like(layer.dendrite_W)]
                if hasattr(layer, "dendrite_bias") and layer.dendrite_bias:
                    layer_m.append(cp.zeros_like(layer.dendrite_b))
                    layer_v.append(cp.zeros_like(layer.dendrite_b))
                if hasattr(layer, "soma_enabled") and layer.soma_enabled:
                    layer_m.append(cp.zeros_like(layer.soma_W))
                    layer_v.append(cp.zeros_like(layer.soma_W))
                    if hasattr(layer, "soma_bias") and layer.soma_bias:
                        layer_m.append(cp.zeros_like(layer.soma_b))
                        layer_v.append(cp.zeros_like(layer.soma_b))
                self.m.append(layer_m)
                self.v.append(layer_v)
            elif hasattr(layer, "gamma") and hasattr(layer, "beta"):  # BatchNormalization
                self.m.append([cp.zeros_like(layer.gamma), cp.zeros_like(layer.beta)])
                self.v.append([cp.zeros_like(layer.gamma), cp.zeros_like(layer.beta)])
            else:  # LinearLayer
                layer_m = [cp.zeros_like(layer.W)]
                layer_v = [cp.zeros_like(layer.W)]
                if hasattr(layer, "bias") and layer.bias:
                    layer_m.append(cp.zeros_like(layer.b))
                    layer_v.append(cp.zeros_like(layer.b))
                self.m.append(layer_m)
                self.v.append(layer_v)

    def zero_grad(self):
        for layer in self.params:
            if hasattr(layer, "dendrite_W"):  # DendriticLayer
                layer.dendrite_dW = 0.0
                if hasattr(layer, "dendrite_bias") and layer.dendrite_bias:
                    layer.dendrite_db = 0.0
                if hasattr(layer, "soma_enabled") and layer.soma_enabled:
                    layer.soma_dW = 0.0
                    if hasattr(layer, "soma_bias") and layer.soma_bias:
                        layer.soma_db = 0.0
            elif hasattr(layer, "gamma") and hasattr(layer, "beta"):  # BatchNormalization
                layer.dgamma = 0.0
                layer.dbeta = 0.0
            else:  # LinearLayer
                layer.dW = 0.0
                if hasattr(layer, "bias") and layer.bias:
                    layer.db = 0.0

    def step(self):
        self.t += 1  # Increment global time step
        for i, layer in enumerate(self.params):
            if hasattr(layer, "dendrite_W"):  # DendriticLayer
                grads = [layer.dendrite_dW]
                params = [layer.dendrite_W]
                if hasattr(layer, "dendrite_bias") and layer.dendrite_bias:
                    grads.append(layer.dendrite_db)
                    params.append(layer.dendrite_b)
                if hasattr(layer, "soma_enabled") and layer.soma_enabled:
                    grads.append(layer.soma_dW)
                    params.append(layer.soma_W)
                    if hasattr(layer, "soma_bias") and layer.soma_bias:
                        grads.append(layer.soma_db)
                        params.append(layer.soma_b)
            elif hasattr(layer, "gamma") and hasattr(layer, "beta"):  # BatchNormalization
                grads = [layer.dgamma, layer.dbeta]
                params = [layer.gamma, layer.beta]
            else:  # LinearLayer
                grads = [layer.dW]
                params = [layer.W]
                if hasattr(layer, "bias") and layer.bias:
                    grads.append(layer.db)
                    params.append(layer.b)

            # Update moment estimates and parameters for each parameter
            for j, (grad, param) in enumerate(zip(grads, params)):
                # Update first moment estimate
                self.m[i][j] = self.beta1 * self.m[i][j] + (1 - self.beta1) * grad
                # Update second moment estimate
                self.v[i][j] = self.beta2 * self.v[i][j] + (1 - self.beta2) * (grad**2)

                # Bias correction
                m_hat = self.m[i][j] / (1 - self.beta1**self.t)
                v_hat = self.v[i][j] / (1 - self.beta2**self.t)

                if self.grad_clip:
                    m_hat = cp.clip(m_hat, -self.grad_clip, self.grad_clip)

                # Update parameters
                param -= self.lr * m_hat / (cp.sqrt(v_hat) + self.eps)
                param -= self.lr * self.weight_decay * param

            # Apply dendrite mask to ensure sparsity is maintained
            if hasattr(layer, "dendrite_W"):
                layer.dendrite_W = layer.dendrite_W * layer.dendrite_mask

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
            if hasattr(layer, "W") or hasattr(layer, "soma_W") or hasattr(layer, "dendrite_W") or (hasattr(layer, "gamma") and hasattr(layer, "beta")):
                params.append(layer)
        return params

    def num_params(self, verbose=False):
        num_params = 0
        for layer in self.layers:
            if hasattr(layer, "num_params"):
                layer_params = layer.num_params()
                num_params += layer_params
                if verbose:
                    print(f"{layer.__class__.__name__}: {layer_params}")
        return num_params

    def var_params(self):
        var_params = []
        for layer in self.layers:
            if hasattr(layer, "var_params"):
                var_params.append(layer.var_params())
        return var_params

    def train(self):
        """Set all layers to training mode."""
        for layer in self.layers:
            if hasattr(layer, "train"):
                layer.train()

    def eval(self):
        """Set all layers to evaluation mode."""
        for layer in self.layers:
            if hasattr(layer, "eval"):
                layer.eval()

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class LinearLayer:
    """A fully connected, feed forward layer"""

    def __init__(self, in_dim, out_dim, bias=True):
        self.W = cp.random.randn(out_dim, in_dim) * cp.sqrt(
            2.0 / (in_dim)
        )  # He init, for ReLU
        self.bias = bias
        if self.bias:
            self.b = cp.zeros(out_dim)
            self.db = 0.0
        self.dW = 0.0
        self.x = None

    def forward(self, x):
        self.x = x
        if self.bias:
            return x @ self.W.T + self.b
        else:
            return x @ self.W.T

    def backward(self, grad):
        self.dW = grad.T @ self.x
        if self.bias:
            self.db = grad.sum(axis=0)
        grad = grad @ self.W
        return grad

    def num_params(self):
        if self.bias:
            return self.W.size + self.b.size
        else:
            return self.W.size

    def var_params(self):
        if self.bias:
            return cp.var(self.W) + cp.var(self.b)
        else:
            return cp.var(self.W)

    def __call__(self, x):
        return self.forward(x)


class DendriticLayer:
    """A sparse dendritic layer, consiting of dendrites and somas"""

    def __init__(
        self,
        in_dim,
        n_neurons=10,
        strategy="random",
        n_dendrite_inputs=16,
        n_dendrites=4,
        synaptic_resampling=True,
        percentage_resample=0.005,
        scaling_resampling_percentage=False,
        steps_to_resample=100,
        probabilistic_resampling=False,
        local_receptive_field_std_dev_factor=0.5,
        lrf_resampling_prob=0.0,
        dynamic_steps_size=False,
        soma_enabled=True,
        dendrite_bias=True,
        soma_bias=True,
    ):
        assert strategy in ("random", "local-receptive-fields", "fully-connected"), (
            "Invalid strategy"
        )
        self.strategy = strategy
        self.soma_enabled = soma_enabled
        self.dendrite_bias = dendrite_bias
        self.soma_bias = soma_bias
        # When soma is disabled, we only need n_dendrites (not n_neurons * n_dendrites)
        # because each dendrite becomes an independent output unit
        if soma_enabled:
            n_soma_connections = n_dendrites * n_neurons
        else:
            n_soma_connections = n_dendrites
        self.n_neurons = n_neurons
        self.n_dendrites = n_dendrites
        # dynamicly resample
        self.synaptic_resampling = synaptic_resampling
        self.percentage_resample = percentage_resample
        self.steps_to_resample = steps_to_resample
        self.scaling_resampling_percentage = scaling_resampling_percentage
        self.probabilistic_resampling = probabilistic_resampling
        self.local_receptive_field_std_dev_factor = local_receptive_field_std_dev_factor
        self.lrf_resampling_prob = lrf_resampling_prob
        self.dynamic_steps_size = dynamic_steps_size
        # to keep track of resampling
        self.num_mask_updates = 1
        self.update_steps = 0

        self.in_dim = in_dim
        self.n_dendrite_inputs = n_dendrite_inputs
        self.dendrite_W = cp.random.randn(n_soma_connections, in_dim) * cp.sqrt(
            2.0 / (in_dim)
        )  # He init, for ReLU
        
        # Create bias and activation for dendrites
        if self.dendrite_bias:
            self.dendrite_b = cp.zeros((n_soma_connections))
            self.dendrite_db = 0.0
        self.dendrite_activation = LeakyReLU()
        self.dendrite_dW = 0.0

        if self.soma_enabled:
            self.soma_W = cp.random.randn(n_neurons, n_soma_connections) * cp.sqrt(
                2.0 / (n_soma_connections)
            )  # He init, for ReLU
            if self.soma_bias:
                self.soma_b = cp.zeros(n_neurons)
                self.soma_db = 0.0
            self.soma_dW = 0.0
            self.soma_activation = LeakyReLU()

        # inputs to save for backprop
        self.dendrite_x = None
        self.soma_x = None

        if self.soma_enabled:
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
        image_size = int(cp.sqrt(self.in_dim))

        # Choose a center pixel randomly from the image
        center_row = cp.random.randint(0, image_size)
        center_col = cp.random.randint(0, image_size)

        # The standard deviation is the square root of n_dendrite_inputs
        std_dev = cp.power(
            self.n_dendrite_inputs, self.local_receptive_field_std_dev_factor
        )

        # Use a set to store unique indices, starting with the center pixel
        center_idx = (center_row * image_size + center_col).item()
        sampled_indices = {center_idx}

        # Sample until we have n_dendrite_inputs
        while len(sampled_indices) < self.n_dendrite_inputs:
            # Sample one point from a Gaussian distribution around the center
            row_offset = cp.random.normal(loc=0.0, scale=std_dev)
            col_offset = cp.random.normal(loc=0.0, scale=std_dev)

            sampled_row = cp.round(center_row + row_offset)
            sampled_col = cp.round(center_col + col_offset)

            # Clip coordinates to be within image bounds
            sampled_row = cp.clip(sampled_row, 0, image_size - 1)
            sampled_col = cp.clip(sampled_col, 0, image_size - 1)

            # Convert to 1D index and add to set
            idx = sampled_row.item() * image_size + sampled_col.item()
            sampled_indices.add(idx)

        input_idx = cp.array(list(sampled_indices), dtype=int)
        return input_idx

    def forward(self, x):
        # dendrites forward pass
        self.dendrite_x = x
        if self.dendrite_bias:
            x = x @ self.dendrite_W.T + self.dendrite_b
        else:
            x = x @ self.dendrite_W.T
        x = self.dendrite_activation(x)
        
        if self.soma_enabled:
            # soma forward pass
            self.soma_x = x
            if self.soma_bias:
                x = x @ self.soma_W.T + self.soma_b
            else:
                x = x @ self.soma_W.T
            x = self.soma_activation(x)
        return x

    def backward(self, grad):
        if self.soma_enabled:
            grad = self.soma_activation.backward(grad)

            # soma back pass, multiply with mask to keep only valid gradients
            self.soma_dW = grad.T @ self.soma_x * self.soma_mask
            if self.soma_bias:
                self.soma_db = grad.sum(axis=0)
            soma_grad = grad @ self.soma_W

            soma_grad = self.dendrite_activation.backward(soma_grad)

            # dendrite back pass
            self.dendrite_dW = soma_grad.T @ self.dendrite_x * self.dendrite_mask
            if self.dendrite_bias:
                self.dendrite_db = soma_grad.sum(axis=0)
            dendrite_grad = soma_grad @ self.dendrite_W
        else:
            # When soma is disabled, gradients go directly to dendrites
            # dendrite back pass (with activation function and bias)
            grad = self.dendrite_activation.backward(grad)
            self.dendrite_dW = grad.T @ self.dendrite_x * self.dendrite_mask
            if self.dendrite_bias:
                self.dendrite_db = grad.sum(axis=0)
            dendrite_grad = grad @ self.dendrite_W

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

        # Ensure dendrite weights stay masked to maintain constant synapse count
        self.dendrite_W = self.dendrite_W * self.dendrite_mask

        return dendrite_grad

    def _lrf_resample(self, dendrites_to_resample):
        """Resample connections for specified dendrites using a local receptive field approach."""
        image_size = int(cp.sqrt(self.in_dim))
        n_to_resample = dendrites_to_resample.size

        # --- 1. Find the center of each unique dendrite's receptive field ---
        unique_dendrites, inverse_indices = cp.unique(
            dendrites_to_resample, return_inverse=True
        )

        # Get all connections for the unique dendrites that need resampling
        dendrite_masks = self.dendrite_mask[unique_dendrites, :]
        connected_indices = cp.where(dendrite_masks)

        # `connected_indices[0]` maps an entry to an index in `unique_dendrites`
        # `connected_indices[1]` is the input feature index (1D)
        unique_dendrite_map = connected_indices[0]
        input_feature_indices = connected_indices[1]

        # Convert 1D input indices to 2D coordinates to calculate centers
        rows_2d = input_feature_indices // image_size
        cols_2d = input_feature_indices % image_size

        # Calculate the mean row and column for each unique dendrite (vectorized groupby)
        dendrite_counts = cp.bincount(unique_dendrite_map)
        center_rows = (
            cp.bincount(unique_dendrite_map, weights=rows_2d) / dendrite_counts
        )
        center_cols = (
            cp.bincount(unique_dendrite_map, weights=cols_2d) / dendrite_counts
        )

        # --- 2. Map centers back to the original `dendrites_to_resample` list ---
        # This gives us a center for each connection we need to create
        resample_center_rows = center_rows[inverse_indices]
        resample_center_cols = center_cols[inverse_indices]

        # --- 3. Sample new connections from a Gaussian distribution around the centers ---
        std_dev = cp.power(
            self.n_dendrite_inputs, self.local_receptive_field_std_dev_factor
        )

        row_offsets = cp.random.normal(loc=0.0, scale=std_dev, size=n_to_resample)
        col_offsets = cp.random.normal(loc=0.0, scale=std_dev, size=n_to_resample)

        new_rows = cp.round(resample_center_rows + row_offsets)
        new_cols = cp.round(resample_center_cols + col_offsets)

        # Clip coordinates to be within image bounds
        new_rows = cp.clip(new_rows, 0, image_size - 1)
        new_cols = cp.clip(new_cols, 0, image_size - 1)

        # Convert 2D coordinates back to 1D indices
        new_input_indices = new_rows * image_size + new_cols

        return new_input_indices.astype(int)

    def resample_dendrites(self):
        # --- Part 1: Connection Removal ---
        if self.probabilistic_resampling:
            # --- Probabilistic pruning based on weight magnitude ---
            P_MAX_PRUNE = 0.95
            THRESHOLD_W = 0.6
            STEEPNESS = 0.1
            # P_MAX_PRUNE = 0.95
            # THRESHOLD_W = 0.5
            # STEEPNESS = 0.1 with 100

            w_abs = cp.abs(self.dendrite_W)
            # Sigmoid-based pruning probability
            prune_probabilities = P_MAX_PRUNE / (
                1 + cp.exp((w_abs - THRESHOLD_W) / STEEPNESS)
            )

            # Probabilistically decide which connections to prune.
            should_prune_mask = (
                cp.random.random(self.dendrite_W.shape) < prune_probabilities
            ) & (self.dendrite_mask == 1)

            rows_to_remove, cols_to_remove = cp.where(should_prune_mask)

            if rows_to_remove.size == 0:
                # print("num of dendrite successful swaps: 0")
                return

            removed_dendrite_indices = rows_to_remove
            removed_input_indices = cols_to_remove
        else:
            # --- Flat-rate pruning ---
            if self.scaling_resampling_percentage:
                resampling_percentage = 1 / (1 + 0.1 * self.num_mask_updates)
            else:
                resampling_percentage = self.percentage_resample

            n_to_remove_per_dendrite = int(
                self.n_dendrite_inputs * resampling_percentage
            )
            if n_to_remove_per_dendrite == 0:
                return

            num_dendrites = self.dendrite_mask.shape[0]

            # For magnitude, we remove the smallest. Set inactive connections to infinity so they are not picked.
            metric = cp.abs(self.dendrite_W)
            metric[self.dendrite_mask == 0] = cp.inf
            sorted_indices = cp.argsort(metric, axis=1)
            cols_to_remove = sorted_indices[:, :n_to_remove_per_dendrite]

            # Create corresponding row indices and flatten for the swap logic
            rows_to_remove = cp.arange(num_dendrites)[:, cp.newaxis]
            removed_dendrite_indices = rows_to_remove.repeat(
                n_to_remove_per_dendrite, axis=1
            ).flatten()
            removed_input_indices = cols_to_remove.flatten()

        n_connections_to_remove = removed_dendrite_indices.size

        # --- Part 2: One-shot Resampling Attempt ---
        num_inputs_per_dendrite = self.dendrite_x.shape[1]

        # Decide whether to use LRF or random resampling
        use_lrf = (
            self.strategy == "local-receptive-fields"
            and cp.random.random() < self.lrf_resampling_prob
        )

        if use_lrf and n_connections_to_remove > 0:
            newly_selected_input_indices = self._lrf_resample(removed_dendrite_indices)
        else:
            newly_selected_input_indices = cp.random.randint(
                0, num_inputs_per_dendrite, size=n_connections_to_remove, dtype=int
            )

        # --- Part 3: Conflict Detection ---
        conflict_with_existing = (
            self.dendrite_mask[removed_dendrite_indices, newly_selected_input_indices]
            == 1
        )

        num_dendrites = self.dendrite_mask.shape[0]
        proposed_flat_indices = (
            removed_dendrite_indices * num_inputs_per_dendrite
            + newly_selected_input_indices
        )
        counts = cp.bincount(
            proposed_flat_indices.astype(int),
            minlength=num_dendrites * num_inputs_per_dendrite,
        )
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

            self.dendrite_W[dendrites_to_swap, new_inputs_to_add] = cp.random.randn(
                dendrites_to_swap.shape[0]
            ) * cp.sqrt(2.0 / self.in_dim)

        self.dendrite_W = self.dendrite_W * self.dendrite_mask

        # print(f"num of dendrite successful swaps: {dendrites_to_swap.size}")

        self.num_mask_updates += 1

        # --- Part 5: Verification ---
        connections_per_dendrite = cp.sum(self.dendrite_mask, axis=1)
        assert cp.all(connections_per_dendrite == self.n_dendrite_inputs), (
            f"Resampling failed: not all dendrites have {self.n_dendrite_inputs} connections."
        )

    def num_params(self, verbose=False):
        dendrite_weights = int(cp.sum(self.dendrite_mask))
        dendrite_biases = self.dendrite_b.size if self.dendrite_bias else 0
        
        if self.soma_enabled:
            soma_weights = int(cp.sum(self.soma_mask))
            soma_biases = self.soma_b.size if self.soma_bias else 0
            
            print(
                f"\nparameters: dendrite_mask: {dendrite_weights}, dendrite_b: {dendrite_biases}, soma_W: {soma_weights}, soma_b: {soma_biases}"
            )
            return dendrite_weights + dendrite_biases + soma_weights + soma_biases
        else:
            print(
                f"\nparameters: dendrite_mask: {dendrite_weights}, dendrite_b: {dendrite_biases} (soma disabled)"
            )
            return dendrite_weights + dendrite_biases

    def var_params(self):
        var_sum = cp.var(self.dendrite_W)
        if self.dendrite_bias:
            var_sum += cp.var(self.dendrite_b)
            
        if self.soma_enabled:
            var_sum += cp.var(self.soma_W)
            if self.soma_bias:
                var_sum += cp.var(self.soma_b)
                
        return var_sum

    def __call__(self, x):
        return self.forward(x)
