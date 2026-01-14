import numpy as np
import math, os
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch import Tensor
import time
import layers_kernel
from typing import Any, Callable, List, Tuple, Literal, Optional


#Wczesna wersja optymalizacji funkcji składowych HCR.
#Nietestowana i niegotowa do użytku.

def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} executed in {end - start:.4f}s")
        return result
    return wrapper

'''
@timer
def printer():
    for x in 10000:
        print('hello world')
'''

#Default imports of HCRNN Components.
#Make sure that __init__.py file is correctly opening layers

__all__ = ['CDFNorm', 
           'MeanEstimation', 
           'ConditionalEstimation', 
           'PropagationEstimation', 
           'EntropyAndMutualInformation', 
           'DynamicEMA', 
           'BaseOptimization', 
           'InformationBottleneck']

class CDFNorm(nn.Module):
    # Using Literal for restricted string options improves IDE autocomplete
    method: Literal['gaussian', 'empirical']
    unbiased: bool
    eps: float
    affine: bool
    track_running_stats: bool
    momentum: float
    
    # Parameters and Buffers are optionally Tensors depending on init flags
    weight: Optional[nn.Parameter]
    bias: Optional[nn.Parameter]
    running_mean: Optional[Tensor]
    running_var: Optional[Tensor]
    num_batches_tracked: Optional[Tensor]

    def __init__(
        self, 
        method: Literal['gaussian', 'empirical'] = 'gaussian', 
        unbiased: bool = True, 
        eps: float = 1e-5, 
        affine: bool = False, 
        track_running_stats: bool = True, 
        momentum: float = 0.1
    ) -> None:
        super().__init__()
        self.method = method
        self.unbiased = unbiased
        self.eps = eps
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.momentum = momentum

        if affine:
            self.weight = nn.Parameter(torch.ones(1))
            self.bias = nn.Parameter(torch.zeros(1))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        if track_running_stats:
            self.register_buffer('running_mean', torch.zeros(1))
            self.register_buffer('running_var', torch.ones(1))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_buffer('running_mean', None)
            self.register_buffer('running_var', None)
            self.register_buffer('num_batches_tracked', None)

    def _gaussian_transform(self, x: Tensor) -> Tensor:
        if self.training and self.track_running_stats:
            # Ensure the buffer exists before usage for strict type checking
            assert self.running_mean is not None and self.running_var is not None
            
            var, mean = torch.var_mean(x, unbiased=self.unbiased)
            with torch.no_grad():
                self.running_mean.lerp_(mean, self.momentum)
                self.running_var.lerp_(var, self.momentum)
                if self.num_batches_tracked is not None:
                    self.num_batches_tracked += 1
        else:
            mean, var = self.running_mean, self.running_var

        # Standard deviation and normalization pass
        std: Tensor = torch.sqrt(var + self.eps)
        x_norm: Tensor = 0.5 * (1 + torch.erf((x - mean) / (std * math.sqrt(2.0))))

        if self.affine and self.weight is not None and self.bias is not None:
            return x_norm * self.weight + self.bias
        return x_norm

    def _empirical_transform(self, x: Tensor) -> Tensor:
        # Double argsort provides the rank of each element
        ranks: Tensor = torch.argsort(torch.argsort(x))
        x_norm: Tensor = (ranks.float() + 1) / (len(x) + 1)
        
        if self.affine and self.weight is not None and self.bias is not None:
            return x_norm * self.weight + self.bias
        return x_norm

    def forward(self, x: Tensor) -> Tensor:
        if self.method == 'gaussian':
            return self._gaussian_transform(x)
        elif self.method == 'empirical':
            return self._empirical_transform(x)
        raise ValueError(f"Unsupported normalization method: {self.method}")
    
class MeanEstimation(nn.Module):
    # Defining attributes for clarity and static analysis
    triplets: List[Tuple[Any, Any, Any]]
    feature_fn: Callable[[Any], Tensor]
    feature_dm: int

    def __init__(
        self,
        *,
        triplets: List[Tuple[Any, Any, Any]],
        feature_fn: Callable[[Any], Tensor],
        feature_dm: int
    ) -> None:
        super().__init__()
        self.triplets = triplets
        self.feature_fn = feature_fn
        self.feature_dm = feature_dm

    def forward(self) -> Tensor:
        """
        Calculates the mean estimation using a custom CUDA kernel.
        
        Returns:
            Tensor: The result from the layers_kernel.mean_estimation.
        """
        N: int = len(self.triplets)
        D: int = self.feature_dm

        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Precompute features
        fx_list: List[Tensor] = []
        fy_list: List[Tensor] = []
        fz_list: List[Tensor] = []
        
        for (x, y, z) in self.triplets:
            fx_list.append(self.feature_fn(x))
            fy_list.append(self.feature_fn(y))
            fz_list.append(self.feature_fn(z))

        # Stack and move to device
        fx: Tensor = torch.stack(fx_list).to(device)
        fy: Tensor = torch.stack(fy_list).to(device)
        fz: Tensor = torch.stack(fz_list).to(device)

        # Run CUDA kernel
        # Note: detach().cpu().numpy() converts to a NumPy array, 
        # so 'a' type depends on what the kernel returns (likely a Tensor or Array)
        result_array: Any = layers_kernel.mean_estimation(
            fx.detach().cpu().numpy(),
            fy.detach().cpu().numpy(),
            fz.detach().cpu().numpy()
        )
        
        # Ensure the return type matches the expected Tensor hint
        return torch.as_tensor(result_array)
 
class ConditionalEstimation(nn.Module):
    # Attribute type hints for class properties
    x_candidates: List[Any]
    y: Any
    z: Any
    a: Tensor
    feature_fn: Callable[[Any], Tensor]

    def __init__(
        self,
        *,
        x_candidates: List[Any],
        y: Any,
        z: Any,
        a: Tensor,
        feature_fn: Callable[[Any], Tensor]
    ) -> None:
        super().__init__()
        self.x_candidates = x_candidates
        self.y = y
        self.z = z
        self.a = a
        self.feature_fn = feature_fn

    def forward(self) -> List[Tensor]:
        """
        Computes conditional scores for candidate inputs.
        
        Returns:
            List[Tensor]: A list of dot-product scores for each candidate in x_candidates.
        """
        target_dtype: torch.dtype = self.a.dtype
        target_device: torch.device = self.a.device
    
        # Transform inputs and move to target device/dtype
        fy: Tensor = self.feature_fn(self.y).to(dtype=target_dtype, device=target_device).view(-1)
        fz: Tensor = self.feature_fn(self.z).to(dtype=target_dtype, device=target_device).view(-1)
    
        # Prepare NumPy versions for CUDA kernel processing
        fy_np: np.ndarray = fy.cpu().numpy()
        fz_np: np.ndarray = fz.cpu().numpy()
        a_np: np.ndarray = self.a.cpu().numpy()
        
        # Determine feature dimension D from the y-feature
        D: int = fy_np.shape[0]
    
        # Create dummy x for shape compatibility in kernel
        dummy_x: np.ndarray = np.zeros((1, D), dtype=np.float32)
    
        # Execute custom CUDA kernel to retrieve context
        _, context_np = layers_kernel.conditional_estimation(
            dummy_x, fy_np, fz_np, a_np, return_context=True
        )
    
        # Restore context to PyTorch tensor format
        context_tensor: Tensor = torch.as_tensor(context_np, dtype=target_dtype, device=target_device)
    
        # Iteratively compute dot-product scores for all candidates
        scores: List[Tensor] = []
        for x in self.x_candidates:
            fx: Tensor = self.feature_fn(x).to(dtype=target_dtype, device=target_device).view(-1)
            score: Tensor = torch.dot(fx, context_tensor)
            scores.append(score)
    
        return scores
    
class PropagationEstimation(nn.Module):
    # Class attribute annotations
    y: Any
    z: Any
    a: Tensor
    feature_fn: Callable[[Any], Tensor]

    def __init__(
        self,
        *,
        y: Any,
        z: Any,
        a: Tensor,
        feature_fn: Callable[[Any], Tensor]
    ) -> None:
        super().__init__()
        self.y = y
        self.z = z
        self.a = a
        self.feature_fn = feature_fn

    def forward(self) -> Tensor:
        """
        Computes propagated expectation ratio using custom kernel operations.
        
        Returns:
            Tensor: The normalized propagation estimation result.
        """
        target_dtype: torch.dtype = self.a.dtype
        target_device: torch.device = self.a.device

        # Map inputs to features and ensure they match 'a' in dtype and device
        fy: Tensor = self.feature_fn(self.y).to(dtype=target_dtype, device=target_device).view(-1)
        fz: Tensor = self.feature_fn(self.z).to(dtype=target_dtype, device=target_device).view(-1)

        # Execution of custom CUDA kernels
        # Using .contiguous() is crucial for stable C++ extension performance
        numerator: Tensor = layers_kernel.propagate_expectation(self.a[1].contiguous(), fy, fz)
        denominator: Tensor = layers_kernel.propagate_expectation(self.a[0].contiguous(), fy, fz)

        # Numerical stability handling for ratio calculation
        ratio: Tensor = numerator / (denominator + 1e-8)

        # Normalization and transformation logic
        centered_ratio: Tensor = ratio - 1.0
        const: Tensor = torch.sqrt(torch.tensor(3.0, dtype=target_dtype, device=target_device))
        propagated: Tensor = 0.5 + (1.0 / (2.0 * const)) * centered_ratio

        return propagated
    
class EntropyAndMutualInformation(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def approximate_entropy(self, activations: Tensor) -> Tensor:
        """
        Calculates the approximate Shannon entropy of the given activations.
        
        Args:
            activations: Input tensor of shape (batch_size, num_classes).
            
        Returns:
            Tensor: A scalar tensor representing the mean entropy.
        """
        probs: Tensor = F.softmax(activations, dim=-1)
        log_probs: Tensor = torch.log(probs + 1e-8)
        # Sum over classes, mean over batch
        entropy: Tensor = -(probs * log_probs).sum(dim=-1).mean()
        return entropy

    def approximate_mutual_information(self, act_X: Tensor, act_Y: Tensor) -> Tensor:
        """
        Calculates the mutual information between two sets of activations using a CUDA kernel.
        
        Args:
            act_X: Activations for variable X.
            act_Y: Activations for variable Y.
            
        Returns:
            Tensor: The calculated mutual information.
        """
        probs_X: Tensor = F.softmax(act_X, dim=1)
        probs_Y: Tensor = F.softmax(act_Y, dim=1)
    
        # Ensure the CUDA kernel receives the probability tensors
        return layers_kernel.approximate_mi_cu(probs_X, probs_Y)
    
class DynamicEMA(nn.Module):
    # Attribute annotations for class state
    x: Tensor
    y: Tensor
    z: Tensor
    ema_lambda: float
    a: Tensor

    def __init__(
        self, 
        x: Tensor, 
        y: Tensor, 
        z: Tensor, 
        ema_lambda: float
    ) -> None:
        super().__init__()
        self.x = x
        self.y = y
        self.z = z
        self.ema_lambda = ema_lambda
        
        # Initialize buffer 'a' matching device and dtype of input tensors
        self.a = torch.zeros(
            (len(x), len(y), len(z)), 
            device=x.device, 
            dtype=x.dtype
        )

    def forward(self) -> Tensor:
        """
        Performs an Exponential Moving Average (EMA) update using a custom CUDA kernel.
        
        Returns:
            Tensor: The updated internal state 'a'.
        """
        # Convert tensors to NumPy for kernel compatibility
        # In 2026, explicit typing of intermediate arrays aids static analysis
        a_np: np.ndarray = self.a.detach().cpu().numpy()
        x_np: np.ndarray = self.x.detach().cpu().numpy()
        y_np: np.ndarray = self.y.detach().cpu().numpy()
        z_np: np.ndarray = self.z.detach().cpu().numpy()

        # Execute the update kernel
        # Casting ema_lambda to float ensures compatibility with C++ float/double signatures
        updated: np.ndarray = layers_kernel.ema_update(
            a_np, x_np, y_np, z_np, float(self.ema_lambda)
        )
        
        # Synchronize updated values back to the original device/dtype
        self.a = torch.as_tensor(updated, device=self.x.device, dtype=self.x.dtype)
        
        return self.a

'''
class EMAOptimizedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, ema_params: dict):
        super().__init__()
        # Standard learnable layer
        self.linear = nn.Linear(in_features, out_features)
        
        # DynamicEMA setup (expects x, y, z tensors for its 3D 'a' buffer)
        self.ema_layer = DynamicEMA(
            x=ema_params['x'], 
            y=ema_params['y'], 
            z=ema_params['z'], 
            ema_lambda=ema_params['lambda']
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        # 1. Standard forward pass (gradients are tracked here)
        linear_out = self.linear(input_tensor)
        
        # 2. Update EMA state
        # In your current code, DynamicEMA.forward() is parameterless 
        # and updates internal 'a'. You call it to trigger the CUDA kernel.
        updated_ema_a = self.ema_layer()
        
        # 3. Combine results
        # Example: using the EMA 'a' to scale or shift the linear output
        # Ensure dimensions match (linear_out might need reshaping)
        return linear_out + updated_ema_a.mean()
'''

class BaseOptimization(nn.Module):
    # Annotate class attributes for static analysis
    a: Tensor

    def __init__(self, *, a: Tensor) -> None:
        super().__init__()
        self.a = a

    def forward(self) -> Tensor:
        """
        Optimizes the tensor 'a' using SVD and a custom kernel transformation.
        
        Returns:
            Tensor: The transformed tensor, reconstructed on the original device.
        """
        # Reshape to 2D for SVD; M has shape (first_dim, combined_rest_dims)
        M: Tensor = self.a.reshape(len(self.a[0]), -1)
        
        # Perform Singular Value Decomposition
        # U, S, Vh shapes are determined by M
        U: Tensor
        S: Tensor
        Vh: Tensor
        U, S, Vh = torch.linalg.svd(M, full_matrices=False)

        # Prepare data for CUDA/C++ kernel
        # We use .T (transpose) and .contiguous() to ensure memory layout matches C++ expectations
        u_np: np.ndarray = U.T.contiguous().cpu().numpy()
        a_np: np.ndarray = self.a.cpu().numpy()

        # Run the custom transformation
        new_a_np: np.ndarray = layers_kernel.transform_tensor(u_np, a_np)

        # Convert back to torch.Tensor while preserving original hardware context
        return torch.as_tensor(new_a_np, dtype=self.a.dtype, device=self.a.device)

''' 
class BOModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, a_tensor):
        super().__init__()
        # Standard PyTorch Layer
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # Your Custom Component
        self.optimizer_layer = BaseOptimization(a=a_tensor)
        # Another Standard Layer
        self.fc2 = nn.Linear(hidden_dim, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Pass through standard linear layer
        x = self.fc1(x)
        # 2. Run your custom optimization (SVD + C++ Kernel)
        # Note: BaseOptimization currently ignores input and uses internal self.a
        optimized_a = self.optimizer_layer() 
        # 3. Use the result in downstream operations
        return self.fc2(x + optimized_a.mean()) 
'''

class InformationBottleneck(nn.Module):
    # hyperparameters
    beta: float

    def __init__(self, beta: float = 1.0) -> None:
        super().__init__()
        self.beta = beta

    def forward(self, X_features: Tensor, Y_features: Tensor) -> Tensor:
        """
        Calculates the Hilbert-Schmidt Independence Criterion (HSIC) or similar 
        trace-based dependency measure (Equation 15).
        
        Optimized via the identity Tr((XXᵀ)(YYᵀ)) = ||XᵀY||²_F.
        
        Args:
            X_features: Tensor of shape (batch_size, d_x)
            Y_features: Tensor of shape (batch_size, d_y)
            
        Returns:
            Tensor: A scalar representing the statistical dependency.
        """
        # Feature normalization across the last dimension
        X_norm: Tensor = F.normalize(X_features, dim=-1)
        Y_norm: Tensor = F.normalize(Y_features, dim=-1)

        # Compute the Frobenius norm squared of the cross-covariance matrix
        # This efficiently implements the trace of the product of gram matrices
        XY: Tensor = torch.matmul(X_norm.T, Y_norm)
        return torch.sum(XY.pow(2))

    def bottleneck_loss(
        self, 
        X_features: Tensor, 
        T_features: Tensor, 
        Y_features: Tensor
    ) -> Tensor:
        """
        Calculates the Information Bottleneck loss (Equation 10).
        
        L = I(X; T) - β * I(T; Y)
        
        Args:
            X_features: Input features.
            T_features: Compressed representation (bottleneck).
            Y_features: Target features (labels/signals).
            
        Returns:
            Tensor: The total IB loss.
        """
        I_XT: Tensor = self.forward(X_features, T_features)
        I_TY: Tensor = self.forward(T_features, Y_features)
        
        return I_XT - self.beta * I_TY
    
'''
Infomration Bottleneck example usage
class IBClassifier(nn.Module):
    def __init__(self, input_dim: int, bottleneck_dim: int, num_classes: int, beta: float):
        super().__init__()
        self.encoder = nn.Linear(input_dim, bottleneck_dim)
        self.classifier = nn.Linear(bottleneck_dim, num_classes)
        self.ib_layer = InformationBottleneck(beta=beta)

    def forward(self, x: Tensor):
        t = self.encoder(x)
        logits = self.classifier(t)
        return logits, t

# --- Small Training Simulation ---
# 1. Setup
input_dim, latent_dim, num_classes = 128, 64, 5
model = IBClassifier(input_dim, latent_dim, num_classes, beta=0.1)
ce_loss_fn = nn.CrossEntropyLoss()

# 2. Mock Data
X = torch.randn(3, input_dim, requires_grad=True)
target_labels = torch.empty(3, dtype=torch.long).random_(num_classes)
# For IB, we need Y features (often one-hot labels in practice)
Y_features = F.one_hot(target_labels, num_classes=num_classes).float()

# 3. Forward Pass
logits, T = model(X)

# 4. Combined Loss Calculation
ce_loss = ce_loss_fn(logits, target_labels)
ib_loss = model.ib_layer.bottleneck_loss(X, T, Y_features)
total_loss = ce_loss + ib_loss

# 5. Output like your example
print(f"CrossEntropy: {ce_loss.item():.4f}")
print(f"IB Component: {ib_loss.item():.4f}")
print(f"Total Combined Loss: {total_loss.item():.4f}")

total_loss.backward()

# Verify gradients exist
print(f"Encoder Gradient Check: {model.encoder.weight.grad is not None}")
'''