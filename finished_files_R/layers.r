library(torch)
library(torchvision) # For nnf_softmax (functional softmax)

CDFNorm <- nn_module(
  classname = "CDFNorm",
  
  initialize = function(method = "gaussian", unbiased = TRUE, eps = 1e-5, affine = FALSE, track_running_stats = TRUE) {
    self$method <- method
    self$unbiased <- unbiased
    self$eps <- eps
    self$affine <- affine
    self$track_running_stats <- track_running_stats
    
    if (self$affine) {
      self$weight <- nn_parameter(torch_ones(1))
      self$bias <- nn_parameter(torch_zeros(1))
    }
    
    if (self$track_running_stats) {
      self$register_buffer("running_mean", torch_zeros(1))
      self$register_buffer("running_var", torch_ones(1))
      self$register_buffer("num_batches_tracked", torch_tensor(0, dtype = torch_long()))
    }
  },
  
  .gaussian_transform = function(x) {
    if (self$training && self$track_running_stats) {
      mean <- x$mean()
      var <- x$var(unbiased = self$unbiased)
      
      # Update running stats without gradients
      torch_no_grad({
        self$running_mean <- (1 - 0.1) * self$running_mean + 0.1 * mean
        self$running_var <- (1 - 0.1) * self$running_var + 0.1 * var
        self$num_batches_tracked <- self$num_batches_tracked + 1
      })
    } else {
      mean <- self$running_mean
      var <- self$running_var
    }
    
    # CDF using error function
    x_norm <- 0.5 * (1 + torch_erf((x - mean) / (torch_sqrt(var + self$eps) * sqrt(2))))
    
    if (self$affine) {
      x_norm <- x_norm * self$weight + self$bias
    }
    
    x_norm
  },
  
  .empirical_transform = function(x) {
    N <- x$numel()
    sorted <- torch_sort(x)
    sorted_x <- sorted[[1]]
    indices <- sorted[[2]]
    
    ranks <- torch_empty_like(indices, dtype = torch_float())
    ranks$index_copy_(1, indices, torch_arange(1, N + 1, device = x$device, dtype = torch_float()))
    
    x_norm <- ranks / N
    
    if (self$affine) {
      x_norm <- x_norm * self$weight + self$bias
    }
    
    x_norm
  },
  
  forward = function(x) {
    if (self$method == "gaussian") {
      self$.gaussian_transform(x)
    } else if (self$method == "empirical") {
      self$.empirical_transform(x)
    } else {
      rlang::abort(paste0("Unsupported normalization method: ", self$method))
    }
  }
)


OrthonormalLegendreBasis <- nn_module(
  classname = "OrthonormalLegendreBasis",
  
  initialize = function(max_degree) {
    self$max_degree <- max_degree
    
    # Legendre coefficients shifted to [0,1]
    coeffs <- torch_tensor(matrix(c(
      1,   0,   0,   0,    # P0(x) = 1
     -1,   2,   0,   0,    # P1(x) = 2x - 1
      1,  -6,   6,   0,    # P2(x) = 6x² - 6x + 1
     -1,  12, -30,  20     # P3(x) = 20x³ - 30x² + 12x - 1
    ), nrow = 4, byrow = TRUE), dtype = torch_float())
    
    self$register_buffer("legendre_coeffs", coeffs)
  },
  
  forward = function(x) {
    x <- x$to(dtype = torch_float())$clamp(0, 1)
    
    # Create powers of x: [x^0, x^1, x^2, x^3]
    powers <- torch_stack(lapply(0:3, function(i) x^i), dim = -1)
    
    # Multiply with coefficients: einsum('...i,ji->...j')
    legendre <- torch_einsum("...i,ji->...j", powers, self$legendre_coeffs)
    
    # Normalization coefficients
    norms <- torch_tensor(c(1.0, sqrt(3), sqrt(5), sqrt(7)), dtype = torch_float(), device = x$device)
    
    return(legendre[ , , 1:(self$max_degree + 1)] / norms[1:(self$max_degree + 1)])
  }
)

JointDistribution <- nn_module(
  classname = "JointDistribution",
  
  initialize = function(dim, basis_size = 4) {
    self$dim <- dim
    self$basis_size <- basis_size
    self$coeffs <- nn_parameter(torch_zeros(rep(basis_size, dim)))
    self$basis <- OrthonormalLegendreBasis(basis_size - 1)
  },
  
  forward = function(...) {
    inputs <- list(...)
    processed_inputs <- list()
    
    # Convert all inputs to torch tensors
    for (x in inputs) {
      if (!inherits(x, "torch_tensor")) {
        x <- torch_tensor(x, dtype = torch_float())
      }
      if (x$ndim == 0) {
        x <- x$unsqueeze(1)
      }
      processed_inputs <- append(processed_inputs, list(x$to(dtype = torch_float())))
    }
    
    basis_values <- lapply(processed_inputs, function(x) self$basis(x))
    
    if (self$dim == 2) {
      out <- torch_einsum("bi,bj,ij->b", basis_values[[1]], basis_values[[2]], self$coeffs)
    } else if (self$dim == 3) {
      out <- torch_einsum("bi,bj,bk,ijk->b",
                          basis_values[[1]], basis_values[[2]], basis_values[[3]], self$coeffs)
    } else {
      rlang::abort(paste0("Invalid dimensionality: ", self$dim, ". Supported: 2 or 3."))
    }
    
    return(out$squeeze())
  }
)

MeanEstimation <- nn_module(
  classname = "MeanEstimation",
  
  initialize = function(triplets, feature_fn, feature_dm) {
    self$triplets <- triplets         # list of (x, y, z)
    self$feature_fn <- feature_fn     # mapping function
    self$feature_dm <- feature_dm     # dimension D
  },
  
  compute_tensor_mean = function() {
    # Initialize 3D array with zeros
    a <- array(0, dim = c(self$feature_dm, self$feature_dm, self$feature_dm))
    
    for (triplet in self$triplets) {
      x <- triplet[[1]]
      y <- triplet[[2]]
      z <- triplet[[3]]
      
      fx <- self$feature_fn(x)
      fy <- self$feature_fn(y)
      fz <- self$feature_fn(z)
      
      # Outer product: einsum('i,j,k->ijk')
      outer <- array(0, dim = c(length(fx), length(fy), length(fz)))
      for (i in seq_along(fx)) {
        for (j in seq_along(fy)) {
          for (k in seq_along(fz)) {
            outer[i, j, k] <- fx[i] * fy[j] * fz[k]
          }
        }
      }
      
      a <- a + outer
    }
    
    a <- a / length(self$triplets)  # normalize by number of triplets
    a
  }
)


ConditionalEstimation <- nn_module(
  classname = "ConditionalEstimation",
  
  initialize = function(x_candidates, y, z, a, feature_fn) {
    self$x_candidates <- x_candidates   # list or vector of x candidates
    self$y <- y                         # single value or vector
    self$z <- z                         # single value or vector
    self$a <- a                         # 3D array [D x D x D]
    self$feature_fn <- feature_fn       # feature mapping function
  },
  
  conditional_score = function() {
    D <- dim(self$a)[1]
    fy <- self$feature_fn(self$y)
    fz <- self$feature_fn(self$z)
    
    # Compute denominator: sum over j,k of a[0,j,k] * fy[j] * fz[k]
    denominator <- 0
    for (j in 1:D) {
      for (k in 1:D) {
        denominator <- denominator + self$a[1, j, k] * fy[j] * fz[k]
      }
    }
    
    scores <- c()
    for (x in self$x_candidates) {
      fx <- self$feature_fn(x)
      
      score <- 0
      for (i in 1:D) {
        context_sum <- 0
        for (j in 1:D) {
          for (k in 1:D) {
            context_sum <- context_sum + self$a[i, j, k] * fy[j] * fz[k]
          }
        }
        score <- score + fx[i] * (context_sum / (denominator + 1e-8))
      }
      
      scores <- c(scores, score)
    }
    
    scores
  }
)


PropagationEstimation <- nn_module(
  classname = "PropagationEstimation",
  
  initialize = function(y, z, a, feature_fn) {
    self$y <- y                 # value or vector for y
    self$z <- z                 # value or vector for z
    self$a <- a                 # 3D torch tensor [D x D x D]
    self$feature_fn <- feature_fn
  },
  
  propagate_expectation = function() {
    fy <- self$feature_fn(self$y)$to(dtype = torch_float())$view(c(-1))
    fz <- self$feature_fn(self$z)$to(dtype = torch_float())$view(c(-1))
    
    numerator <- torch_einsum("jk,j,k->", self$a[2, ..], fy, fz)
    denominator <- torch_einsum("jk,j,k->", self$a[1, ..], fy, fz)
    
    ratio <- numerator / (denominator + 1e-8)
    
    # Center ratio around 1.0
    centered_ratio <- ratio - 1.0
    
    propagated <- 0.5 + (1.0 / (2.0 * torch_sqrt(torch_tensor(3.0, dtype = torch_float())))) * centered_ratio
    
    propagated
  }
)

EntropyAndMutualInformation <- nn_module(
  classname = "EntropyAndMutualInformation",
  
  initialize = function() {
    # No parameters to initialize
  },
  
  approximate_entropy = function(activations) {
    # Normalize activation probabilities
    probs <- nnf_softmax(activations, dim = 2)   # dim=1 in Python -> dim=2 in R
    entropy <- -torch_sum(probs ^ 2, dim = 2)$mean()
    entropy
  },
  
  approximate_mutual_information = function(act_X, act_Y) {
    probs_X <- nnf_softmax(act_X, dim = 2)
    probs_Y <- nnf_softmax(act_Y, dim = 2)
    
    # Compute outer product per batch: torch.bmm equivalent
    joint_probs <- torch_bmm(probs_X$unsqueeze(3), probs_Y$unsqueeze(2))
    
    mi <- torch_sum(joint_probs ^ 2, dim = c(2,3))$mean()
    mi
  }
)

DynamicEMA <- nn_module(
  classname = "DynamicEMA",
  
  initialize = function(x, y, z, ema_lambda) {
    self$x <- x
    self$y <- y
    self$z <- z
    self$ema_lambda <- ema_lambda
    
    # Create empty tensor matching einsum result
    self$a <- torch_zeros_like(torch_einsum("i,j,k->ijk", x, y, z))
  },
  
  EMAUpdateMethod = function() {
    f_i <- function(x) x
    f_j <- function(y) y
    f_k <- function(z) z
    
    update_tensor <- torch_einsum("i,j,k->ijk", f_i(self$x), f_j(self$y), f_k(self$z))
    
    self$a <- (1 - self$ema_lambda) * self$a + self$ema_lambda * update_tensor
    self$a
  }
)

BaseOptimization <- nn_module(
  classname = "BaseOptimization",
  
  initialize = function(a) {
    self$a <- a   # Tensor to optimize (3D torch tensor)
  },
  
  optimization_early = function() {
    # Reshape a into matrix M (like len(self.a[0]) x -1 in Python)
    M <- self$a$reshape(c(dim(self$a)[2], -1))
    
    # Compute SVD
    svd_result <- torch_linalg_svd(M, full_matrices = FALSE)
    U <- svd_result[[1]]
    S <- svd_result[[2]]
    Vh <- svd_result[[3]]
    
    # Example basis transformation function
    f_x <- function(x) {
      torch_sin(x * torch_linspace(0, 1, dim(self$a)[3]))
    }
    
    # g_i(x) = U^T * f(x)
    g_i <- function(x, U) {
      f <- f_x(x)
      torch_matmul(U$t(), f)
    }
    
    # Tensor transformation
    new_a <- torch_einsum("li,ljk->ijk", U$t(), self$a)
    new_a
  }
)

InformationBottleneck <- nn_module(
  classname = "InformationBottleneck",
  
  initialize = function(beta = 1.0) {
    self$beta <- beta
  },
  
  forward = function(X_features, Y_features) {
    # Implements equation (15)
    C_X <- torch_matmul(X_features, X_features$t())
    C_Y <- torch_matmul(Y_features, Y_features$t())
    torch_trace(torch_matmul(C_X, C_Y))
  },
  
  bottleneck_loss = function(X_features, T_features, Y_features) {
    # Implements equation (10)
    I_XT <- self$forward(X_features, T_features)
    I_TY <- self$forward(T_features, Y_features)
    I_XT - self$beta * I_TY
  }
)
