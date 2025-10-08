library(torch)
library(torchvision)


# --- CDF Normalization -----------------------------------------------------------

setClass(
  "CDFNorm",
  slots = list(
    method = "character",
    unbiased = "logical",
    eps = "numeric",
    affine = "logical",
    track_running_stats = "logical",
    weight = "ANY",
    bias = "ANY",
    running_mean = "ANY",
    running_var = "ANY",
    num_batches_tracked = "ANY",
    training = "logical"
  )
)

CDFNorm <- function(method = "gaussian", unbiased = TRUE, eps = 1e-5,
                    affine = FALSE, track_running_stats = TRUE) {

  weight <- bias <- running_mean <- running_var <- num_batches_tracked <- NULL

  if (affine) {
    weight <- nn_parameter(torch_ones(1))
    bias <- nn_parameter(torch_zeros(1))
  }

  if (track_running_stats) {
    running_mean <- torch_zeros(1)
    running_var <- torch_ones(1)
    num_batches_tracked <- torch_tensor(0, dtype = torch_long())
  }

  new("CDFNorm",
      method = method,
      unbiased = unbiased,
      eps = eps,
      affine = affine,
      track_running_stats = track_running_stats,
      weight = weight,
      bias = bias,
      running_mean = running_mean,
      running_var = running_var,
      num_batches_tracked = num_batches_tracked,
      training = TRUE)
}

setGeneric(".gaussian_transform", function(object, x) {
  standardGeneric(".gaussian_transform")
})

setMethod(
  ".gaussian_transform",
  signature(object = "CDFNorm"),
  function(object, x) {
    if (object@training && object@track_running_stats) {
      mean <- x$mean()
      var <- x$var(unbiased = object@unbiased)

      torch_no_grad({
        object@running_mean <- (1 - 0.1) * object@running_mean + 0.1 * mean
        object@running_var <- (1 - 0.1) * object@running_var + 0.1 * var
        object@num_batches_tracked <- object@num_batches_tracked + 1
      })
    } else {
      mean <- object@running_mean
      var <- object@running_var
    }

    # Gaussian CDF via erf
    x_norm <- 0.5 * (1 + torch_erf((x - mean) / (torch_sqrt(var + object@eps) * sqrt(2))))

    if (object@affine) {
      x_norm <- x_norm * object@weight + object@bias
    }

    return(x_norm)
  }
)

setGeneric(".empirical_transform", function(object, x) {
  standardGeneric(".empirical_transform")
})

setMethod(
  ".empirical_transform",
  signature(object = "CDFNorm"),
  function(object, x) {
    N <- x$numel()
    sorted <- torch_sort(x)
    sorted_x <- sorted[[1]]
    indices <- sorted[[2]]

    ranks <- torch_empty_like(indices, dtype = torch_float())
    ranks$index_copy_(1, indices,
                      torch_arange(1, N + 1, device = x$device, dtype = torch_float()))

    x_norm <- ranks / N

    if (object@affine) {
      x_norm <- x_norm * object@weight + object@bias
    }

    return(x_norm)
  }
)

setGeneric("forward", function(object, x) {
  standardGeneric("forward")
})

setMethod(
  "forward",
  signature(object = "CDFNorm"),
  function(object, x) {
    if (object@method == "gaussian") {
      return(.gaussian_transform(object, x))
    } else if (object@method == "empirical") {
      return(.empirical_transform(object, x))
    } else {
      stop(paste0("Unsupported normalization method: ", object@method))
    }
  }
)


# --- Orthonormal Legendre Basis -----------------------------------------------------------

setClass(
  "OrthonormalLegendreBasis",
  slots = list(
    max_degree = "numeric",
    legendre_coeffs = "torch_tensor"
  )
)

OrthonormalLegendreBasis <- function(max_degree) {
  if (!is.numeric(max_degree) || length(max_degree) != 1) {
    stop("'max_degree' must be a single numeric value.")
  }

  # Shifted Legendre coefficients on [0,1]
  coeffs <- torch_tensor(matrix(c(
    1,   0,   0,   0,    # P0(x) = 1
    -1,   2,   0,   0,    # P1(x) = 2x - 1
    1,  -6,   6,   0,    # P2(x) = 6x² - 6x + 1
    -1,  12, -30,  20     # P3(x) = 20x³ - 30x² + 12x - 1
  ), nrow = 4, byrow = TRUE), dtype = torch_float())

  new("OrthonormalLegendreBasis",
      max_degree = max_degree,
      legendre_coeffs = coeffs)
}

setGeneric("forward", function(object, x) {
  standardGeneric("forward")
})

setMethod(
  "forward",
  signature(object = "OrthonormalLegendreBasis"),
  function(object, x) {
    x <- x$to(dtype = torch_float())$clamp(0, 1)

    # Create tensor powers: [x^0, x^1, x^2, x^3]
    powers <- torch_stack(lapply(0:3, function(i) x^i), dim = -1)

    # Multiply with coefficients: einsum('...i,ji->...j')
    legendre <- torch_einsum("...i,ji->...j",
                             list(powers, object@legendre_coeffs))

    # Normalization constants
    norms <- torch_tensor(c(1.0, sqrt(3), sqrt(5), sqrt(7)),
                          dtype = torch_float(),
                          device = x$device)

    # Return only up to max_degree
    legendre_out <- legendre[ , , 1:(object@max_degree + 1)] /
      norms[1:(object@max_degree + 1)]

    return(legendre_out)
  }
)


# --- Joint Distribution -----------------------------------------------------------

setClass(
  "JointDistribution",
  slots = list(
    dim = "numeric",            # Dimensionality (2 or 3)
    basis_size = "numeric",     # Number of basis functions per dimension
    coeffs = "torch_tensor",    # Learnable coefficients tensor
    basis = "ANY"               # Basis object (e.g., OrthonormalLegendreBasis)
  )
)

JointDistribution <- function(dim, basis_size = 4) {
  if (!dim %in% c(2, 3)) {
    stop("Only 2D or 3D distributions are supported.")
  }

  coeff_shape <- rep(basis_size, dim)
  coeffs <- nn_parameter(torch_zeros(coeff_shape))

  # You must define or import OrthonormalLegendreBasis before using this
  basis <- OrthonormalLegendreBasis(basis_size - 1)

  new("JointDistribution",
      dim = dim,
      basis_size = basis_size,
      coeffs = coeffs,
      basis = basis)
}

setGeneric("forward", function(object, ...) {
  standardGeneric("forward")
})

setMethod(
  "forward",
  signature(object = "JointDistribution"),
  function(object, ...) {
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

    # Compute basis values
    basis_values <- lapply(processed_inputs, function(x) object@basis(x))

    # Compute joint density via einsum
    if (object@dim == 2) {
      out <- torch_einsum("bi,bj,ij->b",
                          list(basis_values[[1]], basis_values[[2]], object@coeffs))
    } else if (object@dim == 3) {
      out <- torch_einsum("bi,bj,bk,ijk->b",
                          list(basis_values[[1]], basis_values[[2]], basis_values[[3]], object@coeffs))
    } else {
      stop(paste0("Invalid dimensionality: ", object@dim, ". Supported: 2 or 3."))
    }

    return(out$squeeze())
  }
)


# --- Mean Estimation -----------------------------------------------------------

setClass(
  "MeanEstimation",
  slots = list(
    triplets = "list",        # list of (x, y, z)
    feature_fn = "function",  # mapping function
    feature_dm = "numeric"    # dimension D
  )
)

MeanEstimation <- function(triplets, feature_fn, feature_dm) {
  if (!is.list(triplets)) {
    stop("'triplets' must be a list of (x, y, z) elements.")
  }
  if (!is.function(feature_fn)) {
    stop("'feature_fn' must be a function.")
  }
  if (length(feature_dm) != 1 || !is.numeric(feature_dm)) {
    stop("'feature_dm' must be a single numeric value.")
  }

  new("MeanEstimation",
      triplets = triplets,
      feature_fn = feature_fn,
      feature_dm = feature_dm)
}

setGeneric("compute_tensor_mean", function(object) {
  standardGeneric("compute_tensor_mean")
})

setMethod(
  "compute_tensor_mean",
  signature(object = "MeanEstimation"),
  function(object) {
    D <- object@feature_dm
    a <- array(0, dim = c(D, D, D))  # Initialize accumulator

    for (triplet in object@triplets) {
      x <- triplet[[1]]
      y <- triplet[[2]]
      z <- triplet[[3]]

      fx <- object@feature_fn(x)
      fy <- object@feature_fn(y)
      fz <- object@feature_fn(z)

      # Outer product manually: einsum('i,j,k->ijk')
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

    a <- a / length(object@triplets)  # normalize by number of triplets
    return(a)
  }
)


# --- Conditional Estimation -------------------------------------------------------

setClass(
  "ConditionalEstimation",
  slots = list(
    x_candidates = "list",      # list or vector of candidate x values
    y = "ANY",                  # torch tensor or numeric vector
    z = "ANY",                  # torch tensor or numeric vector
    a = "torch_tensor",         # 3D tensor [D x D x D]
    feature_fn = "function"     # feature mapping function
  )
)

ConditionalEstimation <- function(x_candidates, y, z, a, feature_fn) {
  if (!inherits(a, "torch_tensor")) {
    stop("'a' must be a torch_tensor.")
  }
  if (!is.function(feature_fn)) {
    stop("'feature_fn' must be a function.")
  }
  if (!is.list(x_candidates)) {
    x_candidates <- as.list(x_candidates)
  }

  new("ConditionalEstimation",
      x_candidates = x_candidates,
      y = y,
      z = z,
      a = a,
      feature_fn = feature_fn)
}

setGeneric("conditional_score", function(object) {
  standardGeneric("conditional_score")
})

setMethod(
  "conditional_score",
  signature(object = "ConditionalEstimation"),
  function(object) {
    D <- object@a$size()[1]

    fy <- object@feature_fn(object@y)
    fz <- object@feature_fn(object@z)

    # --- Denominator: sum over j,k of a[1,j,k] * fy[j] * fz[k] ---
    denominator <- torch_zeros(1)
    for (j in 1:D) {
      for (k in 1:D) {
        denominator <- denominator + object@a[1, j, k] * fy[j] * fz[k]
      }
    }

    scores <- c()

    # --- Compute score for each x candidate ---
    for (x in object@x_candidates) {
      fx <- object@feature_fn(x)

      score <- torch_zeros(1)
      for (i in 1:D) {
        context_sum <- torch_zeros(1)
        for (j in 1:D) {
          for (k in 1:D) {
            context_sum <- context_sum + object@a[i, j, k] * fy[j] * fz[k]
          }
        }
        score <- score + fx[i] * (context_sum / (denominator + 1e-8))
      }

      scores <- c(scores, as.numeric(score))
    }

    return(scores)
  }
)


# --- Propagation Estimation -------------------------------------------------------

setClass(
  "PropagationEstimation",
  slots = list(
    y = "ANY",             # input value or torch tensor
    z = "ANY",             # input value or torch tensor
    a = "torch_tensor",    # 3D tensor [D x D x D]
    feature_fn = "function" # function handle
  )
)

PropagationEstimation <- function(y, z, a, feature_fn) {
  if (!inherits(a, "torch_tensor")) {
    stop("'a' must be a torch_tensor.")
  }
  if (!is.function(feature_fn)) {
    stop("'feature_fn' must be a function.")
  }

  new("PropagationEstimation",
      y = y,
      z = z,
      a = a,
      feature_fn = feature_fn)
}

setGeneric("propagate_expectation", function(object) {
  standardGeneric("propagate_expectation")
})

setMethod(
  "propagate_expectation",
  signature(object = "PropagationEstimation"),
  function(object) {
    # Apply feature function and flatten
    fy <- object@feature_fn(object@y)$to(dtype = torch_float())$view(c(-1))
    fz <- object@feature_fn(object@z)$to(dtype = torch_float())$view(c(-1))

    # Select tensor slices (equivalent to self$a[2, ..] and self$a[1, ..])
    a_slice2 <- object@a[2, ..]
    a_slice1 <- object@a[1, ..]

    # Einsum operations: "jk,j,k->"
    numerator <- torch_einsum("jk,j,k->", list(a_slice2, fy, fz))
    denominator <- torch_einsum("jk,j,k->", list(a_slice1, fy, fz))

    ratio <- numerator / (denominator + 1e-8)

    # Center ratio around 1.0
    centered_ratio <- ratio - 1.0

    propagated <- 0.5 + (1.0 / (2.0 * torch_sqrt(torch_tensor(3.0, dtype = torch_float())))) * centered_ratio
    return(propagated)
  }
)


# --- Entropy and Mutual Information -------------------------------------------------------

setClass(
  "EntropyAndMutualInformation",
  slots = list()  # no slots needed
)

EntropyAndMutualInformation <- function() {
  new("EntropyAndMutualInformation")
}

setGeneric("approximate_entropy", function(object, activations) {
  standardGeneric("approximate_entropy")
})

setGeneric("approximate_mutual_information", function(object, act_X, act_Y) {
  standardGeneric("approximate_mutual_information")
})

setMethod(
  "approximate_entropy",
  signature(object = "EntropyAndMutualInformation"),
  function(object, activations) {
    # Softmax normalization (dim=2 corresponds to Python dim=1)
    probs <- nnf_softmax(activations, dim = 2)
    entropy <- -torch_sum(probs ^ 2, dim = 2)$mean()
    return(entropy)
  }
)

setMethod(
  "approximate_mutual_information",
  signature(object = "EntropyAndMutualInformation"),
  function(object, act_X, act_Y) {
    # Compute normalized probabilities
    probs_X <- nnf_softmax(act_X, dim = 2)
    probs_Y <- nnf_softmax(act_Y, dim = 2)

    # Compute outer product for each batch (batched matrix multiplication)
    joint_probs <- torch_bmm(probs_X$unsqueeze(3), probs_Y$unsqueeze(2))

    mi <- torch_sum(joint_probs ^ 2, dim = c(2, 3))$mean()
    return(mi)
  }
)


# --- Dynamic Exponential Moving Average -------------------------------------

setClass(
  "DynamicEMA",
  slots = list(
    x = "torch_tensor",
    y = "torch_tensor",
    z = "torch_tensor",
    ema_lambda = "numeric",
    a = "torch_tensor"
  )
)

DynamicEMA <- function(x, y, z, ema_lambda) {
  if (!inherits(x, "torch_tensor") || !inherits(y, "torch_tensor") || !inherits(z, "torch_tensor")) {
    stop("x, y, and z must be torch_tensor objects.")
  }
  if (!is.numeric(ema_lambda) || ema_lambda <= 0 || ema_lambda > 1) {
    stop("'ema_lambda' must be a numeric value in (0, 1].")
  }

  # initialize a as zero tensor with correct shape
  a_init <- torch_zeros_like(torch_einsum("i,j,k->ijk", list(x, y, z)))

  new("DynamicEMA",
      x = x,
      y = y,
      z = z,
      ema_lambda = ema_lambda,
      a = a_init)
}

setGeneric("EMAUpdateMethod", function(object) {
  standardGeneric("EMAUpdateMethod")
})

setMethod(
  "EMAUpdateMethod",
  signature(object = "DynamicEMA"),
  function(object) {
    # simple identity functions
    f_i <- function(x) x
    f_j <- function(y) y
    f_k <- function(z) z

    update_tensor <- torch_einsum("i,j,k->ijk",
                                  list(f_i(object@x),
                                       f_j(object@y),
                                       f_k(object@z)))

    # EMA update
    object@a <- (1 - object@ema_lambda) * object@a +
      object@ema_lambda * update_tensor

    return(object@a)
  }
)


# --- Base Optimization -----------------------------------------------------------

setClass(
  "BaseOptimization",
  slots = list(
    a = "torch_tensor"   # the tensor to optimize
  )
)

BaseOptimization <- function(a) {
  if (!inherits(a, "torch_tensor")) {
    stop("Input 'a' must be a torch_tensor.")
  }
  new("BaseOptimization", a = a)
}

setGeneric("optimization_early", function(object) {
  standardGeneric("optimization_early")
})

setMethod(
  "optimization_early",
  signature(object = "BaseOptimization"),
  function(object) {
    a <- object@a

    # Reshape a into matrix M (like len(self.a[0]) x -1)
    dims <- a$size()
    M <- a$reshape(c(dims[2], -1))

    # Compute SVD
    svd_result <- torch_linalg_svd(M, full_matrices = FALSE)
    U <- svd_result[[1]]
    S <- svd_result[[2]]
    Vh <- svd_result[[3]]

    # Example basis transformation function f_x
    f_x <- function(x) {
      torch_sin(x * torch_linspace(0, 1, dims[3]))
    }

    # g_i(x) = U^T * f(x)
    g_i <- function(x, U) {
      f <- f_x(x)
      torch_matmul(U$t(), f)
    }

    # Tensor transformation new_a = einsum("li,ljk->ijk", U^T, a)
    new_a <- torch_einsum("li,ljk->ijk", list(U$t(), a))
    new_a
  }
)


# --- Information Bottleneck -----------------------------------------------------------

setClass(
  "InformationBottleneck",
  slots = list(
    beta = "numeric"
  )
)

InformationBottleneck <- function(beta = 1.0) {
  new("InformationBottleneck", beta = beta)
}

setGeneric("forward_IB", function(object, X_features, Y_features) {
  standardGeneric("forward_IB")
})

setMethod(
  "forward_IB",
  signature(object = "InformationBottleneck"),
  function(object, X_features, Y_features) {
    # Implements equation (15)
    C_X <- torch_matmul(X_features, X_features$t())
    C_Y <- torch_matmul(Y_features, Y_features$t())
    torch_trace(torch_matmul(C_X, C_Y))
  }
)

setGeneric("bottleneck_loss", function(object, X_features, T_features, Y_features) {
  standardGeneric("bottleneck_loss")
})

setMethod(
  "bottleneck_loss",
  signature(object = "InformationBottleneck"),
  function(object, X_features, T_features, Y_features) {
    # Implements equation (10)
    I_XT <- forward_IB(object, X_features, T_features)
    I_TY <- forward_IB(object, T_features, Y_features)
    I_XT - object@beta * I_TY
  }
)

