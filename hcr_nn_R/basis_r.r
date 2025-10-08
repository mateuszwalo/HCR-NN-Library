library(torch)

# --- PolynomialBasis --------------------------------------------------------
PolynomialBasis <- PolynomialBasis(
  initialize = function(degree) {
    self$degree <- degree
  },

  forward = function(u) {
    dtype <- u$dtype
    device <- u$device
    u_flat <- u$reshape(c(-1))
    x <- 2 * u_flat - 1  # map [0,1] -> [-1,1]
    x_np <- as.numeric(x$to(device = "cpu"))
    
    # compute shifted Legendre polynomials
    out_list <- list()
    for (i in 0:self$degree) {
      Pi_np <- legendreP(i, x_np)  # base R Legendre polynomial
      Pi <- torch_tensor(Pi_np, dtype = dtype, device = device)
      scale <- torch_sqrt(torch_tensor(2 * i + 1, dtype = dtype, device = device))
      out_list[[length(out_list) + 1]] <- scale * Pi
    }
    
    F <- torch_stack(out_list, dim = -1)
    F$reshape(c(u$shape, self$degree + 1))
  }
)

# --- CosineBasis ------------------------------------------------------------
CosineBasis <- nn_module(
  "CosineBasis",
  initialize = function(degree) {
    self$degree <- degree
  },
  forward = function(u) {
    dtype <- u$dtype
    device <- u$device
    u_flat <- u$reshape(c(-1))
    out_list <- list(torch_ones_like(u_flat, dtype = dtype, device = device))
    for (j in 1:self$degree) {
      coef <- torch_sqrt(torch_tensor(2.0, dtype = dtype, device = device))
      out_list[[length(out_list) + 1]] <- coef * torch_cos(torch_pi * j * u_flat)
    }
    F <- torch_stack(out_list, dim = -1)
    F$reshape(c(u$shape, self$degree + 1))
  }
)

# --- KDEBasis ---------------------------------------------------------------
# Empirical KDE basis:
#   f_k(u) = Gaussian(u; center=u_k, bandwidth=sigma)
KDEBasis <- nn_module(
  "KDEBasis",
  initialize = function(centers, bandwidth) {
    centers <- centers$reshape(c(-1))
    if (centers$numel() == 0)
      rlang::abort("centers must be non-empty")
    if (bandwidth <= 0)
      rlang::abort("bandwidth must be > 0")
    self$register_buffer("centers", centers)
    self$bandwidth <- bandwidth
  },
  forward = function(u) {
    dtype <- u$dtype
    device <- u$device
    u_flat <- u$reshape(c(-1, 1)) # (N, 1)
    centers <- self$centers$reshape(c(1, -1))
    diff <- (u_flat - centers$to(device = device)) / self$bandwidth
    two_pi <- torch_tensor(2 * pi, dtype = dtype, device = device)
    norm <- self$bandwidth * torch_sqrt(two_pi)
    K <- torch_exp(-0.5 * diff^2) / norm
    K$reshape(c(u$shape, -1))
  }
)

# --- Basis selector ---------------------------------------------------------
select_basis <- function(name, degree = NULL, ...) {
  name <- tolower(name)
  args <- list(...)
  
  if (name == "polynomial") {
    if (is.null(degree))
      stop("Polynomial basis requires 'degree'")
    PolynomialBasis(degree)
    
  } else if (name == "cosine") {
    if (is.null(degree))
      stop("Cosine basis requires 'degree'")
    CosineBasis(degree)
    
  } else if (name == "kde") {
    centers <- args$centers
    bandwidth <- if (!is.null(args$bandwidth)) args$bandwidth else 0.05
    if (is.null(centers))
      stop("KDEBasis requires 'centers' tensor")
    KDEBasis(centers, bandwidth)
    
  } else {
    stop(paste("Unknown basis type:", name))
  }
}