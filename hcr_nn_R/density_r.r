library(torch)

# Clamp density values to avoid zeros
clamp_density <- function(rho, eps = 1e-6) {
  torch_clamp(rho, min = eps)
}

# Compute joint density (2D Hermite-Chebyshev-like representation)
joint_density <- function(u, coeffs, basis_vals) {
  # basis_vals shape (..., 2, deg+1)
  # Equivalent to: einsum('...i,ij,...j->...', basis_vals[...,0,:], coeffs, basis_vals[...,1,:])
  torch_einsum("...i,ij,...j->...", list(
    basis_vals[ , 1, ],
    coeffs,
    basis_vals[ , 2, ]
  ))
}

# Conditional density p(u1 | u2)
conditional_density <- function(u1_grid, u2_scalar, coeffs, basis, deg, eps = 1e-6) {
  dtype <- u1_grid$dtype
  device <- u1_grid$device

  # Basis for u2
  u2 <- torch_tensor(c(u2_scalar), dtype = dtype, device = device)
  f2 <- basis(u2)$reshape(c(deg + 1))

  # Basis for each u1 on grid
  f1 <- basis(u1_grid)  # (G, deg+1)

  # Joint density: einsum('i,ij,gj->g', f2, coeffs, f1)
  rho <- torch_einsum("i,ij,gj->g", list(f2, coeffs, f1))
  rho <- clamp_density(rho, eps)

  # Normalize via trapezoidal rule
  du <- 1.0 / (u1_grid$size()[1] - 1)
  Z <- rho$sum() * du
  rho / Z
}

# Expected value E[u1 | u2]
expected_u1_given_u2 <- function(u2_scalar, coeffs, basis, deg, grid_size = 200, eps = 1e-6) {
  dtype <- coeffs$dtype
  device <- coeffs$device

  u1_grid <- torch_linspace(0, 1, grid_size, dtype = dtype, device = device)
  p <- conditional_density(u1_grid, u2_scalar, coeffs, basis, deg, eps)

  du <- 1.0 / (grid_size - 1)
  val <- (u1_grid * p)$sum() * du
  as.numeric(val)
}
