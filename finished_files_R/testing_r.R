source('./layers.r')

library(torch)
library(testthat)

# CDF Testing
norm_layer <- CDFNorm(method = "gaussian")

x <- torch_randn(10)
y <- norm_layer(x)

x <- torch_tensor(c(-2, -1, 0, 1, 2))

norm_gaussian <- CDFNorm(method = "gaussian", affine = FALSE)
out_gaussian <- norm_gaussian(x)
print(out_gaussian)

norm_empirical <- CDFNorm(method = "empirical", affine = FALSE)
out_empirical <- norm_empirical(x)
print(out_empirical)

norm_affine <- CDFNorm(method = "gaussian", affine = TRUE)
norm_affine$weight <- nn_parameter(torch_tensor(2))
norm_affine$bias <- nn_parameter(torch_tensor(-1))

out_affine <- norm_affine(torch_randn(5))
print(out_affine)

norm_stats <- CDFNorm(method = "gaussian", track_running_stats = TRUE)

norm_stats$train()
norm_stats(torch_randn(10))

print(norm_stats$running_mean)
print(norm_stats$running_var)

norm_stats$eval()
norm_stats(torch_randn(10))


test_that("Gaussian CDF outputs between 0 and 1", {
  x <- torch_randn(100)
  norm <- CDFNorm(method = "gaussian")
  y <- norm(x)
  expect_true(all(y >= 0 & y <= 1))
})

test_that("Empirical CDF outputs between 0 and 1", {
  x <- torch_randn(100)
  norm <- CDFNorm(method = "empirical")
  y <- norm(x)
  expect_true(all(y >= 0 & y <= 1))
})

test_that("Affine parameters affect output", {
  x <- torch_randn(10)
  norm <- CDFNorm(method = "gaussian", affine = TRUE)
  norm$weight <- nn_parameter(torch_tensor(2))
  norm$bias <- nn_parameter(torch_tensor(1))
  y <- norm(x)
  expect_true(all(y > 1))
})

#Orthonornal basis test
basis <- OrthonormalLegendreBasis(max_degree = 3)
x <- torch_linspace(0, 1, steps = 5)
y <- basis(x)

print(dim(y))
print(y)

test_that("Known values are correct", {
  basis <- OrthonormalLegendreBasis(max_degree = 3)
  x <- torch_tensor(c(0, 0.5, 1))
  y <- basis(x) * torch_tensor(c(1.0, sqrt(3), sqrt(5), sqrt(7)))
  
  expected <- torch_tensor(matrix(c(
    1, -1,  1,  -1,    # x = 0
    1,  0, -0.5, 0,    # x = 0.5
    1,  1,  1,   1     # x = 1
  ), nrow = 3, byrow = TRUE))
  
  expect_true(torch_allclose(y, expected, atol = 1e-6))
})

#shape tests
test_that("Output shape and clamping", {
  basis <- OrthonormalLegendreBasis(max_degree = 2)
  x <- torch_tensor(c(-1, 0.5, 2))
  y <- basis(x)
  
  expect_equal(dim(y), c(3, 3)) # max_degree=2 -> 3 basis funcs
  expect_true(all(y[1, ] == y[1, ])) # No NaN
})

#orthonormality test
test_that("Basis is orthonormal", {
  basis <- OrthonormalLegendreBasis(max_degree = 3)
  x <- torch_linspace(0, 1, steps = 1000)
  Y <- basis(x) # [1000, 4]
  
  inner <- torch_matmul(Y$t(), Y) / 1000
  eye <- torch_eye(4)
  
  expect_true(torch_allclose(inner, eye, atol = 1e-2))
})

test_that("2D JointDistribution outputs correct shape", {
  jd <- JointDistribution(dim = 2, basis_size = 4)
  x <- torch_rand(5) # 5 samples
  y <- torch_rand(5)
  out <- jd(x, y)
  
  expect_equal(out$shape, c(5))
  expect_true(torch_is_tensor(out))
})

test_that("3D JointDistribution outputs correct shape", {
  jd <- JointDistribution(dim = 3, basis_size = 3)
  x <- torch_rand(2)
  y <- torch_rand(2)
  z <- torch_rand(2)
  out <- jd(x, y, z)
  
  expect_equal(out$shape, c(2))
})

test_that("Known coefficients produce expected output (2D constant 1)", {
  jd <- JointDistribution(dim = 2, basis_size = 2)
  # Set coeffs so that only constant term (i=0, j=0) is 1
  coeffs <- torch_zeros(2, 2)
  coeffs[1, 1] <- 1
  jd$coeffs <- nn_parameter(coeffs)
  
  x <- torch_linspace(0, 1, steps = 3)
  y <- torch_linspace(0, 1, steps = 3)
  out <- jd(x, y)
  
  expect_true(torch_allclose(out, torch_ones(3), atol = 1e-6))
})

test_that("Known coefficients produce separable product output (2D)", {
  jd <- JointDistribution(dim = 2, basis_size = 2)

  coeffs <- torch_zeros(2, 2)
  coeffs[2, 1] <- 1
  jd$coeffs <- nn_parameter(coeffs)
  
  x <- torch_tensor(c(0, 0.5, 1))
  y <- torch_tensor(c(0, 0.5, 1))
  

  P1_x <- jd$basis(x)[, 2]
  expected <- P1_x 
  out <- jd(x, y)
  
  expect_true(torch_allclose(out, expected, atol = 1e-6))
})

test_that("Invalid dim triggers error", {
  expect_error(JointDistribution(dim = 4, basis_size = 3)(torch_rand(1), torch_rand(1)))
})

test_that("Scalar inputs are handled correctly", {
  jd <- JointDistribution(dim = 2, basis_size = 2)
  coeffs <- torch_zeros(2, 2)
  coeffs[1, 1] <- 1
  jd$coeffs <- nn_parameter(coeffs)
  
  out <- jd(0.5, 0.5)
  expect_true(torch_allclose(out, torch_tensor(1), atol = 1e-6))
})

test_that("Gradients flow through coeffs", {
  jd <- JointDistribution(dim = 2, basis_size = 2)
  x <- torch_rand(5)
  y <- torch_rand(5)
  out <- jd(x, y)
  loss <- out$sum()
  loss$backward()
  
  grad_norm <- jd$coeffs$grad()$abs()$sum()$item()
  expect_true(grad_norm > 0)
})


id_feature_fn <- function(x) c(x, x^2)

test_that("Output has correct dimensions", {
  triplets <- list(list(1, 2, 3))
  m <- MeanEstimation(triplets, id_feature_fn, feature_dm = 2)
  out <- m$compute_tensor_mean()
  
  expect_equal(dim(out), c(2, 2, 2))
})

test_that("Single triplet produces correct outer product", {
  triplets <- list(list(1, 2, 3))
  m <- MeanEstimation(triplets, id_feature_fn, feature_dm = 2)
  out <- m$compute_tensor_mean()
  
  fx <- id_feature_fn(1) # c(1, 1)
  fy <- id_feature_fn(2) # c(2, 4)
  fz <- id_feature_fn(3) # c(3, 9)
  
  expected <- array(0, dim = c(2, 2, 2))
  for (i in 1:2) {
    for (j in 1:2) {
      for (k in 1:2) {
        expected[i, j, k] <- fx[i] * fy[j] * fz[k]
      }
    }
  }
  
  expect_equal(out, expected)
})

test_that("Averaging works across multiple triplets", {
  triplets <- list(list(1, 1, 1), list(2, 2, 2))
  m <- MeanEstimation(triplets, id_feature_fn, feature_dm = 2)
  out <- m$compute_tensor_mean()
  
  fx1 <- id_feature_fn(1)
  fy1 <- id_feature_fn(1)
  fz1 <- id_feature_fn(1)
  
  fx2 <- id_feature_fn(2)
  fy2 <- id_feature_fn(2)
  fz2 <- id_feature_fn(2)
  
  outer1 <- array(0, dim = c(2, 2, 2))
  outer2 <- array(0, dim = c(2, 2, 2))
  for (i in 1:2) {
    for (j in 1:2) {
      for (k in 1:2) {
        outer1[i, j, k] <- fx1[i] * fy1[j] * fz1[k]
        outer2[i, j, k] <- fx2[i] * fy2[j] * fz2[k]
      }
    }
  }
  
  expected <- (outer1 + outer2) / 2
  expect_equal(out, expected)
})

test_that("Feature function length matches feature_dm", {
  triplets <- list(list(1, 2, 3))
  wrong_feature_fn <- function(x) c(x) # length 1 vector
  m <- MeanEstimation(triplets, wrong_feature_fn, feature_dm = 2)
  
  expect_error({
    out <- m$compute_tensor_mean()
  }, regexp = "subscript out of bounds|length mismatch", fixed = FALSE)
})

id_feature_fn <- function(x) {
  if (is.numeric(x)) {
    return(rep(x, 2))
  } else {
    stop("Feature fn expects numeric")
  }
}

test_that("ConditionalEstimation returns correct score for simple input", {
  D <- 2
  a <- array(1, dim = c(D, D, D))
  
  ce <- ConditionalEstimation(
    x_candidates = c(1, 2),
    y = 3,
    z = 4,
    a = a,
    feature_fn = id_feature_fn
  )
  
  scores <- ce$conditional_score()
  
  fy <- id_feature_fn(3) # c(3,3)
  fz <- id_feature_fn(4) # c(4,4)
  denom <- sum(1 * outer(fy, fz))
  
  expected_scores <- c()
  for (x in c(1, 2)) {
    fx <- id_feature_fn(x)
    score <- 0
    for (i in 1:D) {
      context_sum <- sum(outer(fy, fz))
      score <- score + fx[i] * (context_sum / (denom + 1e-8))
    }
    expected_scores <- c(expected_scores, score)
  }
  
  expect_equal(scores, expected_scores)
})

test_that("PropagationEstimation computes correct ratio", {
  D <- 2
  a <- torch_ones(D, D, D)
  
  pe <- PropagationEstimation(
    y = 3,
    z = 4,
    a = a,
    feature_fn = function(x) torch_tensor(rep(as.numeric(x), D))
  )
  
  out <- pe$propagate_expectation()
  expect_true(torch_is_tensor(out))
  expect_equal(out$ndim, 0)
  
  fy <- rep(3, 2)
  fz <- rep(4, 2)
  denom <- sum(outer(fy, fz))
  numer <- denom
  ratio <- numer / (denom + 1e-8)
  centered <- ratio - 1.0
  expected <- 0.5 + (1.0 / (2.0 * sqrt(3.0))) * centered
  
  expect_true(abs(as.numeric(out) - expected) < 1e-6)
})

test_that("PropagationEstimation supports gradients", {
  D <- 2
  a <- torch_ones(D, D, D, requires_grad = TRUE)
  
  pe <- PropagationEstimation(
    y = 3,
    z = 4,
    a = a,
    feature_fn = function(x) torch_tensor(rep(as.numeric(x), D))
  )
  
  out <- pe$propagate_expectation()
  out$backward()
  
  grad_sum <- a$grad()$abs()$sum()$item()
  expect_true(grad_sum > 0)
})

test_that("Entropy is non-negative and decreases for peaked distributions", {
  emi <- EntropyAndMutualInformation()
  
  act_uniform <- torch_ones(2, 4) # batch=2, classes=4
  entropy_uniform <- emi$approximate_entropy(act_uniform)
  
  act_peaked <- torch_tensor(matrix(c(10,0,0,0, 0,10,0,0), nrow=2, byrow=TRUE))
  entropy_peaked <- emi$approximate_entropy(act_peaked)
  
  expect_true(as.numeric(entropy_uniform) > as.numeric(entropy_peaked))
  expect_true(as.numeric(entropy_uniform) >= 0)
})

test_that("Mutual information is higher for correlated activations", {
  emi <- EntropyAndMutualInformation()
  
  act_X <- torch_tensor(matrix(c(5,0,0,0, 0,5,0,0), nrow=2, byrow=TRUE))
  act_Y <- act_X
  mi_corr <- emi$approximate_mutual_information(act_X, act_Y)
  
  set.seed(42)
  act_X_rand <- torch_randn(2,4)
  act_Y_rand <- torch_randn(2,4)
  mi_rand <- emi$approximate_mutual_information(act_X_rand, act_Y_rand)
  
  expect_true(as.numeric(mi_corr) > as.numeric(mi_rand))
})

test_that("DynamicEMA initializes with correct shape and updates", {
  x <- torch_tensor(c(1, 2))
  y <- torch_tensor(c(3, 4))
  z <- torch_tensor(c(5, 6))
  
  ema <- DynamicEMA(x, y, z, ema_lambda = 0.5)
  
  expect_true(torch_allclose(
    ema$a,
    torch_zeros_like(torch_einsum("i,j,k->ijk", x, y, z))
  ))
  
  updated <- ema$EMAUpdateMethod()
  expected_outer <- torch_einsum("i,j,k->ijk", x, y, z)
  expect_true(torch_allclose(updated, expected_outer * 0.5))
  
  prev <- ema$a$clone()
  ema$EMAUpdateMethod()
  expect_true(torch_allclose(
    ema$a,
    prev * 0.5 + expected_outer * 0.5
  ))
})

test_that("BaseOptimization performs SVD and transforms tensor", {
  a <- torch_randn(3, 4, 5)
  bo <- BaseOptimization(a)
  
  new_a <- bo$optimization_early()
  
  expect_true(torch_is_tensor(new_a))
  expect_equal(dim(new_a), dim(a))
  expect_false(torch_allclose(new_a, a))
})

test_that("BaseOptimization works with rank-deficient input", {
  a <- torch_ones(3, 4, 5) # rank-1 matrix
  bo <- BaseOptimization(a)
  
  expect_error_free({
    new_a <- bo$optimization_early()
    expect_equal(dim(new_a), dim(a))
  })
})

test_that("InformationBottleneck forward computes trace correctly", {
  ib <- InformationBottleneck(beta = 2.0)
  
  X <- torch_eye(3)
  Y <- torch_eye(3)
  
  val <- ib$forward(X, Y)
  expect_equal(as.numeric(val), 3)
})

test_that("InformationBottleneck bottleneck_loss matches manual calculation", {
  beta <- 0.5
  ib <- InformationBottleneck(beta = beta)
  
  X <- torch_eye(3)
  T <- torch_eye(3) * 2
  Y <- torch_eye(3) * 3
  
  I_XT <- ib$forward(X, T)
  I_TY <- ib$forward(T, Y)
  
  loss <- ib$bottleneck_loss(X, T, Y)
  expect_equal(as.numeric(loss), as.numeric(I_XT - beta * I_TY))
})

