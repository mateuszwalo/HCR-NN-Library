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

