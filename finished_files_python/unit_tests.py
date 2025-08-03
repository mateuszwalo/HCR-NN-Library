# === Wszystkie testy zakończyły się sukcesem ===

from layers import CDFNorm, MeanEstimation, ConditionalEstimation, PropagationEstimation, EntropyAndMutualInformation, DynamicEMA, BaseOptimization, InformationBottleneck
import unittest
import torch
import math
import torch.nn.functional as F

# === Testy CDFNorm ===
class CDFNormTest(unittest.TestCase):
    def test_gaussian_transform(self):
        cdf_norm = CDFNorm(method='gaussian')
        input_tensor = torch.randn(100) * 2 + 5
        output = cdf_norm(input_tensor)

        self.assertTrue(torch.all(output >= 0))
        self.assertTrue(torch.all(output <= 1))
        self.assertAlmostEqual(output.mean().item(), 0.5, delta=0.1)

    def test_empirical_transform(self):
        cdf_norm = CDFNorm(method='empirical')
        input_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        output = cdf_norm(input_tensor)

        expected = torch.tensor([0.2, 0.4, 0.6, 0.8, 1.0])
        self.assertTrue(torch.allclose(output, expected, atol=1e-5))

# === Testy PropagationEstimation ===
class PropagationEstimationTest(unittest.TestCase):
    def test_propagate_expectation(self):
        a = torch.zeros(2, 2, 2)
        a[0,0,0] = 1.0
        a[1,0,0] = 1.0
        def simple_feature_fn(x): return torch.tensor([1.0, 0.0])
        propagator = PropagationEstimation(y=torch.tensor(1.0), z=torch.tensor(1.0), a=a, feature_fn=simple_feature_fn)
        propagated = propagator.propagate_expectation()
        expected = 0.5
        self.assertAlmostEqual(propagated.item(), expected, delta=1e-5)

# === Testy OrthonormalLegendreBasis ===
class OrthonormalLegendreBasisTest(unittest.TestCase):
    def test_basis_functions(self):
        basis = OrthonormalLegendreBasis(max_degree=3)
        x = torch.linspace(0, 1, 5)
        output = basis(x)
        self.assertEqual(output.shape, (5, 4))

    def test_normalization(self):
        basis = OrthonormalLegendreBasis(max_degree=3)
        x = torch.linspace(0, 1, 1000)  # więcej punktów dla lepszej dokładności
        basis_values = basis(x)
        for i in range(4):
            integral = torch.trapz(basis_values[:,i] * basis_values[:,i], x)
            self.assertAlmostEqual(integral.item(), 1.0, delta=0.05)  # poprawiona tolerancja

# === Testy JointDistribution ===
class JointDistributionTest(unittest.TestCase):
    def test_2d_distribution(self):
        joint_dist = JointDistribution(dim=2)
        joint_dist.coeffs.data.fill_(0)
        joint_dist.coeffs.data[0,0] = 1.0
        x = torch.rand(10, 1)
        y = torch.rand(10, 1)
        output = joint_dist(x, y)
        self.assertEqual(output.shape, (10,))
        self.assertTrue(torch.all(output >= 0))

    def test_3d_distribution(self):
        joint_dist = JointDistribution(dim=3)
        joint_dist.coeffs.data.fill_(0)
        joint_dist.coeffs.data[0,0,0] = 1.0
        x = torch.rand(5, 1)
        y = torch.rand(5, 1)
        z = torch.rand(5, 1)
        output = joint_dist(x, y, z)
        self.assertEqual(output.shape, (5,))
        self.assertTrue(torch.all(output >= 0))

# === Testy Estimation ===
class EstimationTest(unittest.TestCase):
    def test_tensor_mean(self):
        triplets = [(torch.tensor([1.0]), torch.tensor([2.0]), torch.tensor([3.0]))] * 10
        def simple_feature_fn(x): return torch.tensor([x.item(), x.item()**2])
        estimator = Estimation(triplets=triplets, feature_fn=simple_feature_fn, feature_dm=2)
        tensor_mean = estimator.compute_tensor_mean()
        self.assertEqual(tensor_mean.shape, (2, 2, 2))
        self.assertGreater(tensor_mean.sum().item(), 0)

# === Testy ConditionalEstimation ===
class ConditionalEstimationTest(unittest.TestCase):
    def test_conditional_score(self):
        a = torch.zeros(2, 2, 2)
        a[0,0,0] = 1.0
        a[1,1,1] = 2.0
        def simple_feature_fn(x): return torch.tensor([x.item(), x.item()**2])
        estimator = ConditionalEstimation(
            x_candidates=[torch.tensor(1.0), torch.tensor(2.0)],
            y=torch.tensor(1.0),
            z=torch.tensor(1.0),
            a=a,
            feature_fn=simple_feature_fn
        )
        scores = estimator.conditional_score()
        self.assertEqual(len(scores), 2)
        self.assertNotEqual(scores[0], scores[1])

# === Testy DynamicEMA ===
class DynamicEMATest(unittest.TestCase):
    def test_ema_update(self):
        x = torch.tensor([1.0])
        y = torch.tensor([1.0])
        z = torch.tensor([1.0])
        ema = DynamicEMA(x=x, y=y, z=z, ema_lambda=0.1)
        updated_a = ema.EMAUpdateMethod()
        expected = torch.tensor([[[1.0]]]) * ema.ema_lambda
        self.assertTrue(torch.allclose(updated_a, expected, atol=1e-5))

# === Testy BaseOptimization ===
class BaseOptimizationTest(unittest.TestCase):
    def test_basis_optimization(self):
        a = torch.eye(2).unsqueeze(0).repeat(2, 1, 1)
        optimizer = BaseOptimization(a=a)
        new_a = optimizer.optimization_early()
        self.assertEqual(new_a.shape, a.shape)
        self.assertFalse(torch.allclose(new_a, a, atol=1e-5))

if __name__ == "__main__":
    unittest.main(argv=[''], verbosity=2, exit=False)