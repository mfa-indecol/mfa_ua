import unittest
import matplotlib.pyplot as plt
from mfa_ua.monte_carlo.parameters import ScalarParameter, ConstantParameter
from mfa_ua.monte_carlo.sampler import Sampler


class TestSampler(unittest.TestCase):
    def set_parameters(self):
        # create two scalar parameters and one constant parameter
        scalar1 = ScalarParameter("para1", "p1", "unit1", "uniform", 0, 1)
        scalar2 = ScalarParameter("para2", "p2", "unit2", "norm", 0, 1)
        constant = ConstantParameter("para3", "p3", "reason", 4)
        return [scalar1, scalar2, constant]

    def test_init(self):
        scalar1, scalar2, constant = self.set_parameters()
        sampler = Sampler([scalar1, scalar2, constant])
        self.assertIsInstance(sampler, Sampler)

    def test_sample_dimensions(self):
        scalar1, scalar2, constant = self.set_parameters()
        sampler = Sampler([scalar1, scalar2, constant])
        samples = sampler.sample(samplesize=10, mode="MC")
        self.assertEqual(len(samples), 10)
        self.assertEqual(len(samples[0]), 3)

        samples_LHS = sampler.sample(samplesize=10, mode="LHS")
        self.assertEqual(len(samples_LHS), 10)
        self.assertEqual(len(samples_LHS[0]), 3)

    def test_LHS_grid(self):
        # TODO: implement test for LHS grid
        pass

    def test_2D_plot(self):
        scalar1, scalar2, constant = self.set_parameters()
        sampler = Sampler([scalar1, scalar2, constant])
        samples = sampler.sample(samplesize=100, mode="MC")
        fig, ax = sampler.plot_2D("p1", "p2", show=False)
        self.assertIsInstance(fig, plt.Figure)
        self.assertIsInstance(ax, plt.Axes)
        # check that there is some data in the plot
        self.assertTrue(len(ax.collections) > 0)

    def test_3D_plot(self):
        scalar1, scalar2, constant = self.set_parameters()
        scalar3 = ScalarParameter("para4", "p4", "unit4", "norm", 10, 1)
        sampler = Sampler([scalar1, scalar2, scalar3])
        samples = sampler.sample(samplesize=100, mode="MC")
        fig, ax = sampler.plot_3D("p1", "p2", "p4", show=False)
        self.assertIsInstance(fig, plt.Figure)
        self.assertIsInstance(ax, plt.Axes)
        # check that there is some data in the plot
        self.assertTrue(len(ax.collections) > 0)


if __name__ == "__main__":
    unittest.main()
