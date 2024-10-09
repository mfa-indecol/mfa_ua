import unittest
import matplotlib.pyplot as plt

from mfa_ua.monte_carlo.parameters import ScalarParameter, ConstantParameter
from mfa_ua.monte_carlo.sampler import Sampler
from mfa_ua.monte_carlo.monte_carlo import MonteCarlo


class MonteCarloTester(unittest.TestCase):

    def set_up_sampler(self):
        scalar1 = ScalarParameter("para1", "p1", "unit1", "uniform", 0, 1)
        scalar2 = ScalarParameter("para2", "p2", "unit2", "norm", 0, 1)
        scalar3 = ScalarParameter("para4", "p4", "unit4", "norm", 10, 1)
        constant = ConstantParameter("para3", "p3", "reason", 4)

        sampler = Sampler([scalar1, scalar2, scalar3, constant])
        sampler.sample(samplesize=1000, mode="MC")
        self.sampler = sampler

    def set_up_function(self):
        def function(inputs: list):
            p1, p2, p4, p3 = inputs
            result1 = p1 * p2 + p4 * p3
            result2 = p1 - p2 * p4
            return [result1, result2], ["result1", "result2"]

        return function

    def test_init(self):
        self.set_up_sampler()
        function = self.set_up_function()
        mc = MonteCarlo(function, self.sampler)
        self.assertIsInstance(mc, MonteCarlo)

    def test_analyze(self):
        self.set_up_sampler()
        function = self.set_up_function()
        mc = MonteCarlo(function, self.sampler)
        mc.analyze(iterations=1000, visualisations=False, show_progress_bar=False)
        self.assertEqual(len(mc.result_sets), 1000)
        self.assertEqual(len(mc.result_lists), 2)

    def test_histogram(self):
        self.set_up_sampler()
        function = self.set_up_function()
        mc = MonteCarlo(function, self.sampler)
        mc.analyze(iterations=1000, visualisations=False, show_progress_bar=False)
        fig, ax = mc.histogram("result1")
        self.assertIsInstance(fig, plt.Figure)
        self.assertIsInstance(ax, plt.Axes)
        self.assertGreater(len(ax.patches), 0)

    def test_analyze_with_auto_visualisations(self):
        self.set_up_sampler()
        function = self.set_up_function()
        mc = MonteCarlo(function, self.sampler)
        mc.analyze(iterations=1000, visualisations=True, show_progress_bar=False)
        self.assertEqual(len(mc.figures), 2)


if __name__ == "__main__":
    unittest.main()
