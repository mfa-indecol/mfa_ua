import unittest
import matplotlib.pyplot as plt
from mfa_ua.monte_carlo.parameters import ScalarParameter, ConstantParameter


class TestScalarParameter(unittest.TestCase):
    name = "test"
    short_name = "t"
    unit = "t_unit"

    def test_init(self):
        distribution = "uniform"
        para1 = 0
        para2 = 1
        parameter = ScalarParameter(
            self.name, self.short_name, self.unit, distribution, para1, para2
        )
        self.assertIsInstance(parameter, ScalarParameter)

    def test_unavailable_distribution(self):
        # check that a  ValueError is raised if an unavailable distribution is passed
        distribution = "unavailable"

        with self.assertRaises(ValueError):
            ScalarParameter(self.name, self.short_name, self.unit, distribution, 0, 1)

    def test_insuffient_parameters(self):
        # check that a  ValueError is raised if insufficient parameters are passed
        distribution = "uniform"

        with self.assertRaises(ValueError):
            ScalarParameter(self.name, self.short_name, self.unit, distribution, 0)

    def test_sampling(self):
        distribution = "uniform"
        para1 = 0
        para2 = 1
        parameter = ScalarParameter(
            self.name, self.short_name, self.unit, distribution, para1, para2
        )
        samples = parameter.sample(10)
        self.assertEqual(len(samples), 10)

    def test_truncation(self):
        # check that if we manually set the truncation,
        # the resulting samples are within the truncation
        distribution = "norm"
        para1 = 0
        para2 = 1
        low, up = [0, 1]
        parameter = ScalarParameter(
            self.name,
            self.short_name,
            self.unit,
            distribution,
            para1,
            para2,
            low_lim=low,
            up_lim=up,
        )
        samples = parameter.sample(100)
        self.assertTrue(all([low <= sample <= up for sample in samples]))

    def test_sample_plot(self):
        # check that we get a plt figure back when making a plot,
        # and that there is a histogram there
        distribution = "uniform"
        para1 = 0
        para2 = 1
        parameter = ScalarParameter(
            self.name, self.short_name, self.unit, distribution, para1, para2
        )
        fig, ax = parameter.plot_samples(show=False)
        self.assertIsInstance(fig, plt.Figure)
        self.assertIsInstance(ax, plt.Axes)
        self.assertGreater(len(ax.patches), 0)

    def test_pdf_plot(self):
        # check that we get a plt figure back when making a plot,
        # and that there is a histogram there
        distribution = "uniform"
        para1 = 0
        para2 = 1
        parameter = ScalarParameter(
            self.name, self.short_name, self.unit, distribution, para1, para2
        )
        fig, ax = parameter.plot_distribution(show=False)
        self.assertIsInstance(fig, plt.Figure)
        self.assertIsInstance(ax, plt.Axes)
        self.assertGreater(len(ax.lines), 0)


def test_get_pdf_return_length(self):
    # check that we get arrays of the right length back
    distribution = "uniform"
    para1 = 0
    para2 = 1
    parameter = ScalarParameter(
        self.name, self.short_name, self.unit, distribution, para1, para2
    )
    n_pdf_points = 100
    x, y = parameter.get_pdf(n_x=n_pdf_points)
    self.assertEqual(len(x), n_pdf_points)
    self.assertEqual(len(y), n_pdf_points)


def test_get_pdf_range(self):
    # check that the x_array is within the range we specify
    distribution = "uniform"
    para1 = 0
    para2 = 1
    parameter = ScalarParameter(
        self.name, self.short_name, self.unit, distribution, para1, para2
    )
    n_pdf_points = 100

    range_x = [0, 0.5]
    x, y = parameter.get_pdf(n_x=n_pdf_points)
    self.assertTrue(all([range_x[0] <= x_val <= range_x[1] for x_val in x]))


def test_get_pdf_autorange(self):
    # ccheck that for a unifrom distribution, we get  only
    # values back for where the pdf is non-zero
    distribution = "uniform"
    para1 = 0
    para2 = 1
    # => pdf is non-0 between 0 and 0+1
    parameter = ScalarParameter(
        self.name, self.short_name, self.unit, distribution, para1, para2
    )
    n_pdf_points = 100
    x, y = parameter.get_pdf(n_x=n_pdf_points)
    self.assertTrue(all([0 <= x_val <= 1 for x_val in x]))
    self.assertTrue(all([y_val == 1 for y_val in y]))


class TestConstantParameter(unittest.TestCase):
    def test_init_constant(self):
        name = "test"
        short_name = "t"
        explanation = "test explanation"
        value = 0
        parameter = ConstantParameter(name, short_name, explanation, value)
        self.assertIsInstance(parameter, ConstantParameter)

    def test_sampling(self):
        name = "test"
        short_name = "t"
        explanation = "test explanation"
        value = 0
        parameter = ConstantParameter(name, short_name, explanation, value)
        samples = parameter.sample(10)
        self.assertEqual(len(samples), 10)


if __name__ == "__main__":
    unittest.main()
