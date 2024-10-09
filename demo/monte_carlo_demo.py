# %%
from mfa_ua.monte_carlo.parameters import ScalarParameter, ConstantParameter
from mfa_ua.monte_carlo.sampler import Sampler
from mfa_ua.monte_carlo.monte_carlo import MonteCarlo


if __name__ == "__main__":
    # init two scalar parameters
    x = ScalarParameter("x", "x", "-", "uniform", 0, 1)
    y = ScalarParameter("y", "y", "-", "norm", 0, 1)
    z = ScalarParameter("z", "z", "-", "triang", 0.5, 1, 4)

    sampler = Sampler([x, y, z])
    samples = sampler.sample(samplesize=1000)

    def some_function(inputs: list):
        x, y, z = inputs
        result1 = x * y + z
        result2 = x - y * z
        return [result1, result2], ["result1", "result2"]

    monte_carlo = MonteCarlo(some_function, sampler)
    monte_carlo.analyze(iterations=10**7, visualisations=False, show_progress_bar=True)
    monte_carlo.analyze(iterations=10**3, visualisations=True, show_progress_bar=True)
# %%
