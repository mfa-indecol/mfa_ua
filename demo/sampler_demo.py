# %%
from mfa_ua.monte_carlo.parameters import ScalarParameter, ConstantParameter
from mfa_ua.monte_carlo.sampler import Sampler

if __name__ == "__main__":
    # init two scalar parameters
    x = ScalarParameter("x", "x", "-", "uniform", 0, 1)
    y = ScalarParameter("y", "y", "-", "norm", 0, 1)
    z = ScalarParameter("z", "z", "-", "triang", 0.5, 1, 4)

    sampler = Sampler([x, y, z])
    samples = sampler.sample(samplesize=100)

    sampler.plot_2D("x", "y")

    sampler.plot_2D("x", "z")

    sampler.plot_3D("x", "y", "z")
# %%
