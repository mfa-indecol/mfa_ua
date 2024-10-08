# %%
from mfa_ua.monte_carlo.parameters import ScalarParameter, ConstantParameter

if __name__ == "__main__":
    # init two scalar parameters
    x = ScalarParameter("x", "x", "-", "uniform", 0, 1)
    y = ScalarParameter("y", "y", "-", "norm", 0, 1)

    # plot samples from both parameters
    x.plot_samples(1000)
    y.plot_samples(1000)

    # plot the distribution:
    x.plot_distribution()
    y.plot_distribution()

# %%
