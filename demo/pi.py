# %%
from matplotlib import pyplot as plt
from mfa_ua.monte_carlo.monte_carlo import MonteCarlo
from mfa_ua.monte_carlo.sampling import Sampler
from mfa_ua.monte_carlo.parameters import ScalarParameter

if __name__ == "__main__":
    # init two scalar parameters
    x = ScalarParameter("x", "x", "-", "uniform", 0, 1)
    y = ScalarParameter("y", "y", "-", "uniform", 0, 1)
    # init the sampler
    sampler = Sampler([x, y])
    # sample the parameters
    sampler.sample(mode="MC", samplesize=1000)
    sampler.plot_2D(x, y)

    def inside_circle(p: list[float]) -> tuple[list[float], list[str]]:
        x, y = p
        return [int(x**2 + y**2 <= 1)], ["inside"]

    monte_carlo = MonteCarlo(
        inside_circle, sampler, iterations=1000, visualisations=False
    )

    monte_carlo.analyze()
    fig = monte_carlo.histogram("inside")
    # fig.show()
    # figs[0].show()


# %%
def inside_circle(p: list[float]):
    x, y = p
    return x**2 + y**2 <= 1


x = 2
y = 3
print(inside_circle([x, y]))

# %%
