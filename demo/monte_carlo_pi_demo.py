# %%
from matplotlib import pyplot as plt
from mfa_ua.monte_carlo.monte_carlo import MonteCarlo
from mfa_ua.monte_carlo.sampler import Sampler
from mfa_ua.monte_carlo.parameters import ScalarParameter

if __name__ == "__main__":
    # init two scalar parameters
    x = ScalarParameter("x", "x", "-", "uniform", 0, 1)
    y = ScalarParameter("y", "y", "-", "uniform", 0, 1)
    # init the sampler
    sampler = Sampler([x, y])
    # sample the parameters
    sampler.sample(mode="MC", samplesize=10**6)

    def inside_circle(p: list[float]) -> tuple[list[float], list[str]]:
        x, y = p
        return [int(x**2 + y**2 <= 1)], ["inside"]

    monte_carlo = MonteCarlo(inside_circle, sampler)

    monte_carlo.analyze()
    fig = monte_carlo.histogram("inside")

    # with the results of how many iterations are inside the circle, we can estimate pi:
    # the area of the circle is pi * r^2, we checked for a circle with radius 1, so the area is pi
    # but we only sampled for a fourth of the circle, so the area is pi / 4
    # we can get the ratio of how many points are inside the circle to the total number of points
    # and this ratio should be pi / 4 - so we can estimate pi by multiplying this ratio by 4
    pi = 4 * sum(monte_carlo.result_lists[0]) / monte_carlo.iterations
    print(f"Estimated pi: {pi}")

    # we can try with an increasing number of iterations:
    for i in range(7):
        monte_carlo.analyze(iterations=10**i, show_progress_bar=False)
        pi = 4 * sum(monte_carlo.result_lists[0]) / monte_carlo.iterations
        print(f"Estimated pi with {10**i} iterations: {pi}")

# %%
