"""
File name: fit_lognormal.py
Purpose: fit distribution for uncertainty analysis at NTNU IndEcol
Author: Nils Dittrich
Date created: 10.12.2022
Date last modified: 16.02.2023
Python Version: 3.9
Status: completed
"""

import numpy as np
import scipy.stats as sts
import matplotlib.pyplot as plt

from scipy.special import erfinv
from scipy.optimize import fsolve

from typing import Tuple
import warnings
from tqdm.notebook import tqdm


def lognorm_fit(
    true_low: float,
    true_upp: float,
    true_mode: float,
    precision: float = 0.05,
    CI_width: float = 0.95,
    diagnostics: bool = False,
) -> Tuple[bool and float]:
    """
    Fitting the scipy lognormal parameters using scipys fsolve.
    For an input of 3 points representing the lower and upper bound of a
    confidence interval as well as the mode for the lognormal distr..
    These three conditions allow the fit of shape, scale and loc.

    Arguments:
    - true_low: lower bound for the CI of the fitted value.
    - true_upp: upper bound for the CI of the fitted value.
    - true_mode: value with highest probability of the distribution.
                 Closer to the lower bound (45%/90%) than upper bound.
    - precision: maximally accepted deviation that the fitted CI and
                 mode can have to the given points while still counting
                 as a successful fit. This is important when we struggle
                 to find a set of starting points that enables
                 convergence for the fsolver, meaning we might end up
                 with a local optimum and thus suboptimal fit overall.
    - CI_width: how probable should it be that the values of the fitted
                value are in between true low and true upp.
    - diagnostics: whether or not to add  prints and figures during fits

    Returns:
    - works: boolean that tells whether the fit succeeded or not
    - distribution parameters: shape, scale and loc for successful fits,
                               mean and std for unsuccessful fits
    """
    ## Check if the inputs can be used for a lognormal fit at all
    inputs_good = True
    if true_low > true_mode:
        inputs_good = False
        raise Exception(f"mode must be higher than lower point")
    if true_upp < true_mode:
        inputs_good = False
        raise Exception(f"mode must be lower than upper point")
    # assert that the mode is skewed significantly to the left
    if (true_mode - true_low) / (true_upp - true_low) > 0.5 and diagnostics:
        inputs_good = False
        warnings.warn(
            f"Warning: low,upp or mode are not compatible with a lognormal distibution."
        )

    ## Preparing for the fsolve
    inv025 = erfinv(2 * (1 - CI_width) / 2 - 1)
    inv975 = erfinv(2 * (1 + CI_width) / 2 - 1)

    # define equations for the fsolver
    def eqs(p):
        shape, scale, loc = p
        # three equations given by the PDF of lognormal in scipy;
        # one: the integral of the pdf from 0 to low shoould be 2.5%
        # two: the integral of the pdf from 0 to upp should be 97.5 %
        # three: the global max of the pdf should be the mode
        one = np.log((true_low - loc) / scale) / (shape * np.sqrt(2)) - inv025
        two = np.log((true_upp - loc) / scale) / (shape * np.sqrt(2)) - inv975
        three = scale * np.exp(-(shape**2)) + loc - true_mode
        return one, two, three

    ## Set the estimates at a resonable place to start the fsolver
    # The samples for the starting values are generated with a Latin
    # Hypercube Sampler, which ensures that we have even coverage of the
    # space of starting points for the fsolver. The LHS is implemented
    # with scipys LHS for generating the probabilities in all dimensions
    # first, the points are then generated using the ppf function of
    # each distribution (dimension). If the fsolver does not converge on
    # a fit, this could either be due do a poor choice of starting
    # values (widen the range below), or because the grid of the LHS is
    # to rough - in this case, increase the number of starting points
    # (n_tries). There could also be no solution (fit) to your problem.
    n_tries = 10**4
    LHS_sampler = sts.qmc.LatinHypercube(d=3)
    lhs_values = LHS_sampler.random(n_tries)

    shapes_distr = sts.truncnorm
    shapes_mean = 0.5
    shapes_std = 2
    shapes_low = (10**-6 - shapes_mean) / shapes_std
    shapes_upp = (10 * 3 - shapes_mean) / shapes_std
    shapes_inputs = {
        "a": shapes_low,
        "b": shapes_upp,
        "loc": shapes_mean,
        "scale": shapes_std,
    }
    est_shapes = shapes_distr.ppf(lhs_values[:, 0], **shapes_inputs)

    scales_distr = sts.truncnorm
    scales_mean = true_upp - true_low
    scales_std = scales_mean * 2
    scales_low = (10**-6 * (true_upp - true_low) - scales_mean) / scales_std
    scales_upp = (scales_mean + 10 * (true_upp - true_low) - scales_mean) / scales_std
    scales_inputs = {
        "a": scales_low,
        "b": scales_upp,
        "loc": scales_mean,
        "scale": scales_std,
    }
    est_scales = scales_distr.ppf(lhs_values[:, 1], **scales_inputs)

    locs_distr = sts.uniform
    locs_low = true_low - 2 * (true_upp - true_low)
    locs_scale = 2.5 * (true_upp - true_low)
    locs_inputs = {"loc": locs_low, "scale": locs_scale}
    est_locs = locs_distr.ppf(lhs_values[:, 2], **locs_inputs)

    if diagnostics:
        plt.figure()
        plt.hist(est_shapes, label="shapes", alpha=0.6, bins="auto")
        plt.hist(est_scales, label="scales", alpha=0.6, bins="auto")
        plt.hist(est_locs, label="locs", alpha=0.6, bins="auto")
        plt.title("distributions of starting points for fsolver")
        plt.legend()
        plt.show()

    shapes = []
    scales = []
    locs = []
    p025_errors = []
    p975_errors = []
    mode_errors = []

    ## turn of warnings (the fsolver will quickly throw a lot of
    # warnings that we can probably ignore, this way warnings in other
    # parts of the code become more apparent
    if not diagnostics:
        warnings.filterwarnings("ignore")
    distribution_works = False
    best_index = -1
    # one version with, one without tqdm for more difficult fits
    # for est_shape, est_scale, est_loc in tqdm(zip(est_shapes, est_scales, est_locs)):
    for est_shape, est_scale, est_loc in zip(est_shapes, est_scales, est_locs):
        estimates = [est_shape, est_scale, est_loc]
        ## do the solving
        shape, scale, loc = fsolve(eqs, estimates, maxfev=10**6)

        # eliminate negative shapes and scales, but negative locs
        # (the fsolver is not as restricted as the scipy function)
        shape, scale, loc = abs(shape), abs(scale), loc
        shapes.append(shape)
        scales.append(scale)
        locs.append(loc)

        # check the resulting distribution and save errors
        p025 = sts.lognorm.ppf(0.025, s=shape, scale=scale, loc=loc)
        p975 = sts.lognorm.ppf(0.975, s=shape, scale=scale, loc=loc)
        mode = scale * np.exp(-(shape**2)) + loc

        p025_errors.append(abs((p025 - true_low) / (true_upp - true_low)))
        p975_errors.append(abs((p975 - true_upp) / (true_upp - true_low)))
        mode_errors.append(abs((mode - true_mode) / (true_upp - true_low)))

        # accept immediatly if all errors are less than 1% of the allowed error
        perfect_precision = precision / 100
        if p025_errors[-1] < perfect_precision:
            if p975_errors[-1] < perfect_precision:
                if mode_errors[-1] < perfect_precision:
                    distribution_works = True
                    best_index = len(p025_errors) - 1

        if distribution_works or not inputs_good:
            break

    ## turn warnings on again
    if not diagnostics:
        warnings.filterwarnings("default")
    ## if no optimal solution (less than 1% of accepted error) was found, we look for the best
    ## possible set of values by inspecting the values of all sets
    if not distribution_works or not inputs_good:
        # init the search
        best_errors = (
            p025_errors[best_index],
            p975_errors[best_index],
            mode_errors[best_index],
        )
        best_average_error = np.average(best_errors)
        best_min_error = min(best_errors)
        # iterate over all sets of errors
        for i, (p025_error, p975_error, mode_error) in enumerate(
            zip(p025_errors, p975_errors, mode_errors)
        ):
            if sum([p025_error, p975_error, mode_error]) <= sum(best_errors):
                # accept new set if sum of errors is lower and alls errors are less than 2x average
                if (
                    p025_error < best_average_error * 2
                    and p975_error < best_average_error * 2
                    and mode_error < best_average_error * 2
                ):
                    best_errors = p025_error, p975_error, mode_error
                    best_index = i
                    best_average_error = np.average(best_errors)
                    best_min_error = min(best_errors)

    # assign the correct parameters and errors (from the best fit after selection above)
    shape, scale, loc = shapes[best_index], scales[best_index], locs[best_index]
    best_min_error = min(
        [p025_errors[best_index], p975_errors[best_index], mode_errors[best_index]]
    )

    ## return best estimate if it is better than required, return a normal distribution if not
    if best_min_error > precision:
        mean = true_mode
        std = (true_upp - true_low) / 4
        if diagnostics:
            p025 = sts.lognorm.ppf(0.025, s=shape, scale=scale, loc=loc)
            p975 = sts.lognorm.ppf(0.975, s=shape, scale=scale, loc=loc)
            mode = scale * np.exp(-(shape**2)) + loc
            warnings.warn(
                f"Warning: the fit for low:{true_low}, upp:{true_upp}, mode:{true_mode} was not "
                f"successful. The best set of values was shape:{shape}, scale:{scale}, loc:{loc}"
                f" with errors of {mode_errors[best_index]} (mode: {mode} instead of "
                f"{true_mode}), {p025_errors[best_index]} (lower bound: {p025} instead of "
                f"{true_low}), {p975_errors[best_index]} (lower bound: {p975} instead of "
                f"{true_upp}). We instead use a normal distribution instead."
            )

        return False, mean, std
    else:
        return (
            True,
            shape,
            scale,
            loc,
        )
