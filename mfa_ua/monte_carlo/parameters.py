"""
This module sets convenient classes for parameters in the Monte Carlo.
These are used in the Sampler, but can also serve for visualisation.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sts

from pathlib import Path

from mfa_ua.parameter_estimation.right_skewed_lognorm import RightSkewedLognorm


class UncertainEntity:
    """
    Parent class for ScalarParamters and ConstantParameters.
    Includes some basic dunder methods for namuing and comparing.

    Attributes:
        name (str): Long (full) name of the parameter.
        short_name (str): Abbreviation/super short name of the parameter.
        unit (str): Very short unit for plots etc.
    """

    def __init__(self, name: str, short_name: str, unit: str = None):
        self.name = name
        self.short_name = short_name
        self.unit = unit

    def sampling(self, samplesize: int):
        """Necessary function that returns samplesize many samples"""
        return

    def __eq__(self, other):
        """
        We make sure that for the uncertainEntities it is only the names that
        determine equality. This should help with comparisons later on.
        """
        return self.name == other.name and self.short_name == other.short_name

    def __str__(self):
        """For printing in f.e. fstrings"""
        return f"{self.name} ({self.short_name})"

    def __repr__(self):
        """For straightup prints"""
        return self.__str__()


class ScalarParameter(UncertainEntity):
    """
    Distribution of an uncertain scalar (floating point) parameter.
    This class is wrapping a scipy.stats distribution.

    Attributes:
        name (str): Long (full) name of the parameter.
        short_name (str): Abbreviation/super short name of the parameter.
        unit (str): Very short unit for plots etc.
        distribution (str): The distribution for your parameter.
        parameter1 (float, optional): Optional parameter that defines the distribution.
        parameter2 (float, optional): Optional parameter that defines the distribution.
        parameter3 (float, optional): Optional parameter that defines the distribution.
        parameter4 (float, optional): Optional parameter that defines the distribution.
        lower_limit (float, optional): Lower limit for manual truncation of the distribution.
        upper_limit (float, optional): Upper limit for manual truncation of the distribution.
        distribution_function (scipy.stats distribution): The distribution function from scipy.stats.
        function_inputs (dict): Dictionary of the inputs for the distribution function.
        mean (float): Mean of the distribution
    """

    # all supported distributions (in the find_sts function) and their
    # required inputs (must have a value) are listed here for checks
    conditions_dict = {
        "norm": [True, True, False, False],
        "truncnorm": [True, True, True, True],
        "truncnorm1": [True, True, True, True],  # more human readable truncnorm
        "lognorm": [True, True, True, False],
        "randint": [True, True, False, False],
        "uniform": [True, True, False, False],
        "triang": [True, True, True, False],
        "right skewed": [True, True, True, False],
    }

    def __init__(
        self,
        name: str,
        short_name: str,
        unit: str,
        distribution: str,
        para1: float = None,
        para2: float = None,
        para3: float = None,
        para4: float = None,
        low_lim: float = None,
        up_lim: float = None,
    ) -> None:
        """
        Sets attributes for the methods; parameters are
        None by default, but most distributions require some values.
        The order in which para1, para2, para3 and para4 should be
        given is the same of parameters used in scipy.stats - see link.
        https://docs.scipy.org/doc/scipy-1.8.0/html-scipyorg/reference/stats.html

        Args:
            name (str): Long (full) name of the parameter.
            short_name (str): Abbreviation/super short name of the parameter.
            unit (str): Very short unit for plots etc.
            distribution (str): The distribution for your parameter. You can
            easily implement new distribution from scipy by
            creating a new elif in sampling.
            Every distribution function is required to have
            a pdf, cdf, ppf and rvs function to ensure full
            compatibility with the rest of the module.
            para1 (float, optional): Optional parameter that defines the distribution.
            para2 (float, optional): Optional parameter that defines the distribution.
            para3 (float, optional): Optional parameter that defines the distribution.
            para4 (float, optional): Optional parameter that defines the distribution.
            low_lim (float, optional): Lower limit for manual truncation of the distribution.
            up_lim (float, optional): Upper limit for manual truncation of the distribution.

        Returns:
            None

        Raises:
            KeyError: If the distribution is not implemented in the module.
            ValueError: If the distribution requires more parameters than provided.
        """
        self.name = name
        self.short_name = short_name
        self.unit = unit
        self.distribution = distribution.lower()
        self.parameter1 = para1
        self.parameter2 = para2
        self.parameter3 = para3
        self.parameter4 = para4
        self.lower_limit = low_lim
        self.upper_limit = up_lim

        # raise informative errors if inputs are missing
        self._input_check()
        # assign the correct distribution(inputs) from scipy
        self.distribution_function, self.function_inputs = self._find_sts()
        # for convenient use of the class we assign the mean of the distribution
        self.mean = self.distribution_function.mean(**self.function_inputs)

    def sample(self, samplesize: int) -> np.ndarray:
        """
        Most important method that is used in the Sampler object.
        Creates an array of 'samplesize' samples for the parameter.
        Limits for the distribution are automatically considered here.
        This happens by rescaling the distribution using cdf and ppf.
        https://stackoverflow.com/questions/11491032/truncating-scipy-random-distributions

        Args:
            samplesize (int): number of samples (points in array) to be returned

        Returns:
            np.ndarray: 1 dimensional np.ndarray of length samplesize with random samples

        Raises:
            ValueError: If samplesize is not a positive integer.
            ValueError: If the lower limit is higher than the upper limit.
            ValueError: If the distribution requires other dogmatic endpoints for truncation.
        """
        # ensure type compatability:
        samplesize = int(samplesize)
        if not (samplesize > 0):
            raise ValueError(
                f"Choose a pos. integer instead of {samplesize} as the samplesize"
            )

        # if limits are given, we will set remaining limits automatically and
        # sample with np.random and the cdf and ppf.
        if self.lower_limit is not None or self.upper_limit is not None:
            # set either that is None to infinite:
            if self.lower_limit is None:
                self.lower_limit = -(10**10)
                if (
                    self.distribution_function.pdf(
                        self.lower_limit, **self.function_inputs
                    )
                    > 10**-6
                ):
                    raise ValueError(
                        f"automatic lower limit not sufficiently low - please reset "
                        f"this in the UA code."
                    )
            if self.upper_limit is None:
                self.upper_limit = 10**10
                if (
                    self.distribution_function.pdf(
                        self.upper_limit, **self.function_inputs
                    )
                    > 10**-6
                ):
                    ValueError(
                        f"automatic upper limit not sufficiently high - please reset this "
                        f"in the UA code."
                    )
            if not self.lower_limit < self.upper_limit:
                raise ValueError(f"Lower limit must be below upper limit.")

            # create truncation normalisation
            nrm = self.distribution_function.cdf(
                self.upper_limit, **self.function_inputs
            ) - self.distribution_function.cdf(self.lower_limit, **self.function_inputs)

            # preparing sampling with the cdf (for truncation)
            yr = np.random.rand(samplesize) * (nrm) + self.distribution_function.cdf(
                self.lower_limit, **self.function_inputs
            )
            xr = self.distribution_function.ppf(yr, **self.function_inputs)
        # if no limits are given, simply return from rvs
        else:
            xr = self.distribution_function.rvs(**self.function_inputs, size=samplesize)

        return xr

    def ppf(self, x: list or float) -> np.ndarray:
        """
        ppf (percent point function) values for a given probability.
        Used for the truncation.

        Args:
            x (list or float): probability or list of probabilities

        Returns:
            np.ndarray: percent point function values for the given probabilities

        Raises:
            ValueError: If the input/any of the inputs is not in between 0 and 1.
        """
        if hasattr(x, "__iter__"):
            if any(x) < 0 or any(x) > 1:
                raise ValueError(f"ppf inputs must be in between 0 and 1.")
        else:
            if x < 0 or x > 1:
                raise ValueError(f"ppf inputs must be in between 0 and 1.")
        return self.distribution_function.ppf(x, **self.function_inputs)

    def get_pdf(
        self, x_range: list[float] = None, n_x: int = 10**5, prec: int = 4
    ) -> list[np.ndarray]:
        """
        Creates an x and y array for the pdf of the distribution.
        If no interval is specified, the first continuous interval of
        non-zero (rounded to 'prec') values of the pdf is chosen.
        [Not fully functional yet - works for most cases, but not all]

        Args:
            x_range (list[float], optional): List of starting and end point for the pdf. Defaults to None.
            n_x (int, optional): Number of points in x and y array. Defaults to 10**5.
            prec (int, optional): Precision to which the pdf needs to be zero for autolimits. Defaults to 4.
            list[np.ndarray]: np.ndarrays representing x and y values of PDF.

        Returns:
            list[np.ndarray]: x and y array: np.ndarrays representing x and y values of PDF
        """
        # if no x_range is given, we construct one that starts at pi ~0
        # and ends alike at a higher value where pi ~0, with values
        # where pi < 0 have to lie in between.
        if x_range is None:
            x_range = [-0.1, 0.1]
            multiplier = 10
            pdf = [1]
            # extend until we have zero values at the ends and non zero values in between.
            # TODO test the automatic x_range
            while (
                (np.round(pdf[0], prec)) > 0
                or (np.round(pdf[-1], prec)) > 0
                or len(np.where(np.round(pdf, prec) > 0)[0]) == 0
            ):
                x_range = [multiplier * lim for lim in x_range]
                x = np.linspace(x_range[0], x_range[1], 10**6)
                pdf = self.distribution_function.pdf(x, **self.function_inputs)
            # cut out zero values at the ends: first at the low, then at the hight values
            pi_x = np.round(pdf, prec)  # just for readability and speed
            if 0 in pi_x:
                if np.where(pi_x == 0)[0][0] < np.where(pi_x > 0)[0][0]:
                    lower_non_zero = np.where(pi_x > 0)[0][0]
                    pdf = pdf[lower_non_zero:]
                    x = x[lower_non_zero:]
            if 0 in np.round(pdf, prec):
                upper_zero = np.where(np.round(pdf, prec) == 0)[0][0]
                pdf = pdf[:upper_zero]
                x = x[:upper_zero]
            # rescale length of arrays to n_x
            # TODO: check why this doesn't reliably work!
            x = np.linspace(x[0], x[-1], n_x)
            pdf = self.distribution_function.pdf(x, **self.function_inputs)
        else:
            x = np.linspace(x_range[0], x_range[-1], n_x)
            pdf = self.distribution_function.pdf(x, **self.function_inputs)
        return x, pdf

    def plot_samples(
        self,
        samplesize: int = 10 * 4,
        no_bins: int = "auto",
        show: bool = True,
        figsize: tuple = None,
        save_path: Path = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        Histogram of some (default 10000) samples.

        Args:
            samplesize (int, optional): Number of samples to be drawn. Defaults to 10*4.
            no_bins (int, optional): Number of bins for the histogram. Defaults to "auto".
            show (bool, optional): Whether or not to display the plot. Defaults to True.
            figsize (tuple, optional): Size of the figure. Defaults to None, then set to 6, 4.5.
            save_path (Path, optional): Path to save the plot. Defaults to None (no saving).

        Returns:
            tuple[plt.Figure, plt.Axes]: Figure and Axes objects of the plot.
        """
        if not figsize:
            figsize = (6, 4.5)
        random_values = self.sample(samplesize)
        fig, ax = plt.subplots(figsize=figsize)
        ax.hist(
            random_values,
            bins=no_bins,
            histtype="bar",
            color="darkorange",
            edgecolor="k",
            alpha=0.8,
        )

        if show:
            plt.show()
        else:
            plt.close()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        return fig, ax

    def plot_distribution(
        self,
        x_range: list[float] = None,
        show: bool = True,
        figsize: tuple = None,
        save_path: Path = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        Simple PDF plot, (optionally from start - stop in x_range).

        Args:
            x_range (list[float], optional): Range of x values to plot. Defaults to None.
            show (bool, optional): Whether or not to display the plot. Defaults to True.
            figsize (tuple, optional): Size of the figure. Defaults to None. (then set to 6, 4.5)
            save_path (Path, optional): Path to save the plot. Defaults to None (no saving).

        Returns:
            tuple[plt.Figure, plt.Axes]: Figure and Axes objects of the plot.
        """
        if not figsize:
            figsize = (6, 4.5)
        x, pdf = self.get_pdf(x_range=x_range)
        fig, ax = plt.subplots(figsize=figsize)
        plt.plot(x, pdf, label="PDF")
        plt.xlabel(f"Values of parameter {self.name} in {self.unit}")
        plt.ylabel(f"probability")
        plt.title(f"Parameter  {self.name}: PDF")
        if show:
            plt.show()
        else:
            plt.close()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        return fig, ax

    def _find_sts(self) -> tuple:  # [sts._continuous_distns.gen, dict]:
        """
        Assigns a base function from scipys stats library for the chosen
        distribution which is used for sampling and the pdf. Also sets
        the base input for these functions with the provided values from
        parameter 1-4. To implement more distributions, just add them
        as a new elif option, and in the conditions_dict of this class.
        Note that we are later using the cdf, ppf, pdf and rvs functions
        of the distribution, so the assigned distribution has to have these.

        Args:
            None (run after innit)

        Returns:
            tuple: Tuple of the scipy.stats distribution function and its inputs.
        """
        if self.distribution == "norm":
            mean = self.parameter1
            dev = self.parameter2
            inputs = {"loc": mean, "scale": dev}
            return sts.norm, inputs

        elif self.distribution == "truncnorm":
            # official scipy truncnorm implementation
            a = self.parameter1
            b = self.parameter2
            mean = self.parameter3
            dev = self.parameter4
            inputs = {"a": a, "b": b, "loc": mean, "scale": dev}
            return sts.truncnorm, inputs

        elif self.distribution == "truncnorm1":
            # more human readable truncnorm implementation
            mean = self.parameter1
            my_low = self.parameter2  # actual lower limit
            my_up = self.parameter3  # actual upper limit
            dev = self.parameter4  # standard dev
            # low and up have to be rescaled for the standard norm:
            if dev > 0:
                low, up = (my_low - mean) / dev, (my_up - mean) / dev
            else:  # need to provide low and up in case std is 0...
                low, up = mean - 1, mean + 1
            inputs = {"a": low, "b": up, "loc": mean, "scale": dev}
            return sts.truncnorm, inputs

        elif self.distribution == "lognorm":
            shape = self.parameter1
            loc = self.parameter2
            scale = self.parameter3
            # shape for how much log it is, loc for the start, scale for the width
            inputs = {"s": shape, "loc": loc, "scale": scale}
            return sts.lognorm, inputs

        # TODO: integrate in tests.
        elif self.distribution == "randint":
            low = self.parameter1
            high = self.parameter2
            # uniform distribution for all integers between low and high
            inputs = {
                "low": low,
                "high": high,
            }
            return sts.randint, inputs

        elif self.distribution == "uniform":
            start = self.parameter1
            length = self.parameter2
            # uniform distribution from loc to loc+scale
            inputs = {"loc": start, "scale": length}
            return sts.uniform, inputs

        elif self.distribution == "triang":
            midmultiplier = self.parameter1
            start = self.parameter2
            length = self.parameter3
            # slopes up from loc to loc+shape*scale and down until loc+scale

            inputs = {"c": midmultiplier, "loc": start, "scale": length}
            return sts.triang, inputs

        elif self.distribution == "right skewed":
            """
            This distribution is a lognormal distribution mirrored at
            the y-axis. The parameters provided for this
            """
            # TODO: test this!
            shape = self.parameter1
            loc = self.parameter2
            scale = self.parameter3

            right_skewed_lognorm = RightSkewedLognorm()
            inputs = {"s": shape, "loc": loc, "scale": scale}
            return right_skewed_lognorm, inputs
        else:
            raise KeyError(
                f"The specified distribution ({self.distribution}) seems to "
                f"not match any of the implemented distributions."
            )

    def _input_check(self) -> None:
        """
        warns (prior to sampling) if values are missing for a distribution

        Raises:
            ValueError: If the distribution is not implemented in the module.
            ValueError: If the distribution requires more parameters than provided.
        """
        if not self.distribution in self.conditions_dict.keys():
            raise ValueError(
                f"The specified distribution {self.distribution} is not "
                f"matching any of the confirmed implemented distributions."
            )
        else:
            rules = self.conditions_dict[self.distribution]
            for i, (rule, value) in enumerate(
                zip(
                    rules,
                    [
                        self.parameter1,
                        self.parameter2,
                        self.parameter3,
                        self.parameter4,
                    ],
                )
            ):
                if rule and value is None:
                    raise ValueError(
                        f"For a {self.distribution} distribution a {i}st/nd/rd/th parameter is required."
                    )

    def __repr__(self):
        """
        For straight up prints of the object.
        """
        return (
            f"ScalarParameter named {self.name} (short {self.short_name}, in {self.unit}) with "
            f"a {self.distribution} distribution with input values {self.function_inputs}."
        )


class ConstantParameter(UncertainEntity):
    """
    Parameter class to integrate constant parameters in the MC workflow
    without changing the model itself. This parameter can take any value
    (bool, string, char, lists etc.) and can therefore be used for
    keyword arguments that cannot be covered else.
    This class can also serve as a blueprint for other custom parameters
    - for example a boolean which is supposed to be True in half of all
    cases and False for the other half.

    Attributes:
        name (str): Long (full) name of the parameter.
        short_name (str): Abbreviation/super short name of the parameter.
        explanation (str): Explanation of the parameter.
        value: Value of the parameter.
        unit (str): Very short unit for plots etc.
    """

    def __init__(
        self, name: str, short_name: str, explanation: str, value, unit: str = None
    ) -> None:
        self.name = name
        self.short_name = short_name
        self.explanation = explanation
        self.value = value
        self.unit = unit

    def ppf(self, x):
        """Equivalent to ppf function - needed for LHS sampling."""
        return [self.value for _ in x]

    def sample(self, samplesize: int) -> np.ndarray:
        """Returns array with copies of original value for the Sampler"""
        return np.array([self.value for _ in range(samplesize)])

    def __repr__(self):
        return (
            f"ConstantParameter named {self.name} (short {self.short_name})"
            f"; explanation: {self.explanation}; value {self.value}."
        )
