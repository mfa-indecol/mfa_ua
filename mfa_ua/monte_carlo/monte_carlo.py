import numpy as np
import matplotlib.pyplot as plt
import warnings
from tqdm import tqdm
from typing import Callable
from pathlib import Path

from mfa_ua.monte_carlo.parameters import ScalarParameter, ConstantParameter
from mfa_ua.monte_carlo.sampler import Sampler


class MonteCarlo:
    """
    For conducting an MC analysis, storing the results and visualizing them.

    Attributes:
        function: the function applied at each iteration of the simulation
        sample: the Sampler object that provides the sets of inputs
        parameters: a list of parameter objects from the Sampler
        iterations: number of function executions and results
        result_lists: a list for each result, with iterations many entries
        results_sets: iteration many sets of all function outputs
        result_names: names of the results from the function
        results_order: dict for easier retrieval of the results
        results_type: scalar or timeseries and so on - for picking plots
        figures: visualisations results
        hist_nbins: number of bins for the histograms (defaults to 'auto')
    """

    def __init__(
        self,
        function: Callable[list, list[list]],
        sample: Sampler,
    ) -> None:
        """
        Initializes the MonteCarlo object without perfoming any calculations.

        Args:
            function: must take a list of parameters and return two lists,
                           one with values, one with names.
            sample: a Sampler object with parameters that fit the function (number and order)
        """
        self.function = function
        self.sample = sample
        self.parameters = sample.parameters

    def analyze(
        self,
        iterations: int = 1000,
        hist_nbins: int = "auto",
        visualisations: bool = False,
        show_progress_bar: bool = True,
    ) -> None:
        """
        The main method of the MC class - conducts a MonteCarlo
        analysis by applying the function to iterations many sets of samples
        of the parameters and stores the results in the object.

        Args:
            iterations (int, optional): number of iterations to be performed (default 1000)
            hist_nbins (int, optional): number of bins for the histograms (default "auto")
            visualisations (bool, optional): whether or not the results will be plotted. (default False)
            show_progress_bar (bool, optional): whether or not to show the progress bar. (default True)

        Returns:
            None

        Raises:
            Exception: if the function does not return the same names each time
            Warning: if the function returns a list that is not supported
        """
        # prepare everything
        self.iterations = iterations
        self._check_iterations()
        self.hist_nbins = hist_nbins

        # calculating the outputs (this takes time!)
        if show_progress_bar:
            self.result_sets, self.lists_result_names = zip(
                *tqdm(
                    [
                        self.function(inp)
                        for inp in self.sample.parameter_sets[:iterations]
                    ],
                    desc="Computing results for each iteration",
                    total=iterations,
                    leave=True,
                )
            )
            self.result_names = self.lists_result_names[0]
        else:
            self.result_sets, self.lists_result_names = zip(
                *[self.function(inp) for inp in self.sample.parameter_sets[:iterations]]
            )
            self.result_names = self.lists_result_names[0]

        # check if the same names are returned each time
        for index, current in enumerate(self.lists_result_names, start=1):
            prior = self.lists_result_names[index - 1]
            if prior != current:
                raise Exception(
                    f"Your function does not always return the same names: in result "
                    f"{index} the names were {current}, before they were {prior}."
                )

        # repackage results
        self.result_lists = []
        for output in range(len(self.result_names)):
            self.result_lists.append([set[output] for set in self.result_sets])

        # establish order of the results:
        self.results_order = {}
        self.results_type = {}
        for index, (result_name, result_example) in enumerate(
            zip(self.result_names, self.result_sets[0])
        ):
            self.results_order[result_name] = index
            if type(result_example) == list:
                if (
                    len(result_example) == 2
                    and type(result_example[0]) == list
                    and type(result_example[1] == list)
                    and type(result_example[0][0]) in [int, float, np.float64]
                ):  # if it looks like time & results are specified
                    self.results_type[result_name] = "timeseries"
                else:
                    self.results_type[result_name] = "other list"
                    warnings.warn(
                        f"Your result {result_name} is a not identifiable list."
                    )
            elif type(result_example) in [int, float, np.float64]:
                self.results_type[result_name] = "scalar"
            else:
                self.results_type[result_name] = f"{type(result_example)}"
                warnings.warn(
                    f"the type of your result {result_name} is {type(result_example)},"
                    f"which means we cannnot plot it."
                )
        # make and show plots as wished:
        if visualisations:
            self.figures = self._plot_results()
        return

    def histogram(
        self,
        result_name: str,
        color: str = "royalblue",
        figsize: tuple = None,
        save_path: Path = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        A simple histogram for the scalar results of the MC.

        Args:
            result_name (str): name of the result to plot
            color (str, optional): colour of the histogram. Defaults to "royalblue".
            figsize (tuple, optional): size of the figure. Defaults to None => (6, 4.5).
            save_path (Path, optional): path to save the figure. Defaults to None.

        Returns:
            tuple[plt.Figure, plt.Axes]: figure and axes of the histogram
        """
        if not figsize:
            figsize = (6, 4.5)
        # Prepare the data
        index = self.results_order[result_name]
        results = self.result_lists[index]

        # Compute statistics
        mu = np.mean(results)
        std = np.std(results)

        # Create figure and axes
        fig, ax = plt.subplots(figsize=figsize)

        # Plot histogram
        n, _, _ = ax.hist(results, bins=self.hist_nbins, color=color, alpha=0.8)
        y_cs = [0, max(n)]  # For the stat intervals

        # Plot mean and standard deviations
        ax.plot([mu, mu], y_cs, c="k", lw=2, label="mean", alpha=0.8)
        ax.plot([mu + std, mu + std], y_cs, "--", c="k", lw=2, alpha=0.6)
        l1 = "1 std - 68%"
        l2 = "2 stds - 95%"
        ax.plot([mu - std, mu - std], y_cs, "--", c="k", lw=2, alpha=0.6, label=l1)
        ax.plot([mu + 2 * std, mu + 2 * std], y_cs, ":", c="k", lw=2, alpha=0.4)
        ax.plot(
            [mu - 2 * std, mu - 2 * std], y_cs, ":", c="k", lw=2, alpha=0.4, label=l2
        )

        # Add legend, labels, and title
        ax.legend(loc="best")
        ax.set_xlabel(f"Values of {result_name}")
        ax.set_ylabel("Number of runs")
        ax.set_title(
            f"{result_name} results from MC simulation with {self.iterations} iterations"
        )

        # Show plot
        plt.show()

        # Save figure if save_path is provided
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        # Close the plot to free up memory
        plt.close(fig)

        return fig, ax

    def plot_timeseries(
        self,
        result_name: str,
        colour: str = "darkorange",
        scatter: bool = False,
        y_unit="unknown",
        time_unit="unknown",
        figsize: tuple = None,
        save_path: Path = None,
    ) -> tuple[plt.figure, plt.Axes]:
        """
        Plots timeseries with CI intervals. untested version.

        Args:
            result_name (str): name of the result to plot
            colour (str, optional): colour of the plot. Defaults to "darkorange".
            scatter (bool, optional): whether to plot scatter points. Defaults to False.
            y_unit (str, optional): unit of the y-axis. Defaults to "unknown".
            time_unit (str, optional): unit of the x-axis. Defaults to "unknown".
            figsize (tuple, optional): size of the figure. Defaults to None => (20, 10).
            save_path (Path, optional): path to save the figure. Defaults to None.

        Returns:
            tuple[plt.figure, plt.Axes]: figure and axes of the plot
        """
        index = self.results_order[result_name]
        results = self.result_lists[index]
        result_by_year = [[] for _ in results[0][0]]
        for result in results:
            for index, year_value in enumerate(result[1]):
                result_by_year[index].append(year_value)
        means = [np.mean(year) for year in result_by_year]
        stds = [np.std(year) for year in result_by_year]
        t = results[0][0]

        fig = plt.figure(figsize=(20, 10))
        plt.fill_between(
            x=t,
            y1=np.subtract(means, np.einsum("i,->i", stds, 2)),
            y2=np.subtract(means, np.einsum("i,->i", stds, 1)),
            alpha=0.25,
            color=colour,
            label="95% interval",
        )
        plt.fill_between(
            x=t,
            y1=np.add(means, np.einsum("i,->i", stds, 2)),
            y2=np.add(means, np.einsum("i,->i", stds, 1)),
            alpha=0.25,
            color=colour,
        )
        plt.fill_between(
            x=t,
            y1=np.subtract(means, np.einsum("i,->i", stds, 1)),
            y2=np.add(means, np.einsum("i,->i", stds, 1)),
            alpha=0.55,
            color=colour,
            label="68% interval",
        )
        plt.plot(t, means, color="black", label=f"mean value for {result_name} ")
        if scatter:
            for xe, ye in zip(t, result_by_year):
                plt.scatter(
                    [xe] * len(ye), ye, color="black", s=10**3 / len(ye), alpha=0.1
                )
            # plt.scatter(x = t, y = result_by_year, alpha = 0.1)
        plt.xlabel(f"Time in {time_unit} ")
        plt.ylabel(f"{result_name} in {y_unit}.")
        plt.legend(loc="best")
        plt.title(f"{result_name} over time")
        plt.show()
        plt.close()
        return fig

    def _plot_results(self, show: bool = False) -> list[plt.figure]:
        """Plots all results from the MC function that can be plotted."""
        figures = []
        # checks which results are timeseries to plot histograms or other plots.
        for name in self.result_names:
            if self.results_type[name] == "scalar":
                fig, ax = self.histogram(result_name=name)
            elif self.results_type[name] == "timeseries":
                fig = self.plot_timeseries(result_name=name)
            else:
                warnings.warn(
                    f"The type of the result {name} ({self.results_type[name]}) cannot "
                    f"be plotted here yet."
                )
                continue
            figures.append((fig, ax))
        if show and False:
            plt.ion()
            for fig, ax in figures:
                if fig:  # if figure is not None
                    fig.show()
        return figures

    def _check_iterations(self) -> None:
        """
        Warns user of unreasonable inputs and ensures sufficient samples.

        Raises:
            Exception: if iterations are not a positive integer > 0.
            Warning: if iterations are < 100.
            Warning: if iterations < samplesize of sampler object.
        """
        if self.iterations < 1:
            raise Exception(
                f"You cannot have {self.iterations} iterations - use pos. integer!"
            )
        elif self.iterations < 100:
            warnings.warn(
                f"You use {self.iterations} iterations, that's probably not enough."
            )

        if self.iterations > self.sample.samplesize:
            mode = self.sample.mode
            _ = self.sample.sample(mode=mode, samplesize=self.iterations)
            warnings.warn(
                f"You picked more MC iterations than there were samples availabale, "
                f"so we resampled with {self.iterations} in {mode} mode."
            )
        return
