import matplotlib.pyplot as plt
import scipy.stats as sts
import warnings

from pathlib import Path

from mfa_ua.monte_carlo.parameters import ScalarParameter, ConstantParameter


class Sampler:
    """
    Collection of probabilistically determined parameters  that
    are samples to create a set of parameter values for a
    MonteCarlo analysis. Also helps to visualize the samples.

    Attributes:
        parameters: list of parameters (ScalarParameter, ConstantParameter and similar)
        n_parameters: number of parameters
        parameters_order: dictionary with the order of the parameters
        parameters_type: dictionary with the type of the parameters
        parameter_samples: list of samples for each parameter
        samplesize: number of sets to be created
        mode: MC (default) or LHS (latin hypercube sampling)
        parameter_sets: list of lists with the parameter values (samplesize x n_parameters)

    """

    parameter_samples = None  # for the __check samples method
    samplesize = 0
    mode = "MC"  # default mode is MC (random)

    def __init__(self, parameters: list[ScalarParameter]) -> None:
        """
        Initializes the sampler with a list of parameters.

        Args:
            parameters: list of parameters (ScalarParameter, ConstantParameter)
        """
        self.parameters = parameters
        self.n_parameters = len(self.parameters)
        self.parameters_order = {}
        self.parameters_type = {}
        for index, parameter in enumerate(parameters):
            self.parameters_order[parameter.name] = index
            self.parameters_order[parameter.short_name] = index

            if type(parameter) == ScalarParameter:
                self.parameters_type[parameter.name] = "scalar"
                self.parameters_type[parameter.short_name] = "scalar"
            elif type(parameter) == ConstantParameter:
                self.parameters_type[parameter.name] = "constant"
                self.parameters_type[parameter.short_name] = "constant"
            else:
                warnings.warn(
                    f"Your parameter {parameter.name} is of type {type(parameter)}, "
                    f"which is not explicitly supported. If it has a compatible sampling "
                    f"method, it might still work for most functions."
                )
                self.parameters_type[parameter.name] = "other"
                self.parameters_type[parameter.short_name] = "other"

    def sample(self, mode: str = "MC", samplesize: int = 1000) -> list:
        """
        Creates and returns sets of parameters (list of lists) - each set
        contains one realization (value) of all parameters.
        On the way, we create samples for each parameter indivually and
        stroe it for later use.

        Args:
            mode (str): MC (default) or LHS (latin hypercube sampling)
            samplesize (int): number of sets to be created

        Returns:
            parameter_sets: list of lists with the parameter values (samplesize x n_parameters)

        Raises:
            ValueError: if mode is not MC or LHS
            TypeError: if a parameter does not have a ppf function (LHS sampling)
        """
        self.samplesize = samplesize
        self.mode = mode

        if mode == "MC":
            # get list of samples for each parameter
            self.parameter_samples = [
                para.sample(samplesize) for para in self.parameters
            ]
            # repackage the samples into sets of parameters
            self.parameter_sets = self._repackage(self.parameter_samples)

        elif mode == "LHS":
            """
            LHS sampling with scipys LHS sampler. Only works with parameters
            that support a PPF (percent point function - inverse of cdf)
            """
            LHS_sampler = sts.qmc.LatinHypercube(d=self.n_parameters)
            lhs_values = LHS_sampler.random(samplesize)
            self.parameter_samples = []
            for i, para in enumerate(self.parameters):
                if not hasattr(para, "ppf"):
                    raise TypeError(
                        f"Parameters in sampler need a ppf function ({para.name} hn)."
                    )
                self.parameter_samples.append(para.ppf(lhs_values[:, i]))
            self.parameter_sets = self._repackage(self.parameter_samples)
        else:
            raise ValueError(f"No support for sampling in mode '{mode}'.")
        # TODO: check options for correlated MC sampling
        return self.parameter_sets

    def plot_2D(
        self,
        parameter1: str,
        parameter2: str,
        n_samples: int = None,
        show: bool = True,
        figsize: tuple = None,
        save_path: Path = None,
    ) -> tuple[plt.figure, plt.axis]:
        """
        Makes a scatterplot of the sample values of two parameters
        (specified by their full  or short name). Autoscales the size of the dots.

        Arguments:
            parameter1 (str): either full or short name of the parameter on x axis
            parameter2 (str): either full or short name of the parameter on y axis
            show: whether or not to display the plot (closes the canvas)
            figsize: size of the figure - defaults to (6,4.5)

        Returns:
        """
        if not n_samples:
            n_samples = self.samplesize
        if not figsize:
            figsize = (6, 4.5)
        self._check_samples_for_plot(s_type="scalar", paras=[parameter1, parameter2])
        index_x = self.parameters_order[parameter1]
        index_y = self.parameters_order[parameter2]
        x = self.parameter_samples[index_x][:n_samples]
        y = self.parameter_samples[index_y][:n_samples]
        fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(x, y, s=10**4 / n_samples, c="crimson", alpha=0.6)
        ax.set_xlabel(f"Parameter {parameter1} in {self.parameters[index_x].unit}")
        ax.set_ylabel(f"Parameter {parameter2} in {self.parameters[index_y].unit}")
        ax.set_title(
            f"Scatterplot ({self.samplesize} points) of {parameter1} and {parameter2}."
        )

        if show:
            plt.show()
        else:
            plt.close()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig, ax

    def plot_3D_old(
        self,
        parameter1: str,
        parameter2: str,
        parameter3: str,
        c_map: str = "hot",
        figsize: tuple = None,
        show: bool = True,
        save_path: Path = None,
    ) -> tuple[plt.figure, plt.axis]:
        """
        For 3D scatter plots of sample sets.

        Args:
            parameter1 (str): long or short name of parameter on x axis
            parameter2 (str): long or short name of parameter on y axis
            parameter3 (str): long or short name of parameter on z axis
            c_map (str): colormap for the z axis - defaults to "hot"
            figsize (tuple): size of the figure - defaults to (6,4.5)
            show(bool): whether or not to display the plot
            save_path (Path): path to save the plot - if None, it will not be saved

        Returns:
            fig, ax: figure and axis of the plot
        """
        if not figsize:
            figsize = (8, 10)
        # Use python guide if you want to change it :
        # https://pythonguides.com/matplotlib-3d-scatter/
        self._check_samples_for_plot(
            s_type="scalar", paras=[parameter1, parameter2, parameter3]
        )
        index_x = self.parameters_order[parameter1]
        index_y = self.parameters_order[parameter2]
        index_z = self.parameters_order[parameter3]
        x = self.parameter_samples[index_x]
        y = self.parameter_samples[index_y]
        z = self.parameter_samples[index_z]

        fig, ax = plt.subplots(figsize=figsize)
        ax = plt.axes(projection="3d")
        color_map = plt.get_cmap(c_map)
        size_dots = (
            4 * 10**4 / self.samplesize
        )  # adjust as needed if it doesn't look nice.
        scatter_plot = ax.scatter3D(
            x, y, z, c=z, cmap=color_map, s=size_dots, alpha=0.5
        )
        plt.colorbar(scatter_plot, shrink=0.5)
        ax.set_xlabel(
            f"Parameter {parameter1} in " f"{self.parameters[index_x].unit}",
            fontweight="bold",
        )
        ax.set_ylabel(
            f"Parameter {parameter2} in " f"{self.parameters[index_y].unit}",
            fontweight="bold",
        )
        ax.set_zlabel(
            f"Parameter {parameter3} in " f"{self.parameters[index_z].unit}",
            fontweight="bold",
        )
        plt.title(
            f"3D SCATTER for {parameter1}, {parameter2} and " f"{parameter3}",
            fontweight="bold",
            size=10,
        )
        if show:
            plt.show()
        else:
            plt.close()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        return fig, ax

    def plot_3D(
        self,
        parameter1,
        parameter2,
        parameter3,
        show=True,
        figsize=None,
        c_map="viridis",
        save_path=None,
    ):
        """
        Plots a 3D scatter plot of three parameters.

        Args:
            parameter1: The first parameter to plot.
            parameter2: The second parameter to plot.
            parameter3: The third parameter to plot.
            show (bool, optional): Whether or not to display the plot. Defaults to True.
            figsize (tuple, optional): Size of the figure. Defaults to None, then set to (6, 4.5).
            c_map (str, optional): Colormap to use for the scatter plot. Defaults to 'viridis'.
            save_path (Path, optional): Path to save the plot. Defaults to None.

        Returns:
            tuple[plt.Figure, plt.Axes]: Figure and Axes objects of the plot.
        """
        if not figsize:
            figsize = (6, 4.5)

        index_x = self.parameters_order[parameter1]
        index_y = self.parameters_order[parameter2]
        index_z = self.parameters_order[parameter3]
        x = self.parameter_samples[index_x]
        y = self.parameter_samples[index_y]
        z = self.parameter_samples[index_z]

        fig = plt.figure(figsize=figsize, constrained_layout=True)
        ax = fig.add_subplot(111, projection="3d")
        color_map = plt.get_cmap(c_map)
        size_dots = (
            4 * 10**3 / self.samplesize
        )  # adjust as needed if it doesn't look nice.

        scatter_plot = ax.scatter(x, y, z, c=z, cmap=color_map, s=size_dots, alpha=0.5)

        # Create a new axes for the colorbar
        cax = fig.add_axes([0.15, 0.25, 0.03, 0.4])  # [left, bottom, width, height]
        fig.colorbar(scatter_plot, cax=cax, orientation="vertical")

        ax.set_xlabel(
            f"Parameter {parameter1} in {self.parameters[index_x].unit}",
            fontweight="bold",
        )
        ax.set_ylabel(
            f"Parameter {parameter2} in {self.parameters[index_y].unit}",
            fontweight="bold",
        )
        ax.set_zlabel(
            f"Parameter {parameter3} in {self.parameters[index_z].unit}",
            fontweight="bold",
        )
        ax.set_title(
            f"3D SCATTER for {parameter1}, {parameter2} and {parameter3}",
            fontweight="bold",
            size=10,
        )

        if show:
            plt.show()
        else:
            plt.close(fig)  # Close the figure if not showing

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        return fig, ax

    def _repackage(self, parameters_individual_samples: list):
        """
        For an input of p parameters with n samples each:
        returns n sets with p parameters.
        """
        sets = []
        for set_index, _ in enumerate(parameters_individual_samples[-1]):
            set = [parameter[set_index] for parameter in parameters_individual_samples]
            sets.append(set)
        return sets

    def _check_samples_for_plot(self, s_type: str = None, paras: list[str] = None):
        """Ensures parameter samples are available before we plot."""
        if s_type is not None:
            for parameter in paras:
                if self.parameters_type[parameter] is not s_type:
                    warnings.warn(
                        f"The parameter {parameter} you try to plot is probably not "
                        f"compatible with the plotting method."
                    )
        if not self.parameter_samples:
            warnings.warn(
                "You wanted to plot before creating samples,so we made the default sample."
            )
            _ = self.sample()
        return
