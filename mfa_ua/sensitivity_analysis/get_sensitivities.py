# %%
import sympy as sy
from typing import Callable


def partial_derivate(
    function: Callable, all_parameters: list[sy.Symbol], single_parameter: sy.Symbol
) -> Callable:
    """
    Get the derivative of a function with respect to a single parameter.

    Args:
        function (Callable): The function to be evaluated.
        all_parameters (list[sy.Symbol]): A list of all parameters as sympy symbols.
        single_parameter (sy.Symbol): The parameter with respect to which the sensitivity is calculated.

    Returns:
        Callable: The derivative of the function with respect to the single parameter as a function.
    """
    derivative = sy.diff(function(*all_parameters), single_parameter)
    return derivative


def get_sensitivity_for_parameter(
    function: Callable,
    all_parameters: list[sy.Symbol],
    parameter_values: dict[sy.Symbol, float],
    single_parameter: sy.Symbol,
) -> tuple[float]:
    """
    Get the sensitivities of a function with respect to a single parameter.

    Args:
        function (Callable): The function to be evaluated.
        all_parameters (list[sy.Symbol]): A list of all parameters as sympy symbols.
        parameter_values (dict[sy.Symbol, float]): A dictionary containing the values of all parameters.
        single_parameter (sy.Symbol): The parameter with respect to which the sensitivity is calculated.

    Returns:
        dict[str,float]: A dict containing the absolute and relative sensitivity.
    """
    normal_operating_point = function(*all_parameters).subs(parameter_values)
    partial_derivative = partial_derivate(function, all_parameters, single_parameter)
    absolute_sensitivity = partial_derivative.subs(parameter_values)
    relative_sensitivity = (
        absolute_sensitivity
        * parameter_values[single_parameter]
        / normal_operating_point
    )
    sensitivities = {}
    sensitivities["absolute_sensitivity"] = absolute_sensitivity
    sensitivities["relative_sensitivity"] = relative_sensitivity
    return sensitivities


def get_sensitivities(
    function: Callable,
    parameters: list[sy.Symbol],
    parameter_values: dict[sy.Symbol, float],
) -> tuple[float]:
    """
    Get the sensitivities of a function with respect to all its parameters.

    Args:
        function (Callable): The function to be evaluated.
        parameters (list[sy.Symbol]): A list of all the function's parameters as sympy symbols.
        parameter_values (dict[sy.Symbol, float]): A dictionary containing the values of all parameters.
    Returns:
        dict[str,float]: A dict containing the absolute and relative sensitivities for each parameters.

    """
    sensitivities = {}
    for parameter in parameters:
        sensitivities[parameter.__str__()] = get_sensitivity_for_parameter(
            function, parameters, parameter_values, parameter
        )
    return sensitivities


if __name__ == "__main__":
    x = sy.symbols("x")
    y = sy.symbols("y")
    z = sy.symbols("z")
    w = sy.symbols("w")
    all_parameters = [x, y, z, w]
    parameter_values = {x: 1, y: 2, z: 3, w: 4}

    def test_function(x, y, z, w):
        return (x * y + x**z) * w

    sensitivities = get_sensitivities(test_function, all_parameters, parameter_values)
    print(sensitivities)

# %%
