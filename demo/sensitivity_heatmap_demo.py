# %%
import sympy as sy
from mfa_ua.sensitivity_analysis.get_sensitivities import get_sensitivities
from mfa_ua.sensitivity_analysis.sensitivity_heatmap import plot_sensitivity_heatmap

if __name__ == "__main__":
    x = sy.symbols("x")
    y = sy.symbols("y")
    z = sy.symbols("z")
    w = sy.symbols("w")
    all_parameters = [x, y, z, w]
    parameter_values = {x: 1, y: 2, z: 3, w: 4}

    def f1(x, y, z, w):
        return (x * y + x**z) * w

    def f2(x, y, z, w):
        return (x + y + x / z) * w**2

    sensitivities = {}
    sensitivities["f1"] = get_sensitivities(f1, all_parameters, parameter_values)
    sensitivities["f2"] = get_sensitivities(f2, all_parameters, parameter_values)

    fig, ax = plot_sensitivity_heatmap(sensitivities)

# %%
