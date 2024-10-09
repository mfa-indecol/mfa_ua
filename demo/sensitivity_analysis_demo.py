# %%
import sympy as sy

from mfa_ua.sensitivity_analysis.get_sensitivities import get_sensitivities

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
    for key, item in sensitivities.items():
        print(f"{key}: {item}")

# %%
