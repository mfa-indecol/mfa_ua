# mfa_ua
Various tools related to uncertainty analysis in material flow analysis (MFA).
Some of the modules were specifically built for the NTNU course TEP4285, others come from research projects.

## current subpackages
- monte_carlo: designed for TEP4285, contains module for setting up parameters, sampling, and the monte carlo analysis
- parameter_estimation: fitting a lognormal distribution on lower and upper bounds and a mode, inversing lognormal into a right skewed distribution
- sensitivity_analysis: computing sensitivities for any fucntion using sympy; plotting sensitivity heatmaps
