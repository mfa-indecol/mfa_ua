from setuptools import setup, find_packages

setup(
    name="mfa_ua",
    version="0.1.0",
    author="Nils Dittrich",
    author_email="nils.dittrich@ntnu.no",
    description="Uncertainty analysis tools for MFA",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/mfa_ua",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        # not all versions and combination of versions are tested
        "numpy>=1.18.0,<2.0.0",
        "scipy>=1.4.0,<2.0.0",
        "matplotlib>=3.1.0,<4.0.0",
        "tqdm>=4.41.0,<5.0.0",
        "sympy>=1.5.0,<2.0.0",
    ],
)

# install with "python -m pip install -e ."
