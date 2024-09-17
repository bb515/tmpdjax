"""
Setup script for tmpd.

This setup is required or else
    >> ModuleNotFoundError: No module named 'tmpd'
will occur.
"""
from setuptools import setup, find_packages
import pathlib

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

extra_compile_args = ['-O3']
extra_link_args = []


setup(
    name="tmpd",
    version="0.0.0",
    description="tmpd is a diffusion package for linear inverse problems",
    long_description=README,
    long_description_content_type="text/markdown",
    license="MIT",
    packages=find_packages(exclude=['*.test']),
    install_requires=[
        "matplotlib",
        "scikit-image",
        "lpips",
        "diffusionjax",
         ],
    extras_require={
        "linting": [
        "flake8",
        "pylint",
        "mypy",
        "typing-extensions",
        "pre-commit",
        "ruff",
        "jaxtyping",
        ],
        "testing": [
        "pytest",
        "pytest-xdist",
        "pytest-cov",
        "coveralls",
        "jax>=0.4.1",
        "jaxlib>=0.4.1",
        "setuptools_scm[toml]",
        "setuptools_scm_git_archive",
        ],
        "song": [
        "tensorflow-gan==2.0.0",
        "tensorflow-io==0.32.0",
        "tensorflow_datasets==4.3.0",
        "tensorflow-probability==0.15.0",
        "tensorboard==2.7.0",
        "flax==0.3.3",
        ],
        "gaussian": [
        "probit",
        # "torch",
        "mlkernels",
        "numpyro",
        "pandas",
        "POT",
        ],
    },
    include_package_data=True)
