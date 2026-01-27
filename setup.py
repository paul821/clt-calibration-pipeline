from setuptools import setup, find_packages

setup(
    name="clt-calibration-pipeline",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "pandas>=1.5.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "matplotlib>=3.7.0",
        "sciris>=3.0.0",
    ],
)
