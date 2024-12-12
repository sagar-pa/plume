from setuptools import find_packages, setup

setup(
    name="abr_gym",
    packages=[package for package in find_packages() if package.startswith("abr_gym")],
    include_package_data=True,
    python_requires='>=3.9', # gym breaking in new python
    package_data={"abr_gym": ["action_lookup.json", "potential_features.json"]},
    install_requires=[
        "gymnasium>=0.26",
        "numpy",
        "pandas",
        "seaborn",
        "tqdm",
        "scipy",
        "statsmodels",
        "pyarrow",
        "requests",
        "ray", # Shared caching, testing, sampling
        "tsfresh", # For smart weighting
        "scikit-learn", 
        "stable-baselines3>=2.0",
        "cython"
    ],
    description="A simulation gym training environment for Adaptive Bitrate using Puffer's public data.",
    author="Sagar Patel",
    url="https://github.com/sagar-pa/plume",
    author_email="sagar.patel@uci.edu",
    license="MIT",
    version=0.18,
)