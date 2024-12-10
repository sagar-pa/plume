from setuptools import find_packages, setup

setup(
    name="toy_abr_gym",
    packages=[package for package in find_packages() if package.startswith("toy_abr_gym")],
    include_package_data=True,
    python_requires='>=3.9',
    package_data={"toy_abr_gym": ["trace_data/*/*"]},
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
    ],
    description="A simulation gym training environment for Adaptive Bitrate for controlled, repeatable experiments.",
    author="Sagar Patel",
    url="https://github.com/sagar-pa/plume",
    author_email="sagar.patel@uci.edu",
    license="MIT",
    version=0.20,
)